"""Genera un STEP (ensamblaje con nombres y colores) a partir de piezas.pkl.

Paso 2 tras volcar_blend.py. Cada malla triangulada del .blend se convierte en un solido
B-rep (triangulos cosidos y caras coplanares unidas con UnifySameDomain), asi que las
medidas son exactamente las del .blend. Ojo: las superficies curvas (jugadores, mangos,
barras) quedan facetadas, no son cilindros/NURBS reales.

Las piezas identicas (jugadores, topes, tornillos...) se exportan como instancias de una
sola geometria, asi el fichero pesa ~5 veces menos.
Estructura: Futbolin > Mesa | Barra_XX_... > pieza. Unidades: mm, Z arriba.
Uso:  python -I exportar_step.py piezas.pkl futbolin_madrid.step
Requiere build123d (pip install build123d), en un entorno aparte de bpy (numpy 2).
"""
import pickle
import sys
import time

import numpy as np
from build123d import Color, Compound, export_step
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon,
                                BRepBuilderAPI_MakeSolid, BRepBuilderAPI_Sewing)
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.ShapeFix import ShapeFix_Solid
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.TopAbs import TopAbs_SHELL
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Compound
from OCP.gp import gp_Pnt


def srgb(c):
    c = max(0.0, min(1.0, c))
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def tapar_agujeros_planos(V, F, tol_plano=1e-3):
    """Cierra con una tapa (abanico desde el centro) los anillos de aristas libres que son
    planos. Hace falta p.ej. en los mangos, cuya malla del .max tiene el extremo abierto
    (la parte que va encajada en la barra); asi se pueden exportar como solido."""
    uso = {}
    for a, b, c in F:
        for u, v in ((a, b), (b, c), (c, a)):
            uso.setdefault((min(u, v), max(u, v)), []).append((u, v))
    libres = {}
    for k, lst in uso.items():
        if len(lst) == 1:
            u, v = lst[0]
            libres[u] = v
    nuevos, extra = [], []
    vistos = set()
    for ini in list(libres):
        if ini in vistos:
            continue
        lazo, u = [], ini
        while u in libres and u not in vistos:
            vistos.add(u)
            lazo.append(u)
            u = libres[u]
        if u != ini or len(lazo) < 3:
            continue   # no es un anillo cerrado simple
        pts = V[lazo]
        c = pts.mean(0)
        if np.linalg.svd(pts - c, compute_uv=False)[-1] > tol_plano * np.sqrt(len(lazo)):
            continue   # el anillo no es plano: no se tapa
        ic = len(V) + len(extra)
        extra.append(c)
        for i, u in enumerate(lazo):
            v = lazo[(i + 1) % len(lazo)]
            nuevos.append((v, u, ic))
    if not nuevos:
        return V, F
    return np.vstack([V, np.array(extra)]), np.vstack([F, np.array(nuevos)])


def a_solido(V, F):
    """Triangulos -> solido (o shell abierto si la malla no es cerrada). Devuelve (shape, cerrado)."""
    # soldar vertices duplicados (0,1 micras) y quitar triangulos degenerados
    clave = np.round(V / 1e-4).astype(np.int64)
    _, idx, inv = np.unique(clave, axis=0, return_index=True, return_inverse=True)
    V = V[idx]
    F = inv.reshape(-1)[F]
    ok = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
    F = F[ok]
    V, F = tapar_agujeros_planos(V, F)
    P = [gp_Pnt(*map(float, v)) for v in V]
    sew = BRepBuilderAPI_Sewing(1e-3)
    for a, b, c in F:
        mp = BRepBuilderAPI_MakePolygon(P[a], P[b], P[c], True)
        if not mp.IsDone():
            continue
        mf = BRepBuilderAPI_MakeFace(mp.Wire(), True)
        if mf.IsDone():
            sew.Add(mf.Face())
    sew.Perform()
    shape = sew.SewedShape()
    # una malla puede tener varios trozos sueltos (patas, aros...): un solido por cada shell
    # cerrado y el shell tal cual si esta abierto
    partes, cerrado = [], True
    exp = TopExp_Explorer(shape, TopAbs_SHELL)
    while exp.More():
        sh = TopoDS.Shell(exp.Current())
        if BRep_Tool.IsClosed_s(sh):
            fix = ShapeFix_Solid(BRepBuilderAPI_MakeSolid(sh).Solid())
            fix.Perform()
            sol = fix.Solid()
            props = GProp_GProps()
            BRepGProp.VolumeProperties_s(sol, props)
            if props.Mass() < 0:   # orientacion invertida
                sol = TopoDS.Solid(sol.Reversed())
            partes.append(sol)
        else:
            cerrado = False
            partes.append(sh)
        exp.Next()
    if not partes:
        return shape, False
    if len(partes) == 1:
        shape = partes[0]
    else:
        shape = TopoDS_Compound()
        BRep_Builder().MakeCompound(shape)
        for q in partes:
            BRep_Builder().Add(shape, q)
    uni = ShapeUpgrade_UnifySameDomain(shape, True, True, True)
    uni.Build()
    return uni.Shape(), cerrado


def agrupar_instancias(piezas, tol=2e-3):
    """Detecta piezas identicas (misma malla, solo trasladada o girada 180 grados en Z).

    Devuelve una lista de (pieza, indice_de_base, rotada) donde indice_de_base apunta a la
    primera pieza con esa geometria (la que se construye una sola vez)."""
    bases = []   # (F, centrado) de cada geometria unica
    res = []
    for p in piezas:
        C = p['V'] - p['V'].mean(0)
        hallada = None
        for j, (F0, C0) in enumerate(bases):
            if F0.shape != p['F'].shape or not np.array_equal(F0, p['F']):
                continue
            if np.abs(C - C0).max() < tol:
                hallada = (j, False)
                break
            if np.abs(C * np.array([-1, -1, 1]) - C0).max() < tol:
                hallada = (j, True)
                break
        if hallada is None:
            bases.append((p['F'], C))
            hallada = (len(bases) - 1, False)
        res.append((p, hallada[0], hallada[1]))
    return res, len(bases)


if __name__ == '__main__':
    from build123d import Location, Shape
    piezas = pickle.load(open(sys.argv[1], 'rb'))
    t0 = time.time()
    plan, n_unicas = agrupar_instancias(piezas)
    print(f'{len(piezas)} piezas, {n_unicas} geometrias unicas', flush=True)
    construidas = {}
    grupos = {}
    abiertas = []
    for i, (p, j, rotada) in enumerate(plan, 1):
        if j not in construidas:
            shape, cerrado = a_solido(p['V'] - p['V'].mean(0), p['F'])
            if not cerrado:
                abiertas.append(p['name'])
            construidas[j] = (Shape.cast(shape), rotada)
            print(f'  geometria {len(construidas)}/{n_unicas} ({p["name"]}, {len(p["F"])} tri) '
                  f'{time.time() - t0:.0f} s', flush=True)
        base, base_rotada = construidas[j]
        # la base se construyo con la orientacion de su primera pieza
        giro = 180 if rotada != base_rotada else 0
        s = base.moved(Location(tuple(float(x) for x in p['V'].mean(0)), (0, 0, giro)))
        s.label = p['name']
        s.color = Color(*[srgb(c) for c in p['color']])
        grupos.setdefault(p['grupo'], []).append(s)
    hijos = [Compound(children=grupos[g], label=g)
             for g in sorted(grupos, key=lambda g: (g != 'Mesa', g))]
    top = Compound(children=hijos, label='Futbolin')
    export_step(top, sys.argv[2])
    print('STEP escrito:', sys.argv[2], f'({time.time() - t0:.0f} s)')
    print('mallas abiertas (exportadas como shell):', abiertas)
