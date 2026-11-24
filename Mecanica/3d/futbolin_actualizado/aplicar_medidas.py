"""Aplica Medidas/mesa.json al futbolin_madrid.blend: misma mesa que el simulador.

Para cada barra:
  - desplaza la barra entera (varilla, mango y jugadores) para que, con la barra a mitad de su
    carrera (location.y = 0), los jugadores queden donde los pone mesa.json; los agujeros del plano
    ya están en la barra desde construir_blend.py --madrid y aquí se comprueba que coinciden;
  - quita los topes genéricos del modelo original (Absorber, Washer, Rubber);
  - pone en cada lado el taco de goma y el muelle junto al jugador del extremo (en el portero,
    junto a los pasadores de los agujeros de los extremos, que también se añaden);
  - fija el recorrido (restricción LIMIT_LOCATION y propiedades recorrido_min_m / recorrido_max_m)
    y rehace la animación de ejemplo (fotogramas 1-96) dentro de él.

No mueve la mesa: comprueba que la plancha y la distancia entre guías del .blend coinciden con
mesa.json y, si no, avisa y para (habría que rehacer la mesa).

Uso (Blender 5.3, desde esta carpeta):
    blender --background futbolin_madrid.blend --python aplicar_medidas.py -- [--glb futbolin_madrid.glb]
Guarda el .blend encima. Después, para el STEP:
    blender --background futbolin_madrid.blend --python volcar_blend.py -- piezas.pkl   (ver volcar_blend.py)
    python -I exportar_step.py piezas.pkl futbolin_madrid.step
"""
import json
import math
import os
import sys

import bmesh
import bpy
from mathutils import Vector

AQUI = os.path.dirname(os.path.abspath(__file__))
MEDIDAS = os.path.normpath(os.path.join(AQUI, "..", "..", "..", "Medidas", "mesa.json"))
ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
MM = 0.001
ROL_POR_JUGADORES = {1: "GK", 2: "DEF", 3: "ATK", 5: "MID"}


def arg(nombre, defecto=None):
    return ARGS[ARGS.index(nombre) + 1] if nombre in ARGS else defecto


def valor(d):
    return d["valor"]


def caja(ob):
    """(mín, máx) en coordenadas de mundo [m]."""
    pts = [ob.matrix_world @ Vector(c) for c in ob.bound_box]
    return (Vector([min(p[i] for p in pts) for i in range(3)]),
            Vector([max(p[i] for p in pts) for i in range(3)]))


def centro(ob):
    lo, hi = caja(ob)
    return (lo + hi) / 2


raw = json.load(open(MEDIDAS, encoding="utf-8"))
campo, topes = raw["campo"], raw["topes"]
G = valor(campo["distancia_entre_guias"])
W = valor(campo["ancho_plancha"])
L = valor(campo["largo_plancha"])
GOMA, MUELLE, R_PASADOR = valor(topes["largo_taco_goma"]), valor(topes["largo_muelle"]), valor(topes["radio_pasador"])
MEDIO_FIG = valor(raw["figura"]["ancho_en_barra"]) / 2.0

sc = bpy.context.scene
sc.frame_set(1)
barras = sorted((o for o in bpy.data.objects if o.type == 'EMPTY' and o.name.startswith("Barra_")), key=lambda o: o.name)
for e in barras:                       # pose de reposo para medir (sin que el límite antiguo la mueva)
    e.animation_data_clear()
    for c in e.constraints:
        if c.type == 'LIMIT_LOCATION':
            c.min_y, c.max_y = -1.0, 1.0
    e.location.y = 0.0
    e.rotation_euler[1] = 0.0
bpy.context.view_layer.update()

# --- la mesa del .blend tiene que ser la de mesa.json -------------------------------------
camp = bpy.data.objects["Camp"]
clo, chi = caja(camp)
cx, cy = (clo.x + chi.x) / 2, (clo.y + chi.y) / 2   # centro del campo [m]

# Posición de cada barra a lo largo de la mesa (x_eje_desde_centro): se mueve la barra con sus guías,
# tornillos y agujeros de la pared, que son las piezas fijas pequeñas alineadas con ella.
fijas = [o for o in bpy.data.objects if o.type == 'MESH' and o.parent is None
         and o.name.split("_")[0] in ("Guide", "Screw", "Hole", "Rivet", "Detail")]
for e in barras:
    rol = ROL_POR_JUGADORES[int(e.name.split("_")[3].replace("jug", ""))]
    x_obj = cx + math.copysign(valor(raw["barras"][rol]["x_eje_desde_centro"]) * MM, e.location.x - cx)
    dx = x_obj - e.location.x
    if abs(dx) > 1e-7:
        for o in fijas:
            if abs(centro(o).x - e.location.x) < 0.020:
                o.location.x += dx
        e.location.x = x_obj
    for c in e.constraints:
        if c.type == 'LIMIT_LOCATION':
            c.min_x = c.max_x = x_obj
bpy.context.view_layer.update()
errores = []
if abs((chi.y - clo.y) / MM - W) > 0.5:
    errores.append(f"ancho de la plancha: .blend {(chi.y - clo.y) / MM:.1f} mm, mesa.json {W} mm")
if abs((chi.x - clo.x) / MM - L) > 0.5:
    errores.append(f"largo de la plancha: .blend {(chi.x - clo.x) / MM:.1f} mm, mesa.json {L} mm")
guias = [o for o in bpy.data.objects if o.name.startswith("Guide")]
for e in barras:
    gd = [caja(g) for g in guias if abs(centro(g).x - e.location.x) < 0.015]
    neg = max(hi.y for lo, hi in gd if hi.y < cy and hi.y > cy - (G / 2 + 20) * MM)
    pos = min(lo.y for lo, hi in gd if lo.y > cy and lo.y < cy + (G / 2 + 20) * MM)
    if abs((pos - neg) / MM - G) > 0.5 or abs((pos + neg) / 2 - cy) > 0.5 * MM:
        errores.append(f"{e.name}: guías a {(pos - neg) / MM:.1f} mm (centradas a {((pos + neg) / 2 - cy) / MM:+.1f}),"
                       f" mesa.json {G} mm")
if errores:
    print("\nLA MESA DEL .BLEND NO COINCIDE CON mesa.json; no se cambia nada:\n  " + "\n  ".join(errores))
    sys.exit(1)

# --- mismas cuentas que Modelo/world.py:load_measurements ----------------------------------
guia_banda = (G - W) / 2.0
amort = GOMA + MUELLE


def barra_medida(rol, equipo):
    b = raw["barras"][rol]
    agujeros = valor(b["agujeros_desde_punta_mango"])
    jug = sorted(agujeros[k] for k in valor(b["jugadores_en_agujeros"]))
    pas = []
    if valor(b["tope"]) == "pasadores":
        p0, p1 = sorted(agujeros[k] for k in valor(b["pasadores_en_agujeros"]))
        pila = R_PASADOR + amort
        primero = jug[0] - p0 + pila
        carrera = G - (p1 - p0) - 2 * pila
        pas = [p0, p1]
    else:
        pila = MEDIO_FIG + amort
        primero = pila
        carrera = G - (jug[-1] - jug[0]) - 2 * pila
    # distancia desde la banda de los motores en s = 0 → y desde el centro a mitad de carrera
    lado = -1.0 if equipo == 'A' else 1.0          # lado de los motores (y del mango)
    casa = [primero - guia_banda + (p - jug[0]) for p in jug]
    off = [lado * (W / 2 - d - carrera / 2) for d in casa]
    # pasadores: a la misma distancia del portero que en el plano, hacia el mango y hacia el otro lado
    off_pas = [off[0] + lado * (jug[0] - p) for p in pas]
    return dict(carrera=carrera, off=sorted(off), pasadores=sorted(off_pas), agujeros=jug, lado=lado)


# --- materiales -----------------------------------------------------------------------------
def material(nombre, color, metal=0.0, rough=0.5):
    m = bpy.data.materials.get(nombre) or bpy.data.materials.new(nombre)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Metallic"].default_value = metal
    bsdf.inputs["Roughness"].default_value = rough
    return m


MAT_GOMA = bpy.data.materials.get("Goma_negra") or material("Goma_negra", (0.02, 0.02, 0.02))
MAT_MUELLE = material("Muelle", (0.85, 0.65, 0.15), 1.0, 0.3)
MAT_PASADOR = bpy.data.materials.get("Cromo") or material("Cromo", (0.85, 0.85, 0.88), 1.0, 0.18)


def colgar(ob, e, mat):
    # solo triángulos, como el resto del modelo (volcar_blend.py y exportar_step.py lo esperan)
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.to_mesh(ob.data)
    bm.free()
    ob.data.materials.append(mat)
    for c in e.users_collection:
        c.objects.link(ob)
    ob.parent = e
    ob.matrix_parent_inverse = e.matrix_world.inverted()


def cilindro_y(nombre, x, y0, y1, z, diam, e, mat):
    """Cilindro a lo largo de y entre y0 e y1 [m]."""
    me = bpy.data.meshes.new(nombre)
    n, r = 24, diam / 2
    verts, caras = [], []
    for y in (y0, y1):
        verts += [(x + r * math.cos(2 * math.pi * k / n), y, z + r * math.sin(2 * math.pi * k / n)) for k in range(n)]
    caras += [(k, (k + 1) % n, n + (k + 1) % n, n + k) for k in range(n)]
    caras += [tuple(range(n))[::-1], tuple(range(n, 2 * n))]
    me.from_pydata(verts, [], caras)
    me.update()
    ob = bpy.data.objects.new(nombre, me)
    colgar(ob, e, mat)
    return ob


def cilindro_x(nombre, x, y, z, diam, largo, e, mat):
    """Cilindro a lo largo de x (pasador que atraviesa la barra)."""
    me = bpy.data.meshes.new(nombre)
    n, r = 16, diam / 2
    verts = []
    for xx in (x - largo / 2, x + largo / 2):
        verts += [(xx, y + r * math.cos(2 * math.pi * k / n), z + r * math.sin(2 * math.pi * k / n)) for k in range(n)]
    caras = [(k, (k + 1) % n, n + (k + 1) % n, n + k) for k in range(n)] + [tuple(range(n))[::-1], tuple(range(n, 2 * n))]
    me.from_pydata(verts, [], caras)
    me.update()
    ob = bpy.data.objects.new(nombre, me)
    colgar(ob, e, mat)
    return ob


def muelle(nombre, x, y0, y1, z, diam, e, hilo=1.6 * MM, vueltas=5, por_vuelta=10):
    """Muelle helicoidal a lo largo de y entre y0 e y1 [m], como malla ligera (el STEP la facetea)."""
    cu = bpy.data.curves.new(nombre, 'CURVE')
    cu.dimensions = '3D'
    cu.bevel_depth = hilo / 2
    cu.bevel_resolution = 0                            # alambre de sección cuadrada: pocas caras
    sp = cu.splines.new('POLY')
    pasos = vueltas * por_vuelta
    sp.points.add(pasos)
    r = diam / 2 - hilo / 2
    a0, a1 = y0 + hilo / 2, y1 - hilo / 2              # el alambre no sobresale de [y0, y1]
    for k in range(pasos + 1):
        a = 2 * math.pi * k / por_vuelta
        sp.points[k].co = (x + r * math.cos(a), a0 + (a1 - a0) * k / pasos, z + r * math.sin(a), 1)
    tmp = bpy.data.objects.new(nombre + "_curva", cu)
    sc.collection.objects.link(tmp)
    bpy.context.view_layer.update()
    me = bpy.data.meshes.new_from_object(tmp.evaluated_get(bpy.context.evaluated_depsgraph_get()))
    bpy.data.objects.remove(tmp)
    bpy.data.curves.remove(cu)
    me.name = nombre
    ob = bpy.data.objects.new(nombre, me)
    colgar(ob, e, MAT_MUELLE)
    return ob


# --- cada barra -------------------------------------------------------------------------------
resumen = []
for e in barras:
    _, num, equipo, nj = e.name.split("_")
    rol = ROL_POR_JUGADORES[int(nj.replace("jug", ""))]
    m = barra_medida(rol, equipo)
    hijos = list(e.children)
    # 1. quitar los topes genéricos
    for h in [h for h in hijos if h.name.split("_")[0] in ("Absorber", "Washer", "Rubber")
              or h.name.split("_")[0] in ("Goma", "Muelle", "Pasador")]:
        bpy.data.objects.remove(h, do_unlink=True)
    hijos = list(e.children)
    # 2. desplazar la barra para que los jugadores queden donde dice mesa.json
    ys = sorted(centro(h).y for h in hijos if h.name.startswith("Player"))
    objetivo = [cy + o * MM for o in m["off"]]
    sep_blend = [round((b - a) / MM, 1) for a, b in zip(ys, ys[1:])]
    sep_mesa = [round(b - a, 1) for a, b in zip(m["off"], m["off"][1:])]
    if any(abs(a - b) > 0.5 for a, b in zip(sep_blend, sep_mesa)):
        print(f"{e.name}: separación entre jugadores del .blend {sep_blend} distinta de mesa.json {sep_mesa}")
        sys.exit(1)
    delta = sum(objetivo) / len(objetivo) - sum(ys) / len(ys)
    for h in hijos:
        h.location.y += delta
    bpy.context.view_layer.update()
    # 3. topes: taco de goma y muelle junto al jugador del extremo (o al pasador) en cada lado
    xr, zr = e.location.x, e.location.z
    if m["pasadores"]:
        for k, p in enumerate(m["pasadores"]):
            cilindro_x(f"Pasador_{num}_{k}", xr, cy + p * MM, zr, 2 * R_PASADOR * MM, 26 * MM, e, MAT_PASADOR)
        bajo, alto = m["pasadores"][0] - R_PASADOR, m["pasadores"][-1] + R_PASADOR
    else:
        bajo, alto = m["off"][0] - MEDIO_FIG, m["off"][-1] + MEDIO_FIG
    for lado, ancla, s in (("bajo", bajo, -1), ("alto", alto, 1)):
        g0, g1 = sorted((ancla, ancla + s * GOMA))
        u0, u1 = sorted((ancla + s * GOMA, ancla + s * (GOMA + MUELLE)))
        cilindro_y(f"Goma_{num}_{lado}", xr, cy + g0 * MM, cy + g1 * MM, zr, 24 * MM, e, MAT_GOMA)
        muelle(f"Muelle_{num}_{lado}", xr, cy + u0 * MM, cy + u1 * MM, zr, 20 * MM, e)
    # 4. recorrido: ±carrera/2 alrededor de la pose de reposo (mitad de la carrera)
    R = m["carrera"] / 2 * MM
    e["recorrido_min_m"], e["recorrido_max_m"] = -R, R
    e["carrera_mm"] = m["carrera"]
    e["lado_motores"] = "-y" if m["lado"] < 0 else "+y"
    for c in [c for c in e.constraints if c.type == 'LIMIT_LOCATION']:
        c.min_y, c.max_y = -R, R
    # comprobación: en cada tope el muelle toca la guía
    assert abs((cy + (bajo - GOMA - MUELLE) * MM - R) - (cy - G / 2 * MM)) < 1e-6
    assert abs((cy + (alto + GOMA + MUELLE) * MM + R) - (cy + G / 2 * MM)) < 1e-6
    resumen.append((e.name, rol, round(m["carrera"], 2), [round(o, 1) for o in m["off"]], round(delta / MM, 1)))

# --- animación de ejemplo dentro del nuevo recorrido (como construir_blend.py) -----------------
ej = {1: (0.0, 0.5), 2: (1.0, 0.8), 3: (2.0, 0.2), 4: (1.0, 0.5), 5: (0.0, 0.6), 6: (2.0, 0.4), 7: (1.0, 0.3), 8: (0.0, 0.5)}
for k, e in enumerate(barras, 1):
    vueltas, frac = ej[k]
    lo, hi = e["recorrido_min_m"], e["recorrido_max_m"]
    for fr, (ang, y) in ((1, (0.0, 0.0)), (48, (vueltas * math.pi, lo + (hi - lo) * frac)), (96, (vueltas * 2 * math.pi, 0.0))):
        e.rotation_euler[1] = ang
        e.location.y = y
        e.keyframe_insert("rotation_euler", index=1, frame=fr)
        e.keyframe_insert("location", index=1, frame=fr)
    e.rotation_euler[1] = 0.0
    e.location.y = 0.0
sc.frame_set(1)

bpy.ops.wm.save_mainfile()
print("\nmesa.json aplicado (barra, rol, carrera mm, jugadores a mitad de carrera mm desde el centro, desplazamiento mm):")
for r in resumen:
    print("  ", *r)
if arg("--glb"):
    bpy.ops.export_scene.gltf(filepath=os.path.join(AQUI, arg("--glb")), export_format='GLB',
                              export_animations=True, export_apply=False, export_extras=True)
    print("glb", arg("--glb"))
