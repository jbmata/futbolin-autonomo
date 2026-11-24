"""Comprueba que el 3D y el simulador siguen Medidas/mesa.json.

Lee el GLB de Mecanica/3d/futbolin_actualizado (exportado por aplicar_medidas.py, con el recorrido
de cada barra en sus propiedades) y la mesa del simulador (Modelo/world.py), y compara barra a barra:
carrera, posición de cada jugador con la barra en el tope de los motores (s = 0), posición del eje
y que en cada tope el muelle toque la guía. Sale con código 1 si algo no coincide.

Uso (desde la raíz del repositorio, con el entorno de Modelo):
    Modelo/.venv/Scripts/python.exe Medidas/comprobar.py
"""
import os
import sys

import numpy as np
from pygltflib import GLTF2

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GLB = os.path.join(RAIZ, "Mecanica", "3d", "futbolin_actualizado", "futbolin_madrid.glb")
sys.path.insert(0, os.path.join(RAIZ, "Modelo"))
os.environ.setdefault("FUTBOLIN_MESA", "mesa")
import world  # noqa: E402

TOL = 0.5   # mm
ROL = {1: "GK", 2: "DEF", 3: "ATK", 5: "MID"}


def leer_glb(ruta):
    """{nombre de nodo: (padre Barra_* o None, mín, máx en mm, ejes de Blender)} y extras de las barras."""
    g = GLTF2().load(ruta)
    blob = g.binary_blob()

    def acc(i):
        a = g.accessors[i]
        bv = g.bufferViews[a.bufferView]
        n = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a.type]
        return np.frombuffer(blob, np.float32, count=a.count * n,
                             offset=(bv.byteOffset or 0) + (a.byteOffset or 0)).reshape(a.count, n)

    def local(n):
        M = np.eye(4)
        if n.matrix:
            return np.array(n.matrix).reshape(4, 4).T
        if n.rotation:
            x, y, z, w = n.rotation
            M[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
        if n.scale:
            M[:3, :3] = M[:3, :3] @ np.diag(n.scale)
        if n.translation:
            M[:3, 3] = n.translation
        return M

    g2b = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)   # glTF Y arriba → Blender Z arriba
    piezas, extras = {}, {}

    def walk(i, P, barra):
        n = g.nodes[i]
        M = P @ local(n)
        if n.name.startswith("Barra_"):
            barra, extras[n.name] = n.name, (n.extras or {})
        if n.mesh is not None:
            V = np.vstack([acc(p.attributes.POSITION) for p in g.meshes[n.mesh].primitives]).astype(float)
            W = (np.c_[V, np.ones(len(V))] @ (g2b @ M).T)[:, :3] * 1000
            piezas[n.name] = (barra, W.min(0), W.max(0))
        for c in n.children or []:
            walk(c, M, barra)

    for r in g.scenes[g.scene or 0].nodes:
        walk(r, np.eye(4), None)
    return piezas, extras


def main():
    m = world.load_measurements()
    piezas, extras = leer_glb(GLB)
    lo, hi = piezas["Camp"][1], piezas["Camp"][2]
    cy = (lo[1] + hi[1]) / 2
    W, L = hi[1] - lo[1], hi[0] - lo[0]
    filas, mal = [], 0

    def fila(que, cad, sim, medida):
        nonlocal mal
        ok = all(abs(a - b) <= TOL for a, b in ((cad, medida), (sim, medida)))
        mal += not ok
        filas.append((que, cad, sim, medida, "ok" if ok else "DISTINTO"))

    fila("plancha, ancho", W, world.FIELD_WIDTH * 1000, m["width"])
    fila("plancha, largo", L, world.FIELD_LENGTH * 1000, m["length"])
    sim_bars = {(b.team, b.role): b for b in world.create_standard_world().bars}
    for nombre in sorted(extras):
        _, _, equipo, nj = nombre.split("_")
        rol = ROL[int(nj.replace("jug", ""))]
        med = m["roles"][rol]
        ex = extras[nombre]
        carrera_cad = (ex["recorrido_max_m"] - ex["recorrido_min_m"]) * 1000
        b = sim_bars[(equipo, rol)]
        fila(f"{nombre} carrera", carrera_cad, b.linear_range * 2000, med["travel"])
        # jugadores en s = 0: la barra desplazada hacia el lado de sus motores
        lado = -1.0 if equipo == "A" else 1.0
        s0 = lado * carrera_cad / 2                      # desplazamiento de la barra en s = 0 [mm]
        ys = sorted((p[1][1] + p[2][1]) / 2 for n, p in piezas.items() if p[0] == nombre and n.startswith("Player"))
        banda = cy + lado * W / 2
        cad_casa = sorted(abs(y + s0 - banda) for y in ys)
        b.linear_pos = world.linear_pos_from_s(b, 0.0)
        sim_casa = sorted(abs(y * 1000 - lado * world.FIELD_WIDTH * 500) for y in b.get_player_abs_y())
        for k, (c, s, d) in enumerate(zip(cad_casa, sim_casa, med["home"])):
            fila(f"{nombre} jugador {k + 1} a la banda de motores en s=0", c, s, d)
        varilla = [p for n, p in piezas.items() if p[0] == nombre and n.startswith("Metal")][0]
        x = (varilla[1][0] + varilla[2][0]) / 2          # eje de la barra
        fila(f"{nombre} eje x desde el centro", abs(x - (lo[0] + hi[0]) / 2), abs(b.bar_x) * 1000, med["x"])
        # en cada tope el muelle de ese lado toca la guía
        muelles = [p for n, p in piezas.items() if p[0] == nombre and n.startswith("Muelle")]
        guias = [p for n, p in piezas.items() if p[0] is None and n.startswith("Guide")
                 and abs((p[1][0] + p[2][0]) / 2 - x) < 15]
        g_neg = max(p[2][1] for p in guias if p[2][1] < cy and p[2][1] > cy - 420)
        g_pos = min(p[1][1] for p in guias if p[1][1] > cy and p[1][1] < cy + 420)
        m_neg = min(p[1][1] for p in muelles) + ex["recorrido_min_m"] * 1000
        m_pos = max(p[2][1] for p in muelles) + ex["recorrido_max_m"] * 1000
        hueco = max(abs(m_neg - g_neg), abs(m_pos - g_pos))
        filas.append((f"{nombre} muelle contra la guía en los topes", hueco, None, 0.0,
                      "ok" if hueco <= TOL else "DISTINTO"))
        mal += hueco > TOL

    print(f"{'comprobación':58s} {'3D':>9s} {'simulador':>10s} {'mesa.json':>10s}")
    for que, cad, sim, med, est in filas:
        f = lambda x: f"{x:10.2f}" if x is not None else f"{'—':>10s}"
        print(f"{que:58s} {f(cad)[1:]:>9s} {f(sim)} {f(med)}  {est}")
    print(f"\n{len(filas) - mal} de {len(filas)} coinciden (tolerancia {TOL} mm)")
    return 1 if mal else 0


if __name__ == "__main__":
    sys.exit(main())
