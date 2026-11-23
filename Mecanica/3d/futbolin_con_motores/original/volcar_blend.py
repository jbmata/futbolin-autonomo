"""Vuelca las mallas de un .blend (en mm, coordenadas de mundo, pose de reposo) a un .pkl.

Paso 1 para generar el STEP con exactamente las mismas medidas que el .blend.
Usa el modulo 'bpy' (numpy<2). El paso 2 es exportar_step.py (build123d).

Uso:  python -I volcar_blend.py futbolin_madrid.blend piezas.pkl
Cada pieza: {name, grupo ('Mesa' o nombre de la barra), color (r,g,b), V (mm), F}
"""
import pickle
import sys

import bpy
import numpy as np

bpy.ops.wm.open_mainfile(filepath=sys.argv[1])
sc = bpy.context.scene
sc.frame_set(1)
dg = bpy.context.evaluated_depsgraph_get()

piezas = []
for ob in bpy.data.objects:
    if ob.type != 'MESH' or ob.name == 'Suelo':
        continue
    ev = ob.evaluated_get(dg)
    M = np.array(ev.matrix_world)
    V = np.array([v.co[:] for v in ob.data.vertices], dtype=float)
    V = (np.c_[V, np.ones(len(V))] @ M.T)[:, :3] * 1000.0   # m -> mm
    F = np.array([p.vertices[:] for p in ob.data.polygons], dtype=int)
    col = (0.5, 0.5, 0.5)
    if ob.data.materials and ob.data.materials[0].use_nodes:
        col = tuple(ob.data.materials[0].node_tree.nodes['Principled BSDF']
                    .inputs['Base Color'].default_value[:3])
    piezas.append(dict(name=ob.name, grupo=ob.parent.name if ob.parent else 'Mesa',
                       color=col, V=V, F=F))
pickle.dump(piezas, open(sys.argv[2], 'wb'))
print(len(piezas), 'piezas,', sum(len(p['F']) for p in piezas), 'caras ->', sys.argv[2])
