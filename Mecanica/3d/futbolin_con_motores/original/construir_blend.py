"""Construye un .blend del futbolin con las 8 barras articuladas.

Cada barra es un Empty "Barra_XX_..." situado sobre el eje de la barra
(paralelo a Y, a la altura del eje). Todas las piezas de esa barra (jugadores,
mango, arandelas, topes de goma...) cuelgan de el:
  - deslizar la barra : location.Y   (limitado por constraint a su recorrido real,
                        calculado con el choque tope de goma <-> guia)
  - girar la barra    : rotation_euler.Y   (radianes; 0 = pose del modelo original)
Al incluir una animacion de ejemplo (fotogramas 1-96) se ven ambos movimientos.

Uso (con el modulo 'bpy' de pip, Python 3.11). 'partes.pkl' sale de la conversion del
.max:  python -I ../futbolin3d/extraer_max.py ../futbolin3d/Foosball.max... partes.pkl
    python -I construir_blend.py partes.pkl futbolin_madrid.blend --madrid --glb futbolin_madrid.glb
Opciones: --glb f.glb | --render f.png | --render-pose f.png | --madrid | --sin-largos | --samples N
    --madrid: jugadores en los agujeros del plano de futbolinmadrid.com (mismo
              reparto 1-2-3-5 del .max), barras de 14 mm en vez de 12,5 mm y plancha
              del campo de 115 x 68 cm (en vez de 113 x 79) con bandas hasta la pared.
              Tambien el largo de barra del plano: portero 110, defensa 124, delantero 115,
              media 105 cm (con --sin-largos se mantienen los 108,8 cm del .max).
Unidades de la escena: metros (el modelo original esta en cm).
"""
import math
import pickle
import sys

import bpy
import numpy as np
from mathutils import Vector

ESC = 0.01  # cm -> m


def arg(nombre, defecto=None):
    return sys.argv[sys.argv.index(nombre) + 1] if nombre in sys.argv else defecto


partes = pickle.load(open(sys.argv[1], 'rb'))
salida = sys.argv[2]

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.unit_settings.system = 'METRIC'
sc.unit_settings.scale_length = 1.0


# ---------------------------------------------------------------- materiales
def material(nombre, color, metal=0.0, rough=0.5):
    m = bpy.data.materials.new(nombre)
    m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value = (*color, 1)
    b.inputs['Metallic'].default_value = metal
    b.inputs['Roughness'].default_value = rough
    return m


MAT = {
    'madera': material('Madera', (0.36, 0.20, 0.09), 0.0, 0.45),
    'campo': material('Campo', (0.05, 0.32, 0.10), 0.0, 0.6),
    'cromo': material('Cromo', (0.85, 0.85, 0.88), 1.0, 0.18),
    'negro': material('Goma_negra', (0.02, 0.02, 0.02), 0.0, 0.5),
    'azul': material('Jugador_azul', (0.05, 0.20, 0.85), 0.0, 0.35),
    'rojo': material('Jugador_rojo', (0.85, 0.05, 0.05), 0.0, 0.35),
    'bola': material('Bola', (0.95, 0.95, 0.95), 0.0, 0.3),
    'suelo': material('Suelo', (0.55, 0.55, 0.57), 0.0, 0.8),
}


# Reparto de equipos igual que Modelo/world.py (BAR_DEFINITIONS), de izquierda a derecha:
# GK-A | DEF-A | ATK-B | MID-A | MID-B | ATK-A | DEF-B | GK-B
EQUIPO_BARRA = ['A', 'A', 'B', 'A', 'B', 'A', 'B', 'B']


def mat_de(nombre, equipo=None):
    n = nombre.split('_')[0]
    if n == 'Player':
        return MAT['azul'] if equipo == 'A' else MAT['rojo']
    return MAT[{'Table': 'madera', 'Miter': 'madera', 'Banda': 'madera', 'Camp': 'campo', 'Hole': 'negro',
                'ball': 'bola', 'Handle': 'negro', 'Rubber': 'negro', 'Absorber': 'negro',
                'Detail': 'negro'}.get(n, 'cromo')]


# ----------------------------------------------------------- geometria / ejes
def centro(p):
    return (p['V'].min(0) + p['V'].max(0)) / 2


barras = {}
for p in partes:
    if p['group'].startswith('Group0'):
        barras.setdefault(p['group'], []).append(p)

info = []
for g, ps in barras.items():
    m = next(p for p in ps if p['name'].startswith('Metal'))
    c = centro(m)
    info.append((c[0], c[2], g))
info.sort()

guias = [p for p in partes if p['name'].startswith('Guide')]
col_mesa = bpy.data.collections.new('Mesa')
col_barras = bpy.data.collections.new('Barras')
sc.collection.children.link(col_mesa)
sc.collection.children.link(col_barras)

objetos_malla = []


def crear_malla(p, coleccion, equipo=None):
    c = centro(p)
    me = bpy.data.meshes.new(p['name'])
    me.from_pydata(((p['V'] - c) * ESC).tolist(), [], p['F'].tolist())
    me.validate()
    me.update()
    ob = bpy.data.objects.new(p['name'], me)
    ob.location = c * ESC
    ob.data.materials.append(mat_de(p['name'], equipo))
    coleccion.objects.link(ob)
    objetos_malla.append(ob)
    return ob


def campo_madrid(lista, largo=115.0, ancho=68.0):
    """Plancha del campo de 115 x 68 cm (futbolinmadrid.com: 'las bandas cubren el resto
    hasta la pared'). En el .max la plancha mide 113 x 79 cm; se reduce a 115 x 68 y el
    espacio que queda hasta las paredes laterales se cubre con dos bandas de madera."""
    out = []
    for p in lista:
        if p['name'] != 'Camp':
            out.append(p)
            continue
        V = p['V']
        cx = (V[:, 0].min() + V[:, 0].max()) / 2
        x = cx + (V[:, 0] - cx) * (largo / np.ptp(V[:, 0]))
        pared_neg = max(q['V'][:, 1].max() for q in lista if q['name'] == 'Table_008')
        pared_pos = min(q['V'][:, 1].min() for q in lista if q['name'] == 'Table_009')
        cy = (pared_neg + pared_pos) / 2          # eje de simetria entre las paredes
        ymin, ymax = V[:, 1].min(), V[:, 1].max()
        mid = (ymin + ymax) / 2
        def caja(y_neg, y_pos):
            W = V.copy()
            W[:, 0] = x
            W[:, 1] = np.where(V[:, 1] < mid, y_neg, y_pos)
            return W
        out.append(dict(p, V=caja(cy - ancho / 2, cy + ancho / 2)))
        out.append(dict(p, name='Banda_01', V=caja(ymin, cy - ancho / 2)))
        out.append(dict(p, name='Banda_02', V=caja(cy + ancho / 2, ymax)))
    return out


# mesa y guias (piezas fijas)
fijas = campo_madrid(partes) if '--madrid' in sys.argv else partes
for p in fijas:
    if not p['group'].startswith('Group0'):
        crear_malla(p, col_mesa)

# Perfil "--madrid": posicion de los AGUJEROS de cada barra segun el plano de
# futbolinmadrid.com ("Medidas de las barras de un futbolin Madrid"), en cm, medidas
# desde la punta del mango. Los jugadores se colocan en esos agujeros; el reparto de
# jugadores por barra (1, 2, 3, 5) es el del .max: el portero lleva uno solo, en el
# agujero central. Ademas la barra pasa a 14 mm de diametro.
# Tambien aplica el largo total de cada barra del plano (110/124/115/105 cm);
# con --sin-largos se mantienen los 108,8 cm del .max.
MADRID = '--madrid' in sys.argv
LARGOS = MADRID and '--sin-largos' not in sys.argv   # largos 110/124/115/105 cm del plano
DIAM_BARRA = 1.4  # cm
AGUJEROS = {   # jugadores en el .max -> (largo de barra del plano, agujeros, indices usados)
    1: (110.0, [40.0, 60.2, 80.4], [1]),                      # portero: 3 agujeros, 1 jugador
    2: (124.0, [58.0, 81.0], [0, 1]),                         # defensa
    3: (115.0, [47.0, 64.2, 81.4], [0, 1, 2]),                # delantero
    5: (105.0, [35.0, 46.5, 58.0, 69.5, 81.0], [0, 1, 2, 3, 4]),  # media
}


def perfil_madrid(ps):
    h = next(p for p in ps if p['name'].startswith('Handle'))
    m = next(p for p in ps if p['name'].startswith('Metal'))
    s = 1 if h['V'][:, 1].mean() < m['V'][:, 1].mean() else -1   # sentido punta -> extremo
    tip = h['V'][:, 1].min() if s == 1 else h['V'][:, 1].max()
    largo = abs((m['V'][:, 1].max() if s == 1 else m['V'][:, 1].min()) - tip)
    pl = sorted((p for p in ps if p['name'].startswith('Player')),
                key=lambda p: s * (centro(p)[1] - tip))
    L, agujeros, usados = AGUJEROS[len(pl)]
    nuevos = []
    for p in ps:
        if p['name'].startswith('Player'):
            continue
        q = dict(p)
        if p is m:
            V = p['V'].copy()
            if LARGOS:  # alarga o acorta solo el extremo lejano
                V[s * (V[:, 1] - tip) > largo / 2, 1] += s * (L - largo)
            for ax in (0, 2):  # diametro 14 mm (en el .max son 12,5 mm), centrado en el eje
                c0 = (V[:, ax].min() + V[:, ax].max()) / 2
                V[:, ax] = c0 + (V[:, ax] - c0) * (DIAM_BARRA / np.ptp(V[:, ax]))
            q['V'] = V
        nuevos.append(q)
    for src, idx in zip(pl, usados):
        V = src['V'].copy()
        V[:, 1] += (tip + s * agujeros[idx]) - centro(src)[1]
        nuevos.append(dict(src, V=V))
    return nuevos


# barras articuladas
empties = []
for k, (xr, zr, g) in enumerate(info, 1):
    ps = barras[g]
    if MADRID:
        ps = perfil_madrid(ps)
    jug = [p for p in ps if p['name'].startswith('Player')]
    equipo = EQUIPO_BARRA[k - 1]
    e = bpy.data.objects.new(f'Barra_{k:02d}_{equipo}_{len(jug)}jug', None)
    e['equipo'] = equipo
    e.empty_display_type = 'PLAIN_AXES'
    e.empty_display_size = 0.05
    e.location = (xr * ESC, 0.0, zr * ESC)
    col_barras.objects.link(e)
    empties.append((e, ps))
bpy.context.view_layer.update()

for e, ps in empties:
    inv = e.matrix_world.inverted()
    for p in ps:
        ob = crear_malla(p, col_barras, e['equipo'])
        ob.parent = e
        ob.matrix_parent_inverse = inv

    # recorrido: tope de goma contra guia
    xr = e.location.x / ESC
    ab = [p for p in ps if p['name'].startswith('Absorber')]
    gd = [q for q in guias if abs(centro(q)[0] - xr) < 1.5]
    ap = [p for p in ab if p['V'][:, 1].mean() > -1.7]
    an = [p for p in ab if p['V'][:, 1].mean() <= -1.7]
    hi = min(q['V'][:, 1].min() for q in gd if q['V'][:, 1].min() > 0) - max(p['V'][:, 1].max() for p in ap)
    lo = max(q['V'][:, 1].max() for q in gd if q['V'][:, 1].max() < -1.7) - min(p['V'][:, 1].min() for p in an)
    e['recorrido_min_m'] = lo * ESC
    e['recorrido_max_m'] = hi * ESC
    c = e.constraints.new('LIMIT_LOCATION')
    c.use_min_y = c.use_max_y = True
    c.min_y, c.max_y = lo * ESC, hi * ESC
    c.use_min_x = c.use_max_x = c.use_min_z = c.use_max_z = True
    c.min_x = c.max_x = e.location.x
    c.min_z = c.max_z = e.location.z
    c.owner_space = 'WORLD'
    e.lock_rotation[0] = e.lock_rotation[2] = True
    e.rotation_mode = 'XYZ'

# suavizado por angulo
bpy.ops.object.select_all(action='DESELECT')
for ob in objetos_malla:
    ob.select_set(True)
bpy.context.view_layer.objects.active = objetos_malla[0]
bpy.ops.object.shade_smooth_by_angle(angle=math.radians(35))

# --------------------------------------------------------------- escena
suelo = bpy.data.objects.new('Suelo', bpy.data.meshes.new('Suelo'))
suelo.data.from_pydata([(-6, -6, 0), (6, -6, 0), (6, 6, 0), (-6, 6, 0)], [], [(0, 1, 2, 3)])
suelo.data.materials.append(MAT['suelo'])
sc.collection.objects.link(suelo)

objetivo = bpy.data.objects.new('Objetivo_camara', None)
objetivo.location = (0, -0.017, 0.75)
sc.collection.objects.link(objetivo)
cam = bpy.data.objects.new('Camara', bpy.data.cameras.new('Camara'))
cam.location = (1.7, -2.0, 1.9)
cam.data.lens = 40
tr = cam.constraints.new('TRACK_TO')
tr.target, tr.track_axis, tr.up_axis = objetivo, 'TRACK_NEGATIVE_Z', 'UP_Y'
sc.collection.objects.link(cam)
sc.camera = cam
for nombre, loc, pot in (('Luz_principal', (0.8, -1.2, 3.0), 600), ('Luz_relleno', (-1.8, 1.5, 2.2), 250)):
    l = bpy.data.objects.new(nombre, bpy.data.lights.new(nombre, 'AREA'))
    l.data.energy = pot
    l.data.size = 1.5
    l.location = loc
    tl = l.constraints.new('TRACK_TO')
    tl.target, tl.track_axis, tl.up_axis = objetivo, 'TRACK_NEGATIVE_Z', 'UP_Y'
    sc.collection.objects.link(l)
sc.world = bpy.data.worlds.new('Mundo')
sc.world.use_nodes = True
sc.world.node_tree.nodes['Background'].inputs[0].default_value = (0.8, 0.82, 0.85, 1)
sc.world.node_tree.nodes['Background'].inputs[1].default_value = 0.6

# ------------------------------------------------------ animacion de ejemplo
# (borrala si no la quieres: Barras > Action Editor)
ej = {  # nombre de barra -> (vueltas de giro, desplazamiento 0..1 del recorrido)
    1: (0.0, 0.5), 2: (1.0, 0.8), 3: (2.0, 0.2), 4: (1.0, 0.5),
    5: (0.0, 0.6), 6: (2.0, 0.4), 7: (1.0, 0.3), 8: (0.0, 0.5)}
for k, (e, _) in enumerate(empties, 1):
    vueltas, frac = ej[k]
    lo, hi = e['recorrido_min_m'], e['recorrido_max_m']
    for fr, (ang, y) in ((1, (0.0, 0.0)), (48, (vueltas * 2 * math.pi * 0.5, lo + (hi - lo) * frac)),
                         (96, (vueltas * 2 * math.pi, 0.0))):
        e.rotation_euler[1] = ang
        e.location.y = y
        e.keyframe_insert('rotation_euler', index=1, frame=fr)
        e.keyframe_insert('location', index=1, frame=fr)
    e.rotation_euler[1] = 0.0
    e.location.y = 0.0
sc.frame_start, sc.frame_end = 1, 96
sc.frame_set(1)

bpy.ops.wm.save_as_mainfile(filepath=salida)
print('guardado', salida)

if arg('--glb'):
    bpy.ops.export_scene.gltf(filepath=arg('--glb'), export_format='GLB', export_animations=True,
                              export_apply=False)
    print('glb', arg('--glb'))


def render(ruta, pose=False):
    if pose:  # pose estatica de ejemplo: giro 40 grados y deslizamiento a un extremo
        sc.frame_set(1)
        for k, (e, _) in enumerate(empties, 1):
            e.animation_data_clear()
            e.rotation_euler[1] = math.radians((-40, 25, 0, 70, -70, 20, 45, 0)[k - 1])
            e.location.y = e['recorrido_max_m'] if k % 2 else e['recorrido_min_m']
    sc.render.engine = 'CYCLES'
    sc.cycles.device = 'CPU'
    sc.cycles.samples = int(arg('--samples', 48))
    sc.cycles.use_denoising = True
    sc.render.resolution_x, sc.render.resolution_y = 1280, 720
    sc.render.filepath = ruta
    bpy.ops.render.render(write_still=True)
    print('render', ruta)


if arg('--render'):
    render(arg('--render'))
if arg('--render-pose'):
    render(arg('--render-pose'), pose=True)
