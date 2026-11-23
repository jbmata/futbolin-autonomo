"""Anima el futbolin existente sin agregar piezas. Ejecutar con Blender --background --python.

Guarda un .blend independiente y una vista previa. Para renderizar el MP4:
    blender --background futbolin_animado.blend --render-anim
"""
import math
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

SALIDA = Path(__file__).resolve().parent
ORIGEN = SALIDA.parent / 'futbolin_madrid.blend'
FPS = 24
FOTOGRAMAS = 288

bpy.ops.wm.open_mainfile(filepath=str(ORIGEN))
escena = bpy.context.scene
barras = sorted((o for o in escena.objects if o.name.startswith('Barra_')), key=lambda o: o.name)
mallas = [o for o in escena.objects if o.type == 'MESH' and o.name != 'Suelo']
nombres_originales = {o.name for o in escena.objects}
assert len(barras) == 8

escena.frame_start, escena.frame_end = 1, FOTOGRAMAS
escena.render.fps, escena.render.fps_base = FPS, 1.0
escena.render.engine = 'BLENDER_EEVEE'
escena.eevee.taa_render_samples = 16
escena.render.resolution_x, escena.render.resolution_y = 1280, 720
escena.render.resolution_percentage = 100
escena.render.film_transparent = False

for indice, barra in enumerate(barras):
    barra.animation_data_clear()
    minimo, maximo = barra['recorrido_min_m'], barra['recorrido_max_m']
    assert minimo < maximo
    fase = indice * .73
    for fotograma in range(1, FOTOGRAMAS + 1):
        t = (fotograma - 1) / FPS
        # Recorrido limitado al 80% del rango para dejar margen respecto a los topes.
        barra.location.y = (minimo + maximo) / 2 + .4 * (maximo - minimo) * math.sin(2 * math.pi * t / 4.8 + fase)
        barra.rotation_euler.y = math.radians(60) * math.sin(2 * math.pi * t / 2.4 + fase)
        assert minimo < barra.location.y < maximo
        barra.keyframe_insert('location', index=1, frame=fotograma)
        barra.keyframe_insert('rotation_euler', index=1, frame=fotograma)

camara = escena.camera
objetivo = bpy.data.objects['Objetivo_camara']
objetivo.location = (0, -.017, .52)
camara.animation_data_clear()
camara.data.lens = 42

def recorrido_camara(radio):
    camara.animation_data_clear()
    for fotograma in range(1, FOTOGRAMAS + 1):
        angulo = math.radians(-55 + 120 * (fotograma - 1) / (FOTOGRAMAS - 1))
        camara.location = (radio * math.cos(angulo), radio * math.sin(angulo), 1.8)
        camara.keyframe_insert('location', frame=fotograma)

radio = 2.9
for intento in range(8):
    recorrido_camara(radio)
    encuadre_valido = True
    for fotograma in (1, 72, 144, 216, 288):
        escena.frame_set(fotograma)
        grafo = bpy.context.evaluated_depsgraph_get()
        for objeto in mallas:
            evaluado = objeto.evaluated_get(grafo)
            for esquina in evaluado.bound_box:
                p = world_to_camera_view(escena, camara, evaluado.matrix_world @ Vector(esquina))
                if not (.07 < p.x < .93 and .07 < p.y < .93 and p.z > 0):
                    encuadre_valido = False
    if encuadre_valido:
        break
    radio *= 1.08
assert encuadre_valido, 'El futbolin queda cortado por la camara'

for fotograma in range(1, FOTOGRAMAS + 1):
    escena.frame_set(fotograma)
    grafo = bpy.context.evaluated_depsgraph_get()
    for barra in barras:
        y = barra.evaluated_get(grafo).matrix_world.translation.y
        assert barra['recorrido_min_m'] - 1e-6 <= y <= barra['recorrido_max_m'] + 1e-6
assert nombres_originales == {o.name for o in escena.objects}, 'Se han agregado piezas al modelo'

escena.render.image_settings.media_type = 'VIDEO'
escena.render.image_settings.file_format = 'FFMPEG'
escena.render.image_settings.color_mode = 'RGB'
escena.render.ffmpeg.format = 'MPEG4'
escena.render.ffmpeg.codec = 'H264'
escena.render.ffmpeg.constant_rate_factor = 'HIGH'
escena.render.ffmpeg.ffmpeg_preset = 'GOOD'
escena.render.filepath = str(SALIDA / 'futbolin_animado.mp4')
escena.frame_set(1)
bpy.ops.wm.save_as_mainfile(filepath=str(SALIDA / 'futbolin_animado.blend'))
print(f'VIDEO_READY: {FOTOGRAMAS / FPS:.0f}s, {FPS}fps, radio={radio:.2f}m, ocho barras animadas', flush=True)

escena.render.image_settings.media_type = 'IMAGE'
escena.render.image_settings.file_format = 'PNG'
escena.render.filepath = str(SALIDA / 'vista_previa.png')
bpy.ops.render.render(write_still=True)
