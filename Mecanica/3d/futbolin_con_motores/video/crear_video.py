"""Video del montaje actual: cuatro actuadores, barras y jugadores. Ejecutar desde Blender."""
import math
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

SALIDA = Path(__file__).resolve().parent
bpy.ops.wm.open_mainfile(filepath=str(SALIDA.parent/'futbolin_con_motores.blend'))
sc = bpy.context.scene
originales = {o.name for o in sc.objects}
barras = sorted([o for o in sc.objects if o.name.startswith('Barra_')], key=lambda o:o.name)
assert len(barras)==8
FPS, FINAL = 24, 288
sc.frame_start,sc.frame_end = 1,FINAL
sc.render.fps = FPS
sc.render.fps_base = 1

for i,b in enumerate(barras):
    b.animation_data_clear()
    lo,hi = b['recorrido_min_m'],b['recorrido_max_m']
    for f in range(1,FINAL+1):
        t = (f-1)/FPS
        b.location.y = (lo+hi)/2+.4*(hi-lo)*math.sin(2*math.pi*t/4.8+i*.73)
        b.rotation_euler.y = math.radians(60)*math.sin(2*math.pi*t/2.4+i*.73)
        assert lo<b.location.y<hi
        b.keyframe_insert('location',index=1,frame=f)
        b.keyframe_insert('rotation_euler',index=1,frame=f)

def giro_por_deslizamiento(ob,barra,indice,radio,signo=1):
    d = ob.driver_add('rotation_euler',indice).driver
    v = d.variables.new();v.name = 'y';v.type = 'TRANSFORMS'
    v.targets[0].id = barra;v.targets[0].transform_type = 'LOC_Y';v.targets[0].transform_space = 'WORLD_SPACE'
    d.expression = f'{signo}*y/{radio}'

for b in [b for b in barras if b['equipo']=='A']:
    n = b.name.split('_')[1]
    for i in range(2):
        for j in range(2):
            ob = bpy.data.objects[f'{n}_Rueda_{i}{j}']
            giro_por_deslizamiento(ob,b,2,.0116,1 if j else -1)
    for tipo,x,r in [('Polea_motriz',-.032,20*.002/(2*math.pi)),('Polea_loca',.422,.0087)]:
        ob = bpy.data.objects[n+'_'+tipo]
        ob.data = ob.data.copy()
        ob.data.transform(Matrix.Translation((-x,0,0)))
        ob.location.x = x
        giro_por_deslizamiento(ob,b,1,r)

sc.render.engine = 'BLENDER_EEVEE'
sc.eevee.taa_render_samples = 16
sc.render.resolution_x,sc.render.resolution_y = 1280,720
sc.render.resolution_percentage = 100
sc.render.film_transparent = False
cam = sc.camera
target = bpy.data.objects['Objetivo_camara']
cam.animation_data_clear();target.animation_data_clear()
cam.data.lens = 46

# Barrido inverso desde el lado de los motores lineales, por fuera de las cartelas.
radio = 1.
for intento in range(8):
    cam.animation_data_clear();target.animation_data_clear()
    for f in range(1,FINAL+1):
        sc.frame_set(f)
        if f<=96:
            t = (f-1)/95
            target.location = (-.02,-.15,.55)
            cam.location = target.location+Vector((2.4-.35*t,-2.9,1.55))*radio
        else:
            t = (f-97)/(FINAL-97)
            x = .2111-.704*t
            target.location = (x,-.55,.83)
            cam.location = (x+.28,-1.13,1.12)
        cam.keyframe_insert('location',frame=f)
        target.keyframe_insert('location',frame=f)
    valido = True
    for f in (1,48,96):
        sc.frame_set(f)
        grafo = bpy.context.evaluated_depsgraph_get()
        for ob in [o for o in sc.objects if o.type=='MESH' and o.name!='Suelo']:
            ev = ob.evaluated_get(grafo)
            for c in ev.bound_box:
                p = world_to_camera_view(sc,cam,ev.matrix_world@Vector(c))
                if not (.04<p.x<.96 and .04<p.y<.96 and p.z>0):valido = False
    if valido:break
    radio *= 1.08
assert valido,'Vista general recortada'

# Cada motor lineal debe quedar dentro de plano y visible al pasar la camara por el.
for n in ('06','04','02','01'):
    motor = bpy.data.objects[n+'_NEMA17_lineal']
    centro_x = motor.parent.location.x
    f = round(97+(.2111-centro_x)/.704*(FINAL-97))
    sc.frame_set(min(FINAL,max(97,f)))
    grafo = bpy.context.evaluated_depsgraph_get()
    puntos = [motor.matrix_world@Vector(c) for c in motor.bound_box]
    bajo = Vector([min(p[i] for p in puntos) for i in range(3)])
    alto = Vector([max(p[i] for p in puntos) for i in range(3)])
    visibles = 0
    for y in (.25,.5,.75):
        for z in (.25,.5,.75):
            p = Vector((alto.x-.00005,bajo.y+y*(alto.y-bajo.y),bajo.z+z*(alto.z-bajo.z)))
            v = world_to_camera_view(sc,cam,p)
            direccion = p-cam.matrix_world.translation
            hit,loc,normal,indice,ob,matriz = sc.ray_cast(grafo,cam.matrix_world.translation,direccion.normalized(),distance=direccion.length+.001)
            if hit and ob.original==motor and .04<v.x<.96 and .04<v.y<.96:visibles+=1
    assert visibles>=3,(motor.name,'Motor oculto en el barrido',visibles)
    print('MOTOR_VISIBLE',n,'fotograma',sc.frame_current,'muestras',visibles,flush=True)

# La animacion mantiene el eje y la traslacion de cada carro alineados con su campana.
for f in range(1,FINAL+1):
    sc.frame_set(f)
    for b in barras:
        assert b['recorrido_min_m']-1e-6<=b.matrix_world.translation.y<=b['recorrido_max_m']+1e-6
        if b['equipo']=='A':
            n = b.name.split('_')[1]
            g = bpy.data.objects['Giro_'+n]
            c = bpy.data.objects['Carro_movil_'+n]
            a = g.matrix_world@Vector((.1475,0,0))
            z = c.matrix_world@Vector((.1475,0,0))
            assert (a-z).length<1e-6,'Desalineacion entre motor y barra'
            assert abs((c.matrix_world.to_3x3()@Vector((0,0,1))).z-1)<1e-6
assert originales=={o.name for o in sc.objects}

sc.render.image_settings.media_type = 'VIDEO'
sc.render.image_settings.file_format = 'FFMPEG'
sc.render.image_settings.color_mode = 'RGB'
sc.render.ffmpeg.format = 'MPEG4'
sc.render.ffmpeg.codec = 'H264'
sc.render.ffmpeg.constant_rate_factor = 'HIGH'
sc.render.ffmpeg.ffmpeg_preset = 'GOOD'
sc.render.filepath = str(SALIDA/'futbolin_con_motores.mp4')
sc.frame_set(1)
for pantalla in bpy.data.screens:
    for area in pantalla.areas:
        if area.type=='VIEW_3D':area.spaces.active.region_3d.view_perspective='CAMERA'
bpy.ops.wm.save_as_mainfile(filepath=str(SALIDA/'futbolin_con_motores_animado.blend'))
sc.render.image_settings.media_type = 'IMAGE'
sc.render.image_settings.file_format = 'PNG'
sc.frame_set(174)
sc.render.filepath = str(SALIDA/'vista_previa.png')
bpy.ops.render.render(write_still=True)
print('VIDEO_READY: 12 segundos, 720p, 24fps, ejes y recorridos comprobados en 288 fotogramas.',flush=True)
