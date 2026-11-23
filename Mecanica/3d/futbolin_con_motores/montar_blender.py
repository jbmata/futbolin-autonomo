"""Montaje visual de cuatro actuadores sobre una copia del futbolin. Blender --background --python."""
import json
import math
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

SALIDA = Path(__file__).resolve().parent
bpy.ops.wm.open_mainfile(filepath=str(SALIDA / 'original/futbolin_madrid.blend'))
sc = bpy.context.scene
sc.frame_set(1)
originales = {o.name for o in sc.objects}
datos = json.loads((SALIDA / 'piezas.json').read_text(encoding='utf-8'))

def material(nombre, color, metal=0, rough=.4):
    m = bpy.data.materials.new('Montaje_' + nombre)
    m.diffuse_color = (*color, 1)
    m.use_nodes = True
    p = m.node_tree.nodes['Principled BSDF']
    p.inputs['Base Color'].default_value = (*color, 1)
    p.inputs['Metallic'].default_value = metal
    p.inputs['Roughness'].default_value = rough
    return m

mat = {n: material(n, c, m) for n,c,m in [
    ('metal',(.55,.59,.64),.8), ('aluminio',(.55,.60,.65),.8),
    ('laton',(.60,.38,.08),.7), ('motor',(.045,.05,.06),.4),
    ('campana',(.035,.45,.25),.05), ('carro',(.035,.18,.65),.1),
    ('soporte',(.85,.26,.045),.05), ('correa',(.025,.025,.03),0),
    ('rueda',(.055,.065,.075),0)]}
mallas = {}
for nombre,d in datos.items():
    me = bpy.data.meshes.new(nombre)
    me.from_pydata(d['vertices'], [], d['caras'])
    me.validate(); me.update()
    me.materials.append(mat[d['material']])
    if nombre.startswith('NEMA17'):
        me.materials.append(mat['metal'])
        for f in me.polygons:
            v = [me.vertices[i].co for i in f.vertices]
            frontal = sum(p.x for p in v)/len(v) < .150 if nombre.endswith('giro') else sum(p.y for p in v)/len(v) < .019
            f.material_index = int(frontal)
    mallas[nombre] = me

def vacio(nombre, col, posicion=(0,0,0), padre=None):
    ob = bpy.data.objects.new(nombre, None)
    col.objects.link(ob)
    ob.empty_display_size = .025
    ob.parent = padre
    ob.location = posicion
    return ob

def pieza(tipo, nombre, col, padre, posicion=(0,0,0)):
    ob = bpy.data.objects.new(nombre, mallas[tipo])
    col.objects.link(ob)
    ob.parent = padre
    ob.location = posicion
    ob['pieza'] = tipo
    return ob

def limites(ob):
    v = [ob.matrix_world @ Vector(c) for c in ob.bound_box]
    return [Vector([f(p[i] for p in v) for i in range(3)]) for f in (min,max)]

pared = limites(bpy.data.objects['Table_008'])[0].y
barras = sorted([o for o in sc.objects if o.name.startswith('Barra_') and o['equipo']=='A'], key=lambda o:o.name)
assert len(barras)==4
informes = []
conjuntos = []
for barra in barras:
    numero = barra.name.split('_')[1]
    col = bpy.data.collections.new('Actuador_' + numero)
    sc.collection.children.link(col)
    mango = next(o for o in barra.children if o.name.startswith('Handle'))
    inferior, superior = limites(mango)
    y_puno = superior.y
    assert y_puno < pared, 'El actuador debe quedar del lado del equipo A'
    assert abs(superior.y-inferior.y-.0895)<.0001
    giro = vacio('Giro_' + numero, col, (0,y_puno,0), barra)
    giro.rotation_euler.z = -math.pi/2
    for tipo in ('Collarin_superior','Collarin_inferior','Campana_superior','Campana_inferior','Salida_eje8','Acople_5_8_referencia'):
        pieza(tipo, numero+'_'+tipo, col, giro)

    movil = vacio('Carro_movil_' + numero, col, (barra.location.x,y_puno,barra.location.z))
    movil.rotation_euler.z = -math.pi/2
    d = movil.driver_add('location',1).driver
    v = d.variables.new(); v.name = 'desplazamiento'; v.type = 'TRANSFORMS'
    v.targets[0].id = barra; v.targets[0].transform_type = 'LOC_Y'; v.targets[0].transform_space = 'WORLD_SPACE'
    d.expression = f'{y_puno!r}+desplazamiento'
    pieza('Carro',numero+'_Carro',col,movil)
    pieza('NEMA17_giro',numero+'_NEMA17_giro',col,movil)
    for i,x in enumerate((.1345,.1745)):
        for j,y in enumerate((-.020,.020)):
            pieza('Rueda_V_referencia',f'{numero}_Rueda_{i}{j}',col,movil,(x,y,-.056))
            pieza('Eje_rueda_referencia',f'{numero}_Eje_rueda_{i}{j}',col,movil,(x,y,-.045))
            bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=.004, depth=.016)
            ob = bpy.context.object
            ob.name = f'{numero}_Separador_rueda_{i}{j}_referencia'
            for c in list(ob.users_collection):c.objects.unlink(ob)
            col.objects.link(ob)
            ob.parent = movil; ob.location = (x,y,-.043)
            ob.data.materials.append(mat['metal'])

    fijo = vacio('Guia_fija_' + numero,col,(barra.location.x,pared-.068,barra.location.z-.056))
    fijo.rotation_euler.z = -math.pi/2
    for tipo in ('Perfil2020','Soporte_pared','Tensor','NEMA17_lineal','Polea_motriz','Polea_loca','Separador','Correa'):
        pieza(tipo,numero+'_'+tipo,col,fijo)
    # El carro solo se traslada; la campana gira con la barra y el motor permanece fijo al carro.
    movil['pendiente'] = 'Ruedas, separadores y anclaje de correa provisionales; falta apoyo axial independiente.'
    lo,hi = barra['recorrido_min_m'],barra['recorrido_max_m']
    posiciones = [pared-y_puno+.1545-.068-y for y in (lo,hi)]
    assert min(posiciones)-.03275>.050 and max(posiciones)+.03275<.358, 'Carro fuera del tramo libre de guia'
    hueco_pared = pared-y_puno-hi-.018
    assert hueco_pared>.030, 'Collarin demasiado cerca de la pared'
    informes.append({'barra':barra.name,'guia_mm':400,'eje_barra_sobre_perfil_mm':56,'separadores_ruedas_mm':16,'carrera_mm':(hi-lo)*1000,
                     'centro_carro_desde_inicio_guia_mm':[x*1000 for x in sorted(posiciones)],
                     'holgura_minima_collarin_pared_mm':hueco_pared*1000})
    conjuntos.append((barra,giro,movil,fijo,y_puno))

# Comprueba la cinemática conservada en la copia, incluyendo los extremos reales del recorrido.
for barra,giro,movil,fijo,yp in conjuntos:
    accion = barra.animation_data.action
    barra.animation_data_clear()
    for y in (barra['recorrido_min_m'],0,barra['recorrido_max_m']):
        for angulo in (0,math.pi/2,math.pi):
            barra.location.y = y; barra.rotation_euler.y = angulo
            bpy.context.view_layer.update()
            assert abs(movil.matrix_world.translation.y-yp-y)<1e-6
            eje_movil = movil.matrix_world @ Vector((.1475,0,0))
            eje_giro = giro.matrix_world @ Vector((.1475,0,0))
            assert (eje_movil-eje_giro).length<1e-6, 'Motor y acople desalineados'
            assert abs(movil.matrix_world.to_euler().x)<1e-6, 'El carro gira con la barra'
    barra.animation_data_create(); barra.animation_data.action = accion
sc.frame_set(1)
bpy.context.view_layer.update()
assert originales.issubset({o.name for o in sc.objects})
assert len([o for o in sc.objects if o.get('pieza','').startswith('NEMA17')])==8
(SALIDA/'comprobaciones.json').write_text(json.dumps(informes,indent=2),encoding='utf-8')

cam = sc.camera
objetivo = bpy.data.objects['Objetivo_camara']
objetivo.location = (-.02,-.15,.55)
cam.location = (-2.15,-2.8,2.05); cam.data.lens = 46
sc.render.engine = 'BLENDER_EEVEE'
sc.eevee.taa_render_samples = 32
sc.render.resolution_x,sc.render.resolution_y = 1600,1000
sc.render.resolution_percentage = 100
sc.render.image_settings.media_type = 'IMAGE'
sc.render.image_settings.file_format = 'PNG'
sc.world.color = (.15,.15,.15)
sc['montaje'] = 'Cuatro barras del equipo A; ocho NEMA17. Soporte fijo rebajado para paso de brida. Referencia pendiente de validacion mecanica.'
sc['acople'] = 'Campana propia actual con salida8 y referencia5-8. No incluye el modelo Cults.'
sc['unidades'] = 'Metros, CAD convertido desde milimetros sin escalar las piezas.'

# Encuadra tambien el extremo de los perfiles, no solo el futbolin original.
for intento in range(8):
    bpy.context.view_layer.update()
    puntos = [world_to_camera_view(sc,cam,o.matrix_world@Vector(c)) for o in sc.objects if o.type=='MESH' and o.name!='Suelo' for c in o.bound_box]
    if all(.035<p.x<.965 and .035<p.y<.965 and p.z>0 for p in puntos):break
    cam.location = objetivo.location+(cam.location-objetivo.location)*1.08
else:raise AssertionError('Montaje fuera del encuadre')

# Se abre directamente por el lado de los motores, con materiales visibles.
for pantalla in bpy.data.screens:
    for area in pantalla.areas:
        if area.type=='VIEW_3D':
            espacio = area.spaces.active
            espacio.shading.type = 'MATERIAL'
            espacio.region_3d.view_rotation = cam.matrix_world.to_quaternion()
            espacio.region_3d.view_location = objetivo.location
            espacio.region_3d.view_distance = 2.8
            espacio.clip_end = 100
bpy.ops.object.select_all(action='DESELECT')
sc.render.filepath = str(SALIDA/'vista_general.png')
bpy.ops.wm.save_as_mainfile(filepath=str(SALIDA/'futbolin_con_motores.blend'))
bpy.ops.render.render(write_still=True)

# Detalle de una unidad: la primera barra, en el lado exterior del futbolin.
barra,giro,movil,fijo,yp = conjuntos[0]
objetivo.location = (barra.location.x,yp-.09,barra.location.z-.025)
cam.location = (barra.location.x-.42,yp-.60,barra.location.z+.34)
cam.data.lens = 52
sc.render.filepath = str(SALIDA/'detalle_actuador.png')
bpy.ops.render.render(write_still=True)
print('OK: 4 actuadores, 8 motores, escala real, copia intacta, ejes y recorrido comprobados.',flush=True)
