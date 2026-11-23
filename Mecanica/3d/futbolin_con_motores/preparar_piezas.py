"""Copia el futbolin original y triangula las piezas CAD existentes para Blender."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

from build123d import Axis, Cylinder, Plane, Polygon, Pos, import_step, revolve, export_step, export_stl

SALIDA = Path(__file__).resolve().parent
CAD = SALIDA.parent
sys.path.insert(0, str(CAD))
import acople_puno_nema17 as acople
from carrovsolt_nema17 import soporte_nema17
import montaje_soportes_lineales2020 as lineal

ORIGEN = CAD / 'futbolin_actualizado'
COPIA = SALIDA / 'original'
if not COPIA.exists():
    shutil.copytree(ORIGEN, COPIA)
for f in ORIGEN.rglob('*'):
    if f.is_file():
        assert hashlib.sha256(f.read_bytes()).digest() == hashlib.sha256((COPIA / f.relative_to(ORIGEN)).read_bytes()).digest()

lineal.LARGO = 400.
p = lineal.montaje()
lineal.comprobar(p)
# La brida choca con la horquilla original al retraerse. Rebaje exclusivo de esta copia.
ALTURA_BARRA = 56.
paso = Pos(47,0,ALTURA_BARRA)*Cylinder(38,146).rotate(Axis.Y,90)
p[1] -= paso
assert p[1].is_valid and len(p[1].solids()) == 1
export_step(p[1], str(SALIDA/'soporte_pared_rebajado.step'))
export_stl(Pos(0,0,68)*p[1].rotate(Axis.Y,-90), str(SALIDA/'soporte_pared_rebajado.stl'), tolerance=.02)
campanas = acople.piezas()
acople.comprobar(campanas)

materiales = ['metal', 'metal', 'campana', 'campana', 'metal']
nombres = ['Collarin_superior', 'Collarin_inferior', 'Campana_superior', 'Campana_inferior', 'Salida_eje8']
piezas = [(n, s, m) for n, s, m in zip(nombres, campanas, materiales)]
piezas.append(('Acople_5_8_referencia', acople.referencias()[2], 'laton'))
motor = import_step(str(CAD / 'nema17_40mm_descargado.step')).rotate(Axis.Y, -90)
piezas.append(('NEMA17_giro', Pos(147.5, 0, 0) * motor, 'motor'))
piezas.append(('Carro', Pos(154.5, 0, -35) * soporte_nema17(), 'carro'))
nombres = ['Perfil2020', 'Soporte_pared', 'Tensor', 'NEMA17_lineal', 'Polea_motriz', 'Polea_loca', 'Separador', 'Correa']
materiales = ['aluminio', 'soporte', 'soporte', 'motor', 'laton', 'laton', 'metal', 'correa']
for n, s, m in zip(nombres, p, materiales):
    if n == 'NEMA17_lineal':
        s = Pos(lineal.X_MOTOR, lineal.Y_MOTOR, 0) * motor.rotate(Axis.Z, 90)
    piezas.append((n, s, m))

# shortcut: rueda V de referencia; sustituir por el modelo comercial y sus separadores al elegirlos.
seccion = Plane.XZ * Polygon((2.6,-5),(9.6,-5),(11.6,0),(9.6,5),(2.6,5), align=None)
rueda = revolve(seccion, axis=Axis.Z)
piezas.append(('Rueda_V_referencia', rueda, 'rueda'))
piezas.append(('Eje_rueda_referencia', Cylinder(2.5,32), 'metal'))
q = p[0] & (Pos(200,20,0)*rueda)
assert not q or q.volume<1e-4, 'Rueda de referencia atraviesa el perfil'

# Comprueba el rebaje en todo el tramo usado y las envolventes de motor/carro.
moviles = campanas + [Pos(147.5,0,0)*motor, Pos(154.5,0,-35)*soporte_nema17()]
fijas = p[:3] + [Pos(lineal.X_MOTOR,lineal.Y_MOTOR,0)*motor.rotate(Axis.Z,90)]
for x in (-16.1,12.82,41.74,70.66,99.583):
    for movil in moviles:
        for fija in fijas:
            q = fija & (Pos(x,0,ALTURA_BARRA)*movil)
            assert not q or q.volume<1e-4, ('Interferencia en el montaje',x,q.volume)
print('OK: soporte rebajado, motor/campana/carro libres en cinco posiciones del recorrido.',flush=True)

datos = {}
for nombre, s, material in piezas:
    assert s.is_valid and s.volume > 0, nombre
    vertices, caras = s.tessellate(.08, .15)
    datos[nombre] = {'vertices': [[v.X/1000, v.Y/1000, v.Z/1000] for v in vertices],
                     'caras': caras, 'material': material}
    print(nombre, len(caras), flush=True)
(SALIDA / 'piezas.json').write_text(json.dumps(datos, separators=(',', ':')), encoding='utf-8')
print('OK: copia verificada y piezas preparadas.', flush=True)
