"""Triangula las piezas CAD del robot para Blender: una guia V-Slot por barra, segun Medidas/mesa.json.

Cada barra tiene su carrera (portero, defensa, medios y delanteros no recorren lo mismo), asi que la guia
que lleva su carro tambien: se calcula su longitud minima y se redondea a multiplos de 10 mm.
Escribe piezas.json (mallas), guias.json (longitud de guia por rol) y soporte_pared_rebajado.step/.stl.
El siguiente paso es montar_blender.py (Blender), que parte de la mesa de ../futbolin_actualizado.

Uso (Python 3.11 con build123d):  python preparar_piezas.py
"""
import json
import math
import sys
from pathlib import Path

from build123d import Axis, Cylinder, Plane, Polygon, Pos, import_step, revolve, export_step, export_stl

SALIDA = Path(__file__).resolve().parent
CAD = SALIDA.parent                                  # Mecanica/3d
RAIZ = CAD.parent.parent
for carpeta in (CAD / 'piezas', CAD / 'montajes', RAIZ / 'Modelo'):
    sys.path.insert(0, str(carpeta))
import acople_puno_nema17 as acople
from carrovsolt_nema17 import soporte_nema17
import montaje_soportes_lineales2020 as lineal
import soporte_lineal_tensor2020 as tensor
import world

# --- longitud de la guia de cada barra ---------------------------------------------------------
# Posiciones sobre la guia, en mm desde su inicio (el inicio esta 68 mm fuera de la cara exterior de la
# pared). El carro va en el puño de la barra, 154,5 mm por delante del eje del puño; el puño queda a
# d mm de la pared: d_s0 con la barra en el tope exterior (s = 0) y d_s1 = d_s0 - carrera en el interior.
# Con los topes reales d_s1 sale de 11 mm (de -7 en el portero): el extremo del mango llega casi a la pared y
# la campana no cabe. El recorrido util del robot acaba donde queda HOLGURA_PARED; lo demas es mecanico.
PARED_FUERA_PLANCHA = 60.0      # mm, cara exterior de la pared a la banda de la plancha (3D de la mesa)
LARGO_MANGO = 89.5              # mm; el plano mide desde la punta del mango y el actuador va en su extremo interior
HOLGURA_PARED = 50.0            # mm minimos entre el extremo interior del mango y la pared: por debajo el collarin
                                # inferior choca con el soporte de pared (comprobado: 48 mm, 632 mm3; 49, 314; 50, libre)
INICIO_GUIA_A_PARED = 68.0      # mm
CARRO_A_PUNO = 154.5            # mm, centro del carro respecto al eje del puño
SOBRE_PUNO = CARRO_A_PUNO - INICIO_GUIA_A_PARED       # 86,5: posicion del centro del carro = d + 86,5
LADO_CARRO = 65.5               # mm (carrovsolt.LADO)
INICIO_LIBRE = 50.0             # mm de guia donde el carro no puede llegar (soporte de pared y motor lineal)
FIN_LIBRE = tensor.PUNTA_INTERIOR                     # 42 mm de guia ocupados por el tensor

raw = json.loads((RAIZ / 'Medidas' / 'mesa.json').read_text(encoding='utf-8'))
medidas = world.load_measurements()


def d_exterior(rol):
    """Distancia del puño a la pared con la barra en el tope exterior (s = 0), en mm."""
    b = raw['barras'][rol]
    primer_agujero = min(b['agujeros_desde_punta_mango']['valor'][k] for k in b['jugadores_en_agujeros']['valor'])
    punta_fuera_de_banda = primer_agujero - medidas['roles'][rol]['home'][0]
    return punta_fuera_de_banda - PARED_FUERA_PLANCHA - LARGO_MANGO


guias, rangos, util = {}, {}, {}
for rol in ('GK', 'DEF', 'MID', 'ATK'):
    carrera = medidas['roles'][rol]['travel']
    d0 = d_exterior(rol)
    d1 = d0 - carrera
    d1_util = max(d1, HOLGURA_PARED, INICIO_LIBRE + LADO_CARRO / 2 - SOBRE_PUNO)
    necesario = d0 + SOBRE_PUNO + LADO_CARRO / 2 + FIN_LIBRE
    guias[rol] = int(math.ceil(necesario / 10.0) * 10)
    util[rol] = d0 - d1_util
    rangos[rol] = (d1_util - 68.0, d0 - 68.0)        # x del origen del puño en el montaje lineal (mm)
    print(f'{rol}: carrera {carrera:.1f} mm, puño a {d1:.1f}..{d0:.1f} mm de la pared (util {util[rol]:.1f} mm), '
          f'guia {guias[rol]} mm', flush=True)
(SALIDA / 'guias.json').write_text(json.dumps(
    {'guia_mm': guias, 'carrera_mm': {r: medidas['roles'][r]['travel'] for r in guias},
     'carrera_util_mm': util, 'pared_fuera_plancha_mm': PARED_FUERA_PLANCHA, 'largo_mango_mm': LARGO_MANGO,
     'holgura_pared_mm': HOLGURA_PARED}, indent=2), encoding='utf-8')

# --- piezas del carro y la campana (iguales en todas las barras) -------------------------------
campanas = acople.piezas()
acople.comprobar(campanas)
motor = import_step(str(CAD / 'referencias' / 'nema17_40mm_descargado.step')).rotate(Axis.Y, -90)
carro = Pos(154.5, 0, -35) * soporte_nema17()
motor_giro = Pos(147.5, 0, 0) * motor
materiales = ['metal', 'metal', 'campana', 'campana', 'metal']
nombres = ['Collarin_superior', 'Collarin_inferior', 'Campana_superior', 'Campana_inferior', 'Salida_eje8']
piezas = [(n, s, m) for n, s, m in zip(nombres, campanas, materiales)]
piezas.append(('Acople_5_8_referencia', acople.referencias()[2], 'laton'))
piezas.append(('NEMA17_giro', motor_giro, 'motor'))
piezas.append(('Carro', carro, 'carro'))

# shortcut: rueda V de referencia; sustituir por el modelo comercial y sus separadores al elegirlos.
seccion = Plane.XZ * Polygon((2.6, -5), (9.6, -5), (11.6, 0), (9.6, 5), (2.6, 5), align=None)
rueda = revolve(seccion, axis=Axis.Z)
piezas.append(('Rueda_V_referencia', rueda, 'rueda'))
piezas.append(('Eje_rueda_referencia', Cylinder(2.5, 32), 'metal'))

# --- guia, soporte, tensor y correa de cada barra ----------------------------------------------
ALTURA_BARRA = 56.
moviles = campanas + [motor_giro, carro]
POR_ROL = ['Perfil2020', 'Tensor', 'Polea_loca', 'Correa']            # dependen de la longitud de la guia
COMUNES = ['Soporte_pared', 'NEMA17_lineal', 'Polea_motriz', 'Separador']
colores = dict(Perfil2020='aluminio', Soporte_pared='soporte', Tensor='soporte', NEMA17_lineal='motor',
               Polea_motriz='laton', Polea_loca='laton', Separador='metal', Correa='correa')
orden = ['Perfil2020', 'Soporte_pared', 'Tensor', 'NEMA17_lineal', 'Polea_motriz', 'Polea_loca', 'Separador', 'Correa']
soporte_escrito = False
for rol, largo in guias.items():
    lineal.LARGO = float(largo)
    p = lineal.montaje()
    lineal.comprobar(p)
    # La brida choca con la horquilla original al retraerse. Rebaje exclusivo de este montaje.
    paso = Pos(47, 0, ALTURA_BARRA) * Cylinder(38, 146).rotate(Axis.Y, 90)
    p[1] -= paso
    assert p[1].is_valid and len(p[1].solids()) == 1
    if not soporte_escrito:
        export_step(p[1], str(SALIDA / 'soporte_pared_rebajado.step'))
        export_stl(Pos(0, 0, 68) * p[1].rotate(Axis.Y, -90), str(SALIDA / 'soporte_pared_rebajado.stl'), tolerance=.02)
        soporte_escrito = True
    q = p[0] & (Pos(200, 20, 0) * rueda)
    assert not q or q.volume < 1e-4, 'Rueda de referencia atraviesa el perfil'
    motor_lineal = Pos(lineal.X_MOTOR, lineal.Y_MOTOR, 0) * motor.rotate(Axis.Z, 90)
    # Interferencias con el carro en cinco posiciones repartidas por todo el recorrido de esta barra
    fijas = p[:3] + [motor_lineal]
    x0, x1 = rangos[rol]
    for k in range(5):
        x = x0 + (x1 - x0) * k / 4
        for movil in moviles:
            for fija in fijas:
                q = fija & (Pos(x, 0, ALTURA_BARRA) * movil)
                assert not q or q.volume < 1e-4, ('Interferencia en el montaje', rol, x, q.volume)
    print(f'OK {rol}: guia de {largo} mm, motor/campana/carro libres en cinco posiciones del recorrido.', flush=True)
    por_nombre = dict(zip(orden, p))
    por_nombre['NEMA17_lineal'] = motor_lineal
    for n in orden:
        if n in POR_ROL:
            piezas.append((f'{n}_{largo}', por_nombre[n], colores[n]))
        elif n in COMUNES and not any(x[0] == n for x in piezas):
            piezas.append((n, por_nombre[n], colores[n]))

datos = {}
for nombre, s, material in piezas:
    assert s.is_valid and s.volume > 0, nombre
    vertices, caras = s.tessellate(.08, .15)
    datos[nombre] = {'vertices': [[v.X / 1000, v.Y / 1000, v.Z / 1000] for v in vertices],
                     'caras': caras, 'material': material}
    print(nombre, len(caras), flush=True)
(SALIDA / 'piezas.json').write_text(json.dumps(datos, separators=(',', ':')), encoding='utf-8')
print('OK: piezas preparadas.', flush=True)
