"""Carro con soporte para el motor de giro NEMA 17, eje paralelo al V-Slot. Unidades: mm.

El V-Slot y el desplazamiento siguen X; la placa base está en XY (se reutiliza `carrovsolt.placa()`).
La cara del motor apoya en X = -7 y los tornillos entran desde X negativo.

    python carrovsolt_nema17.py          # escribe carrovsolt_nema17.step y .stl junto al script
    python carrovsolt_nema17.py --ver    # además lo muestra en OCP CAD Viewer
"""
from build123d import Axis, Box, Cylinder, Plane, Polygon, Pos, extrude, fillet

import _comun
from carrovsolt import ESPESOR, RADIO_ESQUINA, placa
from motor_17hs4401 import (ALTURA_EJE, ANCHO_MOTOR, DIAMETRO_RESALTE, PASO_MOTOR, X_CARA_MOTOR)

ESPESOR_SOPORTE = 6.00
ANCHO_SOPORTE = 54.00
DIAMETRO_TORNILLO = 3.40
HOLGURA_RESALTE = 0.40
ESPESOR_NERVIO = 4.00
LARGO_NERVIO = 17.00
ALTO_NERVIO = 29.00
ALTO_SOPORTE = ALTURA_EJE + 26.00
SOLAPE = 0.20


def comprobar():
    assert DIAMETRO_TORNILLO >= 3 and ESPESOR_SOPORTE > 0, "Comprobar pared y paso para M3."
    assert HOLGURA_RESALTE >= 0
    assert ALTURA_EJE - ANCHO_MOTOR / 2 > ESPESOR + 6.5, \
        "El motor invade el espacio de tuercas/arandelas de las ruedas."
    assert ANCHO_SOPORTE / 2 - ESPESOR_NERVIO > ANCHO_MOTOR / 2 + 1, \
        "Los nervios invaden la envolvente lateral del motor."
    assert 0 < ALTO_NERVIO < ALTURA_EJE and ESPESOR_NERVIO > 0 and LARGO_NERVIO > 0
    assert X_CARA_MOTOR - ESPESOR_SOPORTE > -20 + 6.5 and X_CARA_MOTOR + LARGO_NERVIO < 20 - 6.5, \
        "El soporte invade el acceso a los anclajes de las ruedas."


def _pared_motor():
    pared = Pos(X_CARA_MOTOR - ESPESOR_SOPORTE / 2, 0, ALTO_SOPORTE / 2) * \
        Box(ESPESOR_SOPORTE, ANCHO_SOPORTE, ALTO_SOPORTE)
    return fillet(pared.edges().filter_by(Axis.X), RADIO_ESQUINA)


def _nervios():
    # Quedan por fuera del cuerpo del NEMA y entre los anclajes de las ruedas.
    tri = Polygon((X_CARA_MOTOR - SOLAPE, ESPESOR - SOLAPE),
                  (X_CARA_MOTOR + LARGO_NERVIO, ESPESOR - SOLAPE),
                  (X_CARA_MOTOR - SOLAPE, ESPESOR + ALTO_NERVIO), align=None)
    nervio = extrude(Plane.XZ * tri, amount=ESPESOR_NERVIO)          # hacia -Y
    return [Pos(0, y, 0) * nervio for y in (-ANCHO_SOPORTE / 2 + ESPESOR_NERVIO, ANCHO_SOPORTE / 2)]


def _taladros_motor():
    x0 = X_CARA_MOTOR - ESPESOR_SOPORTE - SOLAPE
    largo = ESPESOR_SOPORTE + 2 * SOLAPE
    xc = x0 + largo / 2

    def agujero(d, y, z):
        return Pos(xc, y, z) * Cylinder(d / 2, largo).rotate(Axis.Y, 90)

    piezas = [agujero(DIAMETRO_RESALTE + HOLGURA_RESALTE, 0, ALTURA_EJE)]
    for y in (-PASO_MOTOR / 2, PASO_MOTOR / 2):
        for z in (ALTURA_EJE - PASO_MOTOR / 2, ALTURA_EJE + PASO_MOTOR / 2):
            piezas.append(agujero(DIAMETRO_TORNILLO, y, z))
    return piezas


def soporte_nema17():
    """Placa + pared del motor + dos nervios, con el taladro del resalte y los cuatro M3."""
    comprobar()
    pieza = placa() + _pared_motor()
    for n in _nervios():
        pieza += n
    for t in _taladros_motor():
        pieza -= t
    return pieza


if __name__ == "__main__":
    args = _comun.argumentos(__doc__.splitlines()[0])
    pieza = soporte_nema17()
    _comun.exportar(pieza, "carrovsolt_nema17", args.out)
    if args.ver:
        _comun.mostrar(pieza, nombres=["soporte"], colores=["orange"])
