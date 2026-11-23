"""Soporte lineal del tensor para V-Slot 2020. Unidades: mm.

Pieza plana en XY, Z=0..6; perfil hacia X negativo. Tensado manual por ranura, eje a la cota del perfil.
    python soporte_lineal_tensor2020.py [--ver]
"""
from build123d import Circle, Pos, RectangleRounded, SlotCenterToCenter, extrude

import _comun
from soporte_lineal_motor2020 import ALTURA_EJE_MOTOR, DIAMETRO_M5, ESPESOR

ANCLAJES = (-32.0, -12.0)
X_POLEA = 22.0
RECORRIDO = 10.0


def soporte_lineal_tensor():
    seccion = Pos(1, 0) * RectangleRounded(86, 28, 4)
    for x in ANCLAJES:
        seccion -= Pos(x, 0) * Circle(DIAMETRO_M5 / 2)
    seccion -= Pos(X_POLEA, ALTURA_EJE_MOTOR) * SlotCenterToCenter(RECORRIDO, DIAMETRO_M5)
    pieza = extrude(seccion, amount=ESPESOR)
    assert pieza.is_valid and len(pieza.solids()) == 1
    assert abs(pieza.bounding_box().size.Z - ESPESOR) < 1e-6
    assert 44 - (X_POLEA + RECORRIDO / 2 + DIAMETRO_M5 / 2) >= 10
    return pieza


if __name__ == "__main__":
    args = _comun.argumentos(__doc__.splitlines()[0])
    pieza = soporte_lineal_tensor()
    _comun.exportar(pieza, "soporte_lineal_tensor2020", args.out)
    if args.ver:
        _comun.mostrar(pieza, nombres=["soporte lineal tensor 2020"], colores=["royalblue"])
