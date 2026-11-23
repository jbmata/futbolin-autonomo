"""Soporte lineal del tensor para V-Slot 2020. Unidades: mm.

Dos mejillas paralelas unidas por la punta, con ranuras M5 alineadas.
Se conserva el plano XY de la primera mejilla; perfil hacia X negativo.
    python soporte_lineal_tensor2020.py [--ver]
"""
from pathlib import Path
from build123d import Axis, Box, Circle, Pos, RectangleRounded, SlotCenterToCenter, extrude, export_stl

import _comun
from soporte_lineal_motor2020 import ALTURA_EJE_MOTOR, DIAMETRO_M5, ESPESOR

ANCLAJES = (-32.0, -12.0)
X_POLEA = 22.0
RECORRIDO = 10.0
SEMILUZ = 10.2                 # 20.4 mm entre caras, para abrazar el perfil 2020.
PUNTA_INTERIOR = 42.0
PUNTA_EXTERIOR = PUNTA_INTERIOR + ESPESOR


def soporte_lineal_tensor():
    seccion = Pos(3, 0) * RectangleRounded(90, 28, 4)
    for x in ANCLAJES:
        seccion -= Pos(x, 0) * Circle(DIAMETRO_M5 / 2)
    seccion -= Pos(X_POLEA, ALTURA_EJE_MOTOR) * SlotCenterToCenter(RECORRIDO, DIAMETRO_M5)
    mejilla = extrude(seccion, amount=ESPESOR)
    pieza = mejilla + Pos(0,0,-2*SEMILUZ-ESPESOR)*mejilla
    pieza += Pos((PUNTA_INTERIOR+PUNTA_EXTERIOR)/2,0,-SEMILUZ)*Box(
        ESPESOR,28,2*(SEMILUZ+ESPESOR))
    assert pieza.is_valid and len(pieza.solids()) == 1
    assert abs(pieza.bounding_box().size.Z - 2*(SEMILUZ+ESPESOR)) < 1e-6
    assert PUNTA_INTERIOR-(X_POLEA+RECORRIDO/2+9.4)>5, 'La correa toca la union de punta'
    return pieza


if __name__ == "__main__":
    args = _comun.argumentos(__doc__.splitlines()[0])
    pieza = soporte_lineal_tensor()
    _comun.exportar(pieza, "soporte_lineal_tensor2020", args.out, stl=False)
    # Punta contra la cama: las dos mejillas suben sin un techo que necesite puentes.
    imprimible = Pos(0,0,PUNTA_EXTERIOR)*pieza.rotate(Axis.Y,90)
    assert abs(imprimible.bounding_box().min.Z)<1e-6
    export_stl(imprimible,str(Path(args.out)/'soporte_lineal_tensor2020.stl'),tolerance=.01,angular_tolerance=.1)
    if args.ver:
        _comun.mostrar(pieza, nombres=["soporte lineal tensor 2020"], colores=["royalblue"])
