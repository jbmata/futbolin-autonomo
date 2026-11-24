"""Placa del carro V-Slot (plano aportado por el usuario). Unidades: mm.

Origen XY en el centro; cara inferior en Z=0; todos los huecos son pasantes.

    python carrovsolt.py          # escribe carrovsolt.step y carrovsolt.stl junto al script
    python carrovsolt.py --ver    # además la muestra en OCP CAD Viewer
"""
from build123d import Circle, Pos, RectangleRounded, SlotCenterToCenter, extrude

import _comun

LADO = 65.50
ESPESOR = 6.00                  # confirmado por el usuario; el plano no lo indica
RADIO_ESQUINA = 3.00            # estimado del dibujo, sin cota de radio
DIAMETRO_TALADRO = 5.10
DIAMETRO_GRANDE = 7.20
PASO = 10.00
ANCHO_RANURA_CENTRAL = 5.00
LARGO_RANURA_CENTRAL = 25.64    # longitud total, con los extremos semicirculares
ANCHO_RANURA_LATERAL = 3.00
LARGO_RANURA_LATERAL = 9.50
BORDE_CENTRO_RANURA_LATERAL = 5.00   # centro a borde exterior: estimado, no acotado

# Retícula centrada de 10 mm: doce taladros de 5,10 (coordenadas en múltiplos de PASO).
TALADROS = [(-2, 2), (-1, 2), (0, 2), (1, 2), (2, 2), (0, 1), (-1, 0), (0, 0), (1, 0),
            (0, -1), (-1, -2), (1, -2)]


def comprobar():
    assert ESPESOR > 0 and 0 < 2 * RADIO_ESQUINA < LADO, "Radio de esquina incompatible con el lado."
    assert LARGO_RANURA_CENTRAL >= ANCHO_RANURA_CENTRAL
    assert LARGO_RANURA_LATERAL >= ANCHO_RANURA_LATERAL


def placa():
    """Placa de 65,5 x 65,5 x 6 con taladros y ranuras pasantes."""
    comprobar()
    s = RectangleRounded(LADO, LADO, RADIO_ESQUINA)
    for i, j in TALADROS:
        s -= Pos(i * PASO, j * PASO) * Circle(DIAMETRO_TALADRO / 2)
    for x in (-2 * PASO, 0, 2 * PASO):                       # tres de 7,20 en la fila inferior
        s -= Pos(x, -2 * PASO) * Circle(DIAMETRO_GRANDE / 2)
    for x in (-2 * PASO, 2 * PASO):                          # dos ranuras largas de 5,00 x 25,64
        s -= Pos(x, 0) * SlotCenterToCenter(LARGO_RANURA_CENTRAL - ANCHO_RANURA_CENTRAL,
                                            ANCHO_RANURA_CENTRAL, rotation=90)
    for x in (-(LADO / 2 - BORDE_CENTRO_RANURA_LATERAL), LADO / 2 - BORDE_CENTRO_RANURA_LATERAL):
        for y in (-2 * PASO, 0, 2 * PASO):                   # seis laterales de 3,00 x 9,50
            s -= Pos(x, y) * SlotCenterToCenter(LARGO_RANURA_LATERAL - ANCHO_RANURA_LATERAL,
                                                ANCHO_RANURA_LATERAL, rotation=90)
    return extrude(s, amount=ESPESOR)


if __name__ == "__main__":
    args = _comun.argumentos(__doc__.splitlines()[0])
    pieza = placa()
    _comun.exportar(pieza, "carrovsolt", args.out)
    if args.ver:
        _comun.mostrar(pieza, nombres=["placa"], colores=["orange"])
