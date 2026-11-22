"""Envolvente de comprobación del motor NEMA 17 17HS4401 (no es una pieza a fabricar). Unidades: mm.

Cara frontal en X = X_CARA_MOTOR; el cuerpo va hacia +X y el eje hacia -X; eje del motor a ALTURA_EJE.
Las medidas salen de fichas de distribuidores (no del plano del fabricante): confirmar con el motor real.

    python motor_17hs4401.py --ver    # solo para verlo en OCP CAD Viewer
"""
from build123d import Axis, Box, Cylinder, Pos

import _comun

X_CARA_MOTOR = -7.00
ALTURA_EJE = 35.00              # provisional, desde la cara inferior de la placa
PASO_MOTOR = 31.00              # patrón frontal 31 x 31, cuatro roscas M3
DIAMETRO_RESALTE = 22.00
ANCHO_MOTOR = 42.30             # envolvente máxima del fabricante
LARGO_MOTOR = 40.00             # cuerpo de 40 mm
LARGO_EJE = 25.00               # eje de 5 mm, 25 ± 1 mm libres
DIAMETRO_EJE = 5.00


def motor_referencia():
    """Cuerpo, resalte de 2 mm y eje, sin conector ni detalles de fabricación."""
    cuerpo = Pos(X_CARA_MOTOR + LARGO_MOTOR / 2, 0, ALTURA_EJE) * \
        Box(LARGO_MOTOR, ANCHO_MOTOR, ANCHO_MOTOR)
    resalte = Pos(X_CARA_MOTOR - 1, 0, ALTURA_EJE) * Cylinder(DIAMETRO_RESALTE / 2, 2).rotate(Axis.Y, 90)
    eje = Pos(X_CARA_MOTOR - LARGO_EJE / 2, 0, ALTURA_EJE) * Cylinder(DIAMETRO_EJE / 2, LARGO_EJE) \
        .rotate(Axis.Y, 90)
    return cuerpo + resalte + eje


if __name__ == "__main__":
    args = _comun.argumentos(__doc__.splitlines()[0])
    if args.ver:
        _comun.mostrar(motor_referencia(), nombres=["motor"], colores=["silver"])
    if not args.ver:
        print(f"motor 17HS4401 (referencia): {motor_referencia().volume:.0f} mm3; usa --ver para verlo")
