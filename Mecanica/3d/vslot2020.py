"""Perfil de aluminio V-Slot 2020, en build123d. Unidades: mm.

Reconstruido a partir de la STEP del fabricante (`3663_2025012894452_2020 V-slot 100mm.STEP`):
las medidas salen de su sección. Perfil centrado en el origen (XY) y extruido a lo largo de Z.
Las cuatro caras son iguales (giradas 90°); cada una tiene una ranura en T con boca de 6,2 mm.

    python vslot2020.py                # escribe vslot2020.step y .stl (100 mm de largo)
    python vslot2020.py --largo 500    # otra longitud
    python vslot2020.py --ver          # además lo muestra en OCP CAD Viewer
    python vslot2020.py --comparar     # lo compara con la STEP original (volumen común)
"""
import math

from build123d import (Edge, Face, Line, Pos, ThreePointArc, Vector, Wire, extrude)

import _comun

MITAD = 10.0                    # perfil de 20 x 20
RADIO_ESQUINA = 1.5
BOCA = 3.1                      # semianchura de la boca de la ranura (6,2 mm)
LABIO = 1.5                     # grosor del labio (de y = 8,5 a 10)
CHAFLAN_BOCA = 1.2              # chaflán de entrada de la ranura (45°)
CAVIDAD_ANCHO = 6.0             # semianchura máxima de la cavidad en T
CAVIDAD_FONDO = 4.5             # y del fondo en el eje central (la V baja hasta 4,0)
RADIO_CAVIDAD = 0.5             # redondeos de la cavidad
RADIO_PUNTA = 0.2               # redondeos del fondo en V

# Taladro central roscable: cuatro lóbulos de radio R_MENOR unidos por cuatro puntas de R_MAYOR.
R_MENOR = 2.75
R_MAYOR = 3.55
ANG_MENOR = math.degrees(math.atan2(1.258, 2.445))      # semiángulo de cada arco menor (27,2°)
ANG_MAYOR = math.degrees(math.atan2(1.910, 2.992))      # ángulo de los puntos mayores al eje (32,5°)

# Lado superior (y = +10), mitad derecha, de la esquina hacia el centro. Cada paso es
# ("L", fin) o ("A", fin, centro): arco que acaba en `fin` con ese centro.
_MITAD_DERECHA = [
    ("L", (4.3, 10.0)),                         # borde exterior hasta el chaflán de la boca
    ("L", (BOCA, 8.8)),
    ("L", (BOCA, 8.5)),                         # boca de la ranura
    ("L", (5.5, 8.5)),                          # bajo el labio
    ("A", (6.0, 8.0), (5.5, 8.0)),
    ("L", (6.0, 7.268)),
    ("A", (5.854, 6.914), (5.5, 7.268)),
    ("L", (3.586, 4.646)),                      # diagonal a 45° hacia el fondo
    ("A", (3.232, 4.5), (3.232, 5.0)),
    ("L", (0.954, 4.5)),
    ("A", (0.854, 4.473), (0.954, 4.3)),
    ("L", (0.1, 4.038)),                        # la V del fondo
    ("A", (-0.1, 4.038), (0.0, 4.211)),         # punta, cruza el eje: ya es la otra mitad
]


def _v(p):
    return Vector(p[0], p[1], 0)


def _giro(p, grados):
    c, s = math.cos(math.radians(grados)), math.sin(math.radians(grados))
    return (p[0] * c - p[1] * s, p[0] * s + p[1] * c)


def _espejo(p):
    return (-p[0], p[1])


def _arco(a, b, centro):
    """Arco corto de `a` a `b` con ese centro (todos los del perfil son menores de 180°)."""
    r = math.dist(a, centro)
    m = (a[0] + b[0] - 2 * centro[0], a[1] + b[1] - 2 * centro[1])
    n = math.hypot(*m)
    mid = (centro[0] + r * m[0] / n, centro[1] + r * m[1] / n)
    return ThreePointArc(_v(a), _v(mid), _v(b))


def _lado():
    """Pasos del lado superior, de (8,5; 10) a (-8,5; 10): mitad derecha y su espejo invertido."""
    pasos, ini = [], (MITAD - RADIO_ESQUINA, MITAD)
    derecha = []
    for p in _MITAD_DERECHA:
        derecha.append((p[0], ini, p[1], p[2] if len(p) > 2 else None))
        ini = p[1]
    # la última (punta) ya cruza el eje: la mitad izquierda es el espejo de las anteriores
    pasos.extend(derecha)
    for tipo, a, b, c in reversed(derecha[:-1]):
        pasos.append((tipo, _espejo(b), _espejo(a), _espejo(c) if c else None))
    return pasos


def _borde(tipo, a, b, c):
    return Line(_v(a), _v(b)) if tipo == "L" else _arco(a, b, c)


def perfil() -> Face:
    """Sección 20 x 20 con el taladro central, centrada en el origen."""
    lado = _lado()
    aristas = []
    for k in range(4):
        g = 90 * k
        for tipo, a, b, c in lado:
            aristas.append(_borde(tipo, _giro(a, g), _giro(b, g), _giro(c, g) if c else None))
        # esquina redondeada: del final de este lado al inicio del siguiente
        fin = _giro(lado[-1][2], g)
        sig = _giro(lado[0][1], g + 90)
        centro = _giro((-(MITAD - RADIO_ESQUINA), MITAD - RADIO_ESQUINA), g)
        aristas.append(_arco(fin, sig, centro))
    exterior = Wire(aristas)

    # Taladro: arcos menores sobre los ejes, arcos mayores sobre las diagonales.
    hueco = []
    for k in range(4):
        base = 180 - 90 * k                                  # eje del lóbulo menor
        pm = [(R_MENOR * math.cos(math.radians(base + s * ANG_MENOR)),
               R_MENOR * math.sin(math.radians(base + s * ANG_MENOR))) for s in (1, -1)]
        eje_mayor = base - 45                                # diagonal siguiente (en sentido horario)
        pM = [(R_MAYOR * math.cos(math.radians(eje_mayor + s * (45 - ANG_MAYOR))),
               R_MAYOR * math.sin(math.radians(eje_mayor + s * (45 - ANG_MAYOR)))) for s in (1, -1)]
        hueco.append(_arco(pm[0], pm[1], (0.0, 0.0)))        # arco menor (sentido horario)
        hueco.append(Line(_v(pm[1]), _v(pM[0])))
        hueco.append(_arco(pM[0], pM[1], (0.0, 0.0)))        # arco mayor
        hueco.append(Line(_v(pM[1]), _v(_siguiente_menor(base))))
    return Face(exterior, [Wire(hueco)])


def _siguiente_menor(base):
    a = base - 90 + ANG_MENOR
    return (R_MENOR * math.cos(math.radians(a)), R_MENOR * math.sin(math.radians(a)))


def vslot2020(largo: float = 100.0, centrado: bool = True):
    """Perfil V-Slot 2020 de `largo` mm a lo largo de Z (centrado en Z=0 si `centrado`)."""
    assert largo > 0
    pieza = extrude(perfil(), amount=largo)
    return Pos(0, 0, -largo / 2) * pieza if centrado else pieza


def comparar(origen: str):
    """Separación máxima entre el contorno de la STEP original y el nuestro (sección en Z=0)."""
    from build123d import Plane, Vertex, import_step
    ref = import_step(origen).intersect(Plane.XY)[0]
    mia = Pos(8.5, 10.0, 0) * perfil()                       # la STEP está desplazada (8,5; 10)

    def separacion(a: Face, b: Face) -> float:
        peor = 0.0
        for wa, wb in zip(sorted(a.wires(), key=lambda w: -w.length), sorted(b.wires(), key=lambda w: -w.length)):
            for e in wa.edges():
                for t in [i / 10 for i in range(11)]:
                    peor = max(peor, Vertex(*tuple(e @ t)).distance_to(wb))
        return peor

    print(f"área original {ref.area:.3f} mm2 | nuestra {mia.area:.3f} mm2 "
          f"(diferencia {abs(ref.area - mia.area) / ref.area * 100:.3f} %)")
    print(f"separación máxima entre contornos: original->nuestro {separacion(ref, mia):.4f} mm, "
          f"nuestro->original {separacion(mia, ref):.4f} mm")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--largo", type=float, default=100.0)
    ap.add_argument("--out", default=str(_comun.CARPETA))
    ap.add_argument("--ver", action="store_true")
    ap.add_argument("--comparar", action="store_true")
    args = ap.parse_args()
    pieza = vslot2020(args.largo)
    _comun.exportar(pieza, "vslot2020", args.out)
    if args.comparar:
        comparar(str(_comun.CARPETA / "3663_2025012894452_2020 V-slot 100mm.STEP"))
    if args.ver:
        _comun.mostrar(pieza, nombres=["V-Slot 2020"], colores=["silver"])
