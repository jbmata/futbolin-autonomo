"""Montaje: carro con soporte + motor 17HS4401. Comprueba que no se tocan. Unidades: mm.

    python montaje.py          # escribe montaje.step y comprueba la interferencia
    python montaje.py --ver    # además lo muestra en OCP CAD Viewer
"""
from build123d import Compound

import _comun
from carrovsolt_nema17 import soporte_nema17
from motor_17hs4401 import motor_referencia

if __name__ == "__main__":
    args = _comun.argumentos(__doc__.splitlines()[0])
    soporte, motor = soporte_nema17(), motor_referencia()
    _comun.exportar(Compound([soporte, motor]), "montaje", args.out, stl=False)
    bb = soporte.bounding_box()
    print(f"soporte: X {bb.min.X:.2f}..{bb.max.X:.2f}  Y {bb.min.Y:.2f}..{bb.max.Y:.2f}  "
          f"Z {bb.min.Z:.2f}..{bb.max.Z:.2f}")
    choque = soporte & motor
    print(f"interferencia motor/soporte: {choque.volume if choque else 0.0:.3f} mm3 (esperado ~0)")
    if args.ver:
        _comun.mostrar(soporte, motor, nombres=["soporte", "motor"], colores=["orange", "silver"],
                       alfas=[1.0, 0.5])
