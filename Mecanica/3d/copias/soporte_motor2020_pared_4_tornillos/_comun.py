"""Utilidades compartidas por las piezas: exportar a STEP/STL y mostrar en OCP CAD Viewer."""
from __future__ import annotations

import argparse
from pathlib import Path

from build123d import export_step, export_stl

CARPETA = Path(__file__).parent


def argumentos(descripcion: str) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=descripcion)
    ap.add_argument("--out", default=str(CARPETA), help="carpeta de salida (por defecto: la del script)")
    ap.add_argument("--ver", action="store_true",
                    help="mostrarlo en OCP CAD Viewer (VS Code: Ctrl+Shift+P > OCP CAD Viewer: Open viewer)")
    return ap.parse_args()


def exportar(pieza, nombre: str, out: str, stl: bool = True) -> None:
    carpeta = Path(out)
    carpeta.mkdir(parents=True, exist_ok=True)
    export_step(pieza, str(carpeta / f"{nombre}.step"))
    if stl:
        export_stl(pieza, str(carpeta / f"{nombre}.stl"), tolerance=0.01, angular_tolerance=0.1)
    print(f"{nombre}: {pieza.volume:.0f} mm3 -> {carpeta}")


def mostrar(*piezas, nombres=None, colores=None, alfas=None) -> None:
    """Envía las piezas al visor. Si el visor no está abierto, la librería lo avisa y sigue."""
    from ocp_vscode import show
    show(*piezas, names=nombres, colors=colores, alphas=alfas)

