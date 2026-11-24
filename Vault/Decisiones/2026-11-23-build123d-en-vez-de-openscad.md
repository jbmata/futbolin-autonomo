# Decisión: piezas 3D en build123d, un fichero por pieza

- Fecha: 2026-11-23
- Estado: adoptada

## Contexto
Las piezas del carro estaban en OpenSCAD. El usuario preguntó si era mejor build123d y pidió migrarlas.

## Decisión
- Las piezas de `Mecanica/3d/` se escriben en Python con build123d, un `.py` por pieza, y cada una
  genera su `.step` y su `.stl`. `--ver` las abre en OCP CAD Viewer (en VS Code: «OCP CAD Viewer: Open viewer»).
- Utilidades comunes en `Mecanica/3d/piezas/_comun.py` (desde 2026-11-24; ver [[2026-11-24-organizacion-carpeta-3d]]).
- Se borraron los `.scad`.
- La mesa completa (`futbolin_actualizado`) sigue en Blender, porque viene de un modelo de malla.

## Por qué (y alternativas descartadas)
- STEP de verdad (B-rep) para CAD y fabricación, y el mismo lenguaje que el simulador.
- Descartado: un visor HTML propio para ver las piezas. El usuario lo rechazó («al final metes mierda»):
  no añadir extras que no se han pedido.
