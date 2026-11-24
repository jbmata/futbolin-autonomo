# Decisión: organización de `Mecanica/3d`

- Fecha: 2026-11-24
- Estado: adoptada

## Contexto
El usuario pidió organizar la carpeta 3D. En la raíz estaban mezclados los scripts de las piezas con
sus STEP, STL y PNG, los STEP descargados de terceros, los montajes y las dos carpetas de la mesa.

## Decisión
| Carpeta | Contenido |
|---|---|
| `piezas/` | `_comun.py` y un `.py` por pieza (carro, soporte NEMA17, motor 17HS4401, V-Slot, soportes lineales, acople del puño) con su STEP/STL/PNG |
| `montajes/` | `montaje.py` y `montaje_soportes_lineales2020.py` con sus salidas |
| `referencias/` | STEP descargados que no son nuestros: V-Slot 100 mm del fabricante y NEMA17 de Xem Studio (CC BY 4.0) |
| `futbolin_actualizado/` | La mesa (Blender), sin cambiar de nombre: es la referencia del 3D |
| `futbolin_con_motores/` | Los actuadores del robot montados sobre la mesa, sin cambiar de nombre |
| `copias/` | Versiones guardadas, sin tocar |

- Movido con `git mv` (se conserva el historial). Cada script escribe por defecto junto a sí mismo
  (`_comun.carpeta_del_script()`); los de `montajes/` importan las piezas de `../piezas`.
- **Comprobado** 2026-11-24: los 8 scripts funcionan desde su sitio nuevo y el V-Slot sigue a 0,004 mm
  del STEP del fabricante. Ejecutarlos sin `python -I`, que quita su carpeta del `sys.path`.

## Por qué (y alternativas descartadas)
- Separa lo que se fabrica (piezas), lo que se comprueba junto (montajes) y lo que es de terceros (referencias).
- Descartado renombrar `futbolin_actualizado` y `futbolin_con_motores` (por ejemplo, a `mesa/` y `robot/`):
  son los nombres que usa el usuario y los cita mucha documentación.
