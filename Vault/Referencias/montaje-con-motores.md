# Montaje con motores (`Mecanica/3d/futbolin_con_motores`)

## Qué es
Los cuatro actuadores del robot (campana sobre el puño, motor de giro, carro sobre guía V-Slot 2020,
motor de correa y tensor) añadidos a la mesa. `preparar_piezas.py` (build123d) triangula las piezas y
`montar_blender.py` (Blender) las coloca sobre `../futbolin_actualizado/futbolin_madrid.blend` y guarda
`futbolin_con_motores.blend`, `vista_general.png`, `detalle_actuador.png` y `comprobaciones.json`.
Ver también el `LEEME.txt` de la carpeta (desactualizado a 2026-11-24).

## Cambios del 2026-11-24
- La mesa se lee directamente de `futbolin_actualizado` (antes de una copia en `original/`).
- Cada barra lleva su guía, con la longitud calculada a partir de la carrera de `mesa.json`
  (`guias.json`): portero 400 · defensa 570 · medios 340 · delanteros 460 mm.
  **Comprobado**: las posiciones del puño que usa el cálculo coinciden con las medidas en el `.blend`.

## Problema encontrado (**Comprobado** 2026-11-24)
Con los topes reales, en el tope interior el extremo interior del mango queda muy cerca de la pared:
portero −7,3 mm (la atravesaría), defensa y medios 11,2 mm, delanteros 15,2 mm. El actuador necesita
**50 mm**: por debajo, el collarín inferior choca con el soporte de pared (a 48 mm, 632 mm³ de interferencia;
a 49 mm, 314 mm³; a 50 mm, libre).

| Barra | Carrera de la mesa | Recorrido útil del robot |
|---|---:|---:|
| Portero | 239,6 mm | 182,3 mm |
| Defensa | 388,6 mm | 349,8 mm |
| Medios | 158,6 mm | 119,8 mm |
| Delanteros | 274,6 mm | 239,8 mm |

Opciones (sin decidir; las decide el usuario): limitar el recorrido del robot a lo que cabe, rediseñar
el soporte de pared y el collarín, o cambiar los topes de la mesa.

## Estado de los ficheros (2026-11-24)
- `preparar_piezas.py`: holgura de pared `HOLGURA_PARED = 50` mm (medida). **Comprobado**: motor, campana y
  carro libres en cinco posiciones del recorrido útil de cada barra.
- `guias.json` y `piezas.json`: coinciden (guías de 400 / 570 / 340 / 460 mm).
- `montar_blender.py`: no para con la pared; anota los conflictos en `comprobaciones.json`
  (`recorrido_util_robot_mm`, `mango_a_pared_en_tope_interior_mm`, `conflictos`). **Comprobado**: la mesa
  de `futbolin_actualizado` queda intacta (mismo hash antes y después).
- `futbolin_con_motores.blend`, `vista_general.png`, `detalle_actuador.png` y `comprobaciones.json`:
  regenerados el 2026-11-24 sobre la mesa nueva. 5 conflictos anotados (dos en el portero: además el carro
  entra 3,6 mm en el tramo del soporte de pared).
- `original/`: borrada (copia antigua de la mesa; se recupera de git si hiciera falta).
- **Pendiente**: el vídeo de `video/` sigue siendo del 2026-11-23 (mesa anterior).
- **Pendiente**: la decisión sobre el conflicto con la pared (arriba).
