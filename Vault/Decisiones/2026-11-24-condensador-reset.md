# Decisión: restaurar el condensador de RESET

- Fecha: 2026-11-24
- Estado: adoptada
- Procedencia: **Comprobado** en los archivos y solicitado por el usuario.

## Contexto
Al revisar qué faltaba en el esquema MCU-RT, C1033 estaba ausente y documentado como pendiente.
El usuario pidió añadir el condensador de RESET.

## Decisión
- Restaurar **C1033, 100 nF / 16 V, huella 0805**.
- Pin 1 a la red NRST compartida con J7-5 y el pulsador SW2; pin 2 a GND.
  Queda en paralelo a SW2, en el lado externo de la resistencia R1003 de 33 Ω que llega a
  NRST del STM32. No es una conexión directa que puentee R1003.
- Incluirlo en la PCB y colocarlo cerca del circuito de RESET del MCU al cerrar el ruteado.
- `check_design.py` comprueba su presencia, valor y ambas conexiones.

## Por qué (y alternativas descartadas)
- El condensador filtra perturbaciones en RESET. **Fuente externa**: la ficha del STM32H733
  recomienda 0,1 µF en su circuito de protección de NRST.
- Se descarta dejar el condensador ausente: el usuario pidió restaurarlo y el esquema ya
  lo tenía identificado como trabajo pendiente.
- La posición y el comportamiento eléctrico reales aún no se han validado en una placa fabricada.

Referencias: [documentación del circuito](../../Hardware/MCU-RT/README.md#alimentación-y-mcu),
[ficha ST, protección recomendada de NRST](https://www.st.com/resource/en/datasheet/stm32h733vg.pdf),
[[hardware-mcu-rt]].
