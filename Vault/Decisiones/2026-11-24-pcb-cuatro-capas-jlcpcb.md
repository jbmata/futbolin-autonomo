# Decisión: máximo cuatro capas y fabricación económica en JLCPCB

- Fecha: 2026-11-24
- Estado: adoptada
- Procedencia: **Comprobado**: el usuario limita la PCB a cuatro capas como máximo y pide
  que pueda fabricarse a buen precio en JLCPCB.

## Contexto
El esquema MCU-RT estaba completo como borrador, pero la PCB estaba vacía. Se inició una
colocación de 210 componentes en 240 × 160 mm. El usuario precisó después el límite de
capas y el objetivo de coste, por lo que se compactó antes de empezar a rutear.

## Decisión
- **No superar cuatro capas**. La PCB actual tiene cuatro y `check_design.py` rechaza más.
- Priorizar el proceso estándar de JLCPCB y reducir área sin eliminar componentes, validaciones,
  refrigeración ni las necesidades de potencia y desacoplo.
- **Comprobado**: el borrador actual mide **160 × 140 mm**, con dos filas de cuatro TMC2209.
  Ocupa un 41,7 % menos de área que los 240 × 160 mm iniciales. Las dimensiones y las
  posiciones no son restricciones mecánicas definitivas.
- **Supuesto / objetivo de pedido**, pendiente de cotización: FR-4 de 1,6 mm, cobre exterior
  de 1 oz e interior de 0,5 oz, máscara verde y HASL sin plomo. El apilado exacto y las
  opciones se confirmarán antes de calcular impedancias y fabricar.
- Vías pasantes previstas de **0,60 mm / taladro 0,30 mm**, anillo mínimo 0,15 mm.
  Las reglas prohíben vías ciegas, enterradas y microvías. No se prevén vías rellenas ni
  tapadas con cobre.
- Pista y separación por defecto de 0,20 mm; mínimo absoluto 0,15 mm. Solo entre pads
  del USB J17 se permite separación de 0,15 mm. Estas reglas no dimensionan la potencia.
- Mantener serigrafía legible: altura de 1 mm y trazo de 0,15 mm como mínimo previsto.
  Sus solapes siguen pendientes de corregir.

## Por qué (y alternativas descartadas)
- Se descartan seis o más capas por el límite explícito del usuario.
- Se sustituye la colocación inicial de 240 × 160 mm por otra más compacta para reducir
  superficie. No se ha calculado un ahorro en euros ni se promete un porcentaje de ahorro.
- Se evitan procesos de vía especiales para no introducir recargos de fabricación.
- Dos capas no se han validado para este diseño; no se afirma que sean imposibles. Cuatro
  es la solución de trabajo actual dentro del máximo autorizado.
- No se fija un precio hasta tener cantidad, placa terminada, opciones y envío. Tampoco
  se ha decidido contratar montaje de componentes en JLCPCB.

**Fuente externa**: [capacidades JLCPCB](https://jlcpcb.com/capabilities/Capabilities) y
[apilados JLCPCB](https://jlcpcb.com/impedance), consultados el 2026-11-24. La revisión del
fabricante y su cotización siguen pendientes.

Documentación técnica vigente: [PCB y perfil de fabricación](../../Hardware/MCU-RT/README.md#pcb-colocación-inicial).
Estado y evidencias: [[hardware-mcu-rt]], [[Estado]], [[Pendientes]].
