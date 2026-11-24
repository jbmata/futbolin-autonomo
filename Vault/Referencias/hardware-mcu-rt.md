# Hardware MCU-RT: estado y evidencias

- Fecha de comprobación: 2026-11-24.
- Procedencia: **Comprobado** en los archivos KiCad y mediante comprobaciones locales;
  **Fuente externa** para las capacidades de JLCPCB enlazadas en la documentación técnica.
- Alcance: revisión de esquema y colocación inicial, sin ruteado ni pruebas físicas.

## Fuente técnica
La documentación de conexiones, pinout, alimentación, montaje, fuentes de fabricante y
limitaciones está en [Hardware/MCU-RT/README.md](../../Hardware/MCU-RT/README.md).
Esta nota enlaza ese documento; no lo sustituye.

| Archivo | Uso |
|---|---|
| [MCU-RT.kicad_pro](../../Hardware/MCU-RT/MCU-RT.kicad_pro) | Proyecto KiCad 10 y reglas generales |
| [MCU-RT.kicad_sch](../../Hardware/MCU-RT/MCU-RT.kicad_sch) | Esquema actual, una hoja A1 |
| [MCU-RT.kicad_pcb](../../Hardware/MCU-RT/MCU-RT.kicad_pcb) | Fuente de la colocación actual, con vínculos al esquema |
| [MCU-RT.kicad_dru](../../Hardware/MCU-RT/MCU-RT.kicad_dru) | Prohibición de vías especiales y regla local de J17 |
| [Pinout.csv](../../Hardware/MCU-RT/Pinout.csv) | 65 asignaciones de GPIO/AF, incluidas diez reservas Ethernet |
| [check_design.py](../../Hardware/MCU-RT/check_design.py) | Comprobador de esquema, pinout, huellas y PCB en fase de borrador |
| [ERC.rpt](../../Hardware/MCU-RT/checks/ERC.rpt) | Informe ERC del esquema |
| [MCU-RT.pdf](../../Hardware/MCU-RT/checks/MCU-RT.pdf) | Esquema exportado, con C1033 |
| [PCB-DRC.json](../../Hardware/MCU-RT/checks/PCB-DRC.json) | Informe DRC y paridad de la colocación compactada |
| [PCB-placement.png](../../Hardware/MCU-RT/checks/PCB-placement.png) | Vista de la colocación, sin pistas |

## Qué se ha hecho
- Restaurado C1033 y actualizado su comprobación: [[2026-11-24-condensador-reset]].
- Transferidos los 210 componentes físicos, sus campos, conexiones de pads e identificadores
  del esquema a la PCB; se preservan las asociaciones para actualizarla desde el esquema.
- Colocación inicial compactada a 160 × 140 mm, cuatro capas y perfil de fabricación
  económica: [[2026-11-24-pcb-cuatro-capas-jlcpcb]].
- Activadas comprobaciones de separación de cobre y taladros; ajustados mínimos de vías
  y serigrafía. Las advertencias de serigrafía se conservan para corregirlas.
- Los scripts auxiliares de colocación fueron temporales y no forman parte del repositorio.
  Para continuar se edita el `.kicad_pcb` vigente; no se debe reconstruir sobre él desde un
  generador temporal que pueda borrar cambios posteriores.

## Comprobaciones realizadas
Desde la raíz del repositorio:

```powershell
python Hardware/MCU-RT/check_design.py
```

Requiere KiCad 10 y la base de datos MCU de STM32CubeMX. Admite `--kicad-cli` y `--cube-db`
si las instalaciones están en rutas distintas; ver la documentación técnica.

| Comprobación | Resultado del 2026-11-24 |
|---|---|
| ERC | 0 errores y 10 avisos de etiquetas Ethernet reservadas |
| Paridad entre esquema y PCB | 0 diferencias |
| DRC de colocación | Sin cortos, errores de separación, contorno ni solapes de courtyards |
| Conexiones sin rutear | 499 |
| Serigrafía | 363 avisos: 199 solapes y 164 sobre cobre |
| Pistas y planos de cobre | 0 segmentos de pista y 0 zonas |
| Huellas de PCB | 210; redes e identificadores verificados tras la colocación |

También se hicieron pruebas puntuales con archivos temporales: el comprobador rechazó
una placa de seis capas; las reglas permitieron una vía pasante y rechazaron una vía ciega.
No son una batería de pruebas persistente del repositorio.

**Que el comprobador pase no significa que la PCB pueda fabricarse**: en esta fase permite
conexiones sin rutear y avisos de serigrafía. No se ha pasado una revisión DFM de JLCPCB,
ni se han generado archivos de fabricación finales o realizado un pedido.

## Qué no está validado
Ruteado, apilado definitivo, impedancias USB, retornos, corriente y temperatura, protección
de ramas, frenado regenerativo, parada de emergencia, medidas mecánicas y fijaciones,
consumo USB/Pi, integridad de I²C con cables reales, módulos AS5600, contactos de finales
de carrera a 3,3 V y firmware. Ver la lista de trabajo en [[Pendientes]] y las limitaciones
eléctricas en el README técnico.

El precio final depende de cantidad, acabado, opciones, montaje si se contrata, envío e
impuestos. El área se ha reducido; todavía no hay cotización ni precio de fabricación confirmado.
