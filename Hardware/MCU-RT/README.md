# MCU-RT: STM32H733, ocho TMC2209 y ATX

Abre `MCU-RT.kicad_pro` con **KiCad 10**. El esquema vuelve a utilizar el **STM32H733ZGT6 integrado en la propia PCB**, en lugar de la NUCLEO-H753ZI. Conserva el regulador, cristal de 24 MHz, ocho TMC2209 V1.3, ocho conectores AS5600 mediante TCA9548A, ocho finales de carrera y conectores actuales. El USB-C es el bloque aportado por el usuario, con las correcciones descritas abajo. El circuito Ethernet se ha retirado.

Todo queda en **una hoja A1**, con bloques agrupados y etiquetas locales. El formato A1 permite incluir los circuitos recuperados sin dispersarlos en otras hojas. La PCB sigue pendiente de colocación y ruteado; este cambio afecta al esquema, bibliotecas, pinout y documentación.

## Contenido

- [Alimentación y MCU](#alimentación-y-mcu)
- [USB-C y selección de alimentación](#usb-c-y-selección-de-alimentación)
- [Programación ST-LINK y UART de depuración](#programación-st-link-y-uart-de-depuración)
- [Raspberry Pi](#raspberry-pi)
- [Motores y UART](#motores-y-uart)
- [Encoders AS5600](#encoders-as5600)
- [Finales de carrera](#finales-de-carrera)
- [Validación y trabajo pendiente](#validación-y-trabajo-pendiente)

## Alimentación y MCU

| Elemento | Conexión |
| --- | --- |
| J1 | ATX de 24 pines: entrada ATX_5V, seleccionada por JP1 para alimentar la lógica y U10; sus 3,3 V alimentan MCU, encoders y finales. |
| JP1 | Selector de alimentación, tira macho 1×3 a 2,54 mm: 1 ATX_5V, 2 +5V de placa, 3 USB_5V después de D2. |
| J2 | EPS/CPU de 8 pines (4+4): 12 V y GND para motores y ventiladores. Usar el extremo de placa del cable suministrado con la fuente. |
| J3 | Interruptor mantenido entre PS_ON# y GND; cerrado enciende la ATX. |
| J4 / J5 | Ventiladores de 12 V: dos pines genéricos a 2,54 mm. Pin 1 GND; pin 2 VMOTOR_BUS. Velocidad fija, sin RPM ni PWM. |
| U10 | LD39200PU33R, regulador de 5 V a 3,3 V. EN unido a entrada, SENSE a salida, GND y pad térmico a GND; PG sin conectar. |
| U5 | STM32H733ZGT6, LQFP144 de 20 × 20 mm, paso de 0,5 mm. |
| C1 / D1 | 470 µF/35 V y SMCJ13A en el bus de motores. |

Los 12 V del ATX principal quedan separados de los del EPS; no se ponen en paralelo. **Los 3,3 V de la ATX no se unen a los 3,3 V generados por U10** y solo llegan a TP3. Los puntos de prueba de ATX_3V3, ATX_12V_MAIN, 5VSB y PWR_OK se conservan. GND es común a lógica, motores, conectores y Raspberry Pi. La Raspberry Pi recibe directamente ATX_5V por J10-2 y J10-4, antes de JP1; no usa el LDO ni la alimentación USB de programación.

Se mantiene U10 para generar los mismos 3,3 V del STM32 tanto con ATX como al programar desde USB sin ATX. Con una alimentación exclusivamente ATX podría utilizarse su salida de 3,3 V en otro diseño, pero ya no se conservaría este modo de programación sin añadir otra alimentación de 3,3 V y su selección. El LDO no implica que la ATX carezca de una salida regulada.

El STM32 recibe 3,3 V en VDD, VDDA, VBAT, PDR_ON y VDD33USB. Se recuperan el desacoplo, el filtrado de VREF mediante 47 Ω y 1 µF + 100 nF, y los dos VCAP con **2,2 µF cada uno, ESR <100 mΩ**. VCAP no se conecta a 3,3 V ni alimenta periféricos externos. VDD33USB tiene 1 µF + 100 nF propios. Colocar estos componentes junto a sus patillas al diseñar la PCB.

Y10 es el cristal HSE de **24 MHz** recuperado, con condensadores de carga de 10 pF. Seleccionar su referencia real y ajustar las cargas a su CL y a las capacidades parásitas. SW1 permite elevar BOOT0, que tiene pull-down de 10 kΩ para arrancar desde flash. SW2 lleva RESET directamente a GND; el reset mantiene pull-up de 10 kΩ y 33 Ω hacia NRST del MCU. El usuario retiró C1033: falta el condensador de 100 nF de RESET, pendiente de restaurar.

Los 2 A de U10 son el límite del componente, no una corriente continua garantizada en cualquier PCB. Comprobar carga de MCU, multiplexor y módulos, caída de tensión y disipación: aproximadamente **(5 − 3,3) × I**. Conectar el pad térmico y dimensionar cobre/vías según la ficha de ST. Los pasivos del regulador son 0805, con capacidad efectiva verificada para la referencia que se compre.

## USB-C y selección de alimentación

**J17, USB-C**, conserva la geometría y conexiones del bloque aportado por el usuario:

- PA11, patilla 103: D− a A7/B7 a través de R32; PA12, patilla 104: D+ a A6/B6 a través de R33.
- **R32/R33 = 0 Ω en 0805**. ST indica que el PHY interno FS ya gestiona la impedancia; no se añaden 22 Ω externos.
- **R34 = 33 kΩ ±1 %** entre VBUS y PA9; **R35 = 82 kΩ ±1 %** entre PA9 y GND, ambas 0805. La detección toma VBUS antes del diodo y del selector. Configurar PA9 como **entrada GPIO**, sin pulls, y gestionar conexión/desconexión USB por firmware. Con este divisor no activar el detector dedicado `OTG_GCCFG.VBDEN`, según AN4879 Rev.13.
- R4 y R3 son las dos Rd de **5,1 kΩ independientes**, de CC1 y CC2 a GND.
- Pantalla y masas a GND. El símbolo local `MCU-RT:USB_C_USB4085` conserva el dibujo del usuario y numera la pantalla **SH**, coincidente con la huella estándar GCT USB4085.
- D2 conserva el Schottky LSM115JE3/TR13: pin 2 ánodo a VBUS y pin 1 cátodo a USB_5V. Símbolo y huella están incluidos en las bibliotecas locales, sin depender de la antigua biblioteca externa.

**JP1 lleva un único puente**, que se cambia con ambas alimentaciones desconectadas:

| Posición | Alimentación de la lógica | Uso |
| --- | --- | --- |
| **1–2** | ATX_5V → +5V de placa → U10 → 3,3 V. | Funcionamiento normal con ATX. El USB sigue disponible para datos y detección. |
| **2–3** | VBUS USB → D2 → USB_5V → +5V de placa → U10 → 3,3 V. | Programar por ST-LINK sin ATX y comunicar por USB con firmware adecuado. La salida ATX_5V queda aislada. |
| Sin puente | Lógica sin alimentación. | No programa ni comunica. |

Los **motores y ventiladores requieren los 12 V del EPS**; USB solo alimenta la lógica y los pull-up de finales. No colocar dos puentes ni unir los tres contactos. C2 permanece en la entrada ATX_5V; C1001 permanece en la entrada de U10, después del selector. J7-1 es VTref de 3,3 V, nunca una entrada de alimentación.

Configurar **USB_OTG_HS, dispositivo, PHY interno Full-Speed, USB CDC**. No hay PHY externo ni pull-up externo en D+. Cerrar el reloj USB de 48 MHz y el árbol completo en firmware/CubeMX. El consumo de arranque, antes de enumerar, en suspensión y tras enumerar debe respetar el puerto USB; todavía no se ha medido. Usar un arranque de bajo consumo para programación y validar tensión a la entrada de U10 con la caída de D2 y del cable. El jumper no limita corriente ni garantiza cumplimiento USB por sí mismo.

**Pendiente:** el bloque USB del usuario no incluye protección ESD externa. Añadirla junto al conector antes de cerrar el diseño físico. Ethernet no está poblado en el esquema: se conservan diez etiquetas de asignación en el MCU y sus filas en `Pinout.csv`, que generan diez avisos ERC de etiqueta conectada a una sola patilla.

## Programación ST-LINK y UART de depuración

**J7** es una tira macho genérica **1×6, 2,54 mm**, para cablear un ST-LINK externo. Es un pinout propio con señales SWD; no coincide físicamente con un cable STDC14 o MIPI10.

| Pin J7 | Señal | Destino |
| --- | --- | --- |
| 1 | VTref, 3,3 V de la placa | Referencia de tensión para el programador. |
| 2 | GND | Masa común. |
| 3 | SWDIO | PA13, patilla 105, mediante 33 Ω. |
| 4 | SWCLK | PA14, patilla 109, mediante 33 Ω. |
| 5 | NRST | RESET, mediante 33 Ω a la patilla 25. |
| 6 | SWO, opcional | PB3, patilla 133. |

Alimentar la placa mediante JP1: 1–2 con ATX o 2–3 con USB; J7-1 sirve como referencia, no como entrada de alimentación de otro regulador. Adaptar por nombre de señal al ST-LINK concreto. Mantener los cables SWD cortos y comenzar a frecuencia moderada. SWO solo es necesario para traza.

**J8** es un conector genérico **1×4, 2,54 mm**, para una UART de depuración separada de los motores y de la Raspberry Pi:

| Pin J8 | Señal |
| --- | --- |
| 1 | GND |
| 2 | TX del MCU: PD8 / USART3_TX, patilla 77. Conectar a RX del adaptador. |
| 3 | RX del MCU: PD9 / USART3_RX, patilla 78. Conectar a TX del adaptador. |
| 4 | Referencia 3,3 V. No conectar la salida de alimentación del adaptador. |

Usar un adaptador USB-UART de lógica **3,3 V** o el VCP de un ST-LINK compatible. No es RS-232 ni una entrada de 5 V.

## Raspberry Pi

Se añade **J10, GPIO estándar de 40 contactos, 2×20 a 2,54 mm**, con huella `Connector_PinHeader_2.54mm:PinHeader_2x20_P2.54mm_Vertical`. Es una tira macho para conexión mediante cable/adaptador 1:1; no define todavía montaje mecánico ni fijaciones de una Raspberry Pi encima de la placa. Verificar pin 1 y orientación; la numeración es física, no BCM.

Quedan disponibles **SPI0 de la Pi hacia SPI4 del STM32** y **UART8**, para órdenes de movimiento y telemetría. La visión se procesa en el ordenador/Raspberry Pi, sin enviar imágenes al STM32. UART8 es independiente de USART3 de depuración y de los dos buses TMC.

| Contacto físico J10 | Raspberry Pi | STM32H733 |
| --- | --- | --- |
| 8 | GPIO14 / TXD | PE0 / UART8_RX, patilla 141. |
| 10 | GPIO15 / RXD | PE1 / UART8_TX, patilla 142. |
| 19 | GPIO10 / SPI0_MOSI, salida de la Pi | PE14 / SPI4_MOSI, patilla 67, AF5. |
| 21 | GPIO9 / SPI0_MISO, entrada de la Pi | PE13 / SPI4_MISO, patilla 66, AF5. |
| 23 | GPIO11 / SPI0_SCLK, salida de la Pi | PE12 / SPI4_SCK, patilla 65, AF5. |
| 24 | GPIO8 / SPI0_CE0_N, salida de la Pi | PE11 / SPI4_NSS, patilla 64, AF5. |
| 6, 9, 14, 20, 25, 30, 34, 39 | GND | GND común. |
| 1, 17 | Salidas 3,3 V de la Pi | Sin conectar. |
| 2, 4 | Entrada de alimentación 5 V de la Pi | ATX_5V directamente desde J1, antes de JP1. No 5VSB. |
| Resto | Otros GPIO / ID EEPROM | Sin conectar en esta fase. |

La Raspberry Pi se alimenta **desde los 5 V de la ATX por ambos contactos 2 y 4**, utilizando varios retornos GND. Su salida de 3,3 V permanece sin conectar y no se pone en paralelo con U10 ni con los 3,3 V de la ATX. Es una entrada de alimentación admitida en el [pinout oficial de Raspberry Pi](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html#pin-functions). No conectar simultáneamente una fuente al USB-C/micro-USB de alimentación de la Pi: este diseño une directamente sus 5 V a la ATX, sin circuito de reparto ni bloqueo de corriente inversa.

Dimensionar la rama de 5 V, contactos y cable completo para el modelo real y sus periféricos, usando ambos contactos de 5 V y suficientes masas. Como referencia, las fuentes recomendadas son 5 V/3 A para Pi 4 y 5 V/5 A para Pi 5; comprobar tensión bajo carga en la Pi y caída de cable. Un cable Dupont o cinta genérico no queda validado para esa corriente por tener paso 2,54 mm. No se ha seleccionado el cable ni medido el consumo. [Requisitos oficiales de alimentación](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html#power-supply).

Con ATX apagada y JP1 en 2–3, USB alimenta solo la lógica del MCU; **la Raspberry Pi queda apagada**. No mantener señales SPI/UART activas hacia ella: deshabilitarlas en firmware, o desconectar J10 al programar en ese estado, para evitar alimentación parásita por GPIO.

Las señales son de 3,3 V. Configurar **Pi como maestra SPI0 y STM32 como esclavo SPI4 de cuatro hilos, AF5 en PE11–PE14, NSS por hardware activo bajo**. Mantener CE0 alto fuera de transferencias y MISO en alta impedancia con NSS alto; configurar pull-up interno en NSS. CPOL/CPHA, tamaño de palabra y protocolo deben coincidir en ambos equipos. Habilitar SPI0 en la Pi (`dtparam=spi=on`) y validar inicialmente con cable corto y reloj bajo. La UART usa los pines 8/10; desactivar la consola serie de Linux en ese puerto. El firmware y protocolo siguen pendientes de implementación y prueba.

## Motores y UART

Motor de referencia elegido para los ocho canales: **17HS4401**, NEMA 17 bipolar de cuatro cables. Corriente nominal de **1,7 A por fase**, par de retención de **0,40 N·m** (40 N·cm), paso de **1,8°** (200 pasos/vuelta), resistencia de **1,5 Ω por fase** e inductancia de **2,8 mH**. Eje de 5 mm. Son los valores típicos de este modelo y los que usa el simulador (`Modelo/hardware.py`); confirmarlos con la ficha del fabricante del motor que se compre.

La referencia sirve para diseñar la base y probar el prototipo; todavía hay que validar velocidad, par durante el movimiento y temperatura con la mecánica del futbolín. Los 1,7 A por fase no son el consumo de cada motor desde la ATX ni fijan directamente el ajuste RMS del TMC2209. Ajustar la corriente del driver y comprobar su refrigeración antes de utilizar los ocho motores. Medir el consumo conjunto de motores y ventiladores para dimensionar la fuente, las pistas y los conectores.

| Motor | STEP | Temporizador | DIR | EN_N | Bus / dirección |
| --- | --- | --- | --- | --- | --- |
| 1 | PE9 | TIM1_CH1 | PG0 | PF0 | UART7 / 0 |
| 2 | PA0 | TIM2_CH1 | PG1 | PF1 | UART7 / 1 |
| 3 | PB4 | TIM3_CH1 | PG2 | PF2 | UART7 / 2 |
| 4 | PD12 | TIM4_CH1 | PG3 | PF3 | UART7 / 3 |
| 5 | PC6 | TIM8_CH1 | PG4 | PF4 | USART2 / 0 |
| 6 | PB15 | TIM12_CH2 | PG5 | PF5 | USART2 / 1 |
| 7 | PE5 | TIM15_CH1 | PG9 | PF6 | USART2 / 2 |
| 8 | PB9 | TIM17_CH1 | PG15 | PF7 | USART2 / 3 |

UART A usa **PE8/TX y PE7/RX**; UART B usa **PD5/TX y PD6/RX**. Cada TX llega al bus PDN mediante una resistencia de 1 kΩ, R1 o R2, y el RX se conecta directamente. Hay una resistencia TX por bus, compartida por sus cuatro módulos. Las direcciones se fijan con resistencias de 10 kΩ en MS1/AD0 y MS2/AD1: 00, 01, 10 y 11. No conectar los ocho módulos a un único bus.

Por decisión del usuario, la base queda **sin fusibles locales**: F1 ya estaba retirado y se elimina F2; tampoco hay fusibles individuales por motor. El EPS alimenta directamente `VMOTOR_BUS`, compartido por los ocho TMC y los dos ventiladores. Los 5 V seleccionados en JP1 alimentan la lógica de entrada y el regulador U10. La protección de sobrecorriente de la ATX no garantiza proteger pistas, conectores o ramas finas frente a un cortocircuito. Comprobar su corriente de fallo y tiempo de desconexión, y dimensionar todas las ramas según esa protección externa antes de fabricar.

Las resistencias y condensadores cerámicos de baja tensión usan huellas **0805**, incluidas las cuatro resistencias USB modificadas. Los nueve electrolíticos de potencia conservan sus huellas THT. D1 sigue en SMC. Elegir los cerámicos reales comprobando capacidad efectiva con polarización, tensión y ESR, especialmente los 10 µF del regulador y los 2,2 µF de VCAP.

Cada EN_N tiene pull-up de 10 kΩ y STEP/DIR pull-down de 100 kΩ. CLK va a GND. Se conserva el pinout **P1A** de BIGTREETECH: 1 DIR, 2 STEP, 3 CLK, 4 UART_TX, 5 UART_RX/PDN, 6 MS2, 7 MS1, 8 EN; 9 VM, 10 GND, 11 A2, 12 A1, 13 B1, 14 B2, 15 VIO y 16 GND. El pad 4 queda NC; R10 del módulo debe permanecer abierto, como en el esquema oficial. Solo se usa el pad 5 para UART.

Los motores se conectan a **J101, J201, J301, J401, J501, J601, J701 y J801**, uno por cada TMC. Son tiras macho genéricas de **cuatro pines y paso 2,54 mm**, con huella `Connector_PinHeader_2.54mm:PinHeader_1x04_P2.54mm_Vertical`. **Pin 1 = A2; pin 2 = A1; pin 3 = B1; pin 4 = B2**: 1–2 son una bobina y 3–4 la otra. Comprobar las parejas del cable del motor y la corriente admisible del conector real; el paso por sí solo no garantiza esa corriente. Cada módulo tiene 100 µF/35 V y 100 nF/50 V junto a VM, y 100 nF en VIO. Se reutiliza la huella local StepStick: 12,70 mm entre filas, paso 2,54 mm y taladros de 1,02 mm. Verificar dimensiones y altura del zócalo y del disipador real.

`Pinout.csv` recoge las **65 asignaciones de GPIO (55 conectadas y diez reservas Ethernet)**, su patilla LQFP144, función, AF y red. Se contrastan con la base oficial de STM32CubeMX para STM32H733ZGTx. Los ocho motores conservan sus asignaciones actuales. Para PB4/TIM3, configurar **SWD sin JTAG completo**; SWO en PB3 puede usarse opcionalmente para traza.

## Encoders AS5600

Se añaden **ocho conectores para módulos AS5600 externos**, uno por motor, mediante **U3 TCA9548APWR**, huella TSSOP-24 de 4,4 × 7,8 mm, paso 0,65 mm. No son ocho AS5600 montados directamente en la PCB. Los ocho sensores comparten la dirección I²C fija **0x36**: conectarlos a un mismo bus sin multiplexor produciría conflictos. U3 aísla las ramas; A0, A1 y A2 a GND fijan su dirección de siete bits **0x70**.

| Señal STM32 | GPIO / patilla LQFP144 | Conexión U3 |
| --- | --- | --- |
| ENC_I2C_SCL | PB6 / 136, I2C1_SCL, AF4 | SCL, pin 22. |
| ENC_I2C_SDA | PB7 / 137, I2C1_SDA, AF4 | SDA, pin 23. |
| ENC_MUX_RESET_N | PD4 / 118, GPIO open-drain | RESET, pin 3, pull-up de 10 kΩ a 3,3 V. |

U3, los pull-ups y los módulos usan **3,3 V de U10**, disponibles tanto con ATX como con JP1 en programación USB. C1101 = 100 nF y C1102 = 1 µF desacoplan U3. Mantener C1101 junto a VCC/GND al diseñar la PCB. R1117/R1118 son los dos pull-ups de 2,2 kΩ del bus principal; cada rama lleva dos de 4,7 kΩ, R1101…R1116. Todas las resistencias y estos condensadores son 0805.

**J1101…J1108** son tiras genéricas **1×4, paso 2,54 mm**:

| Pin | Función |
| --- | --- |
| 1 | GND. |
| 2 | +3V3 de U10. No 5 V. |
| 3 | SDA de la rama correspondiente. |
| 4 | SCL de la rama correspondiente. |

El canal 0 conecta J1101/encoder del motor 1; el canal 7 conecta J1108/encoder del motor 8. **Seleccionar un solo canal cada vez**, escribiendo `1 << canal` a 0x70; después leer el AS5600 a 0x36. No activar varias ramas con sensores de igual dirección. Al arrancar o tras RESET todas las ramas quedan deshabilitadas. PD4 permite aislar una rama bloqueada y reiniciar el multiplexor; el firmware debe contemplar timeouts y recuperación del periférico I2C1.

El módulo de la fotografía no tiene una salida de interrupción: **OUT es analógica/PWM; PGO es una entrada de programación**, aunque la serigrafía pueda parecer «GPO». En esta arquitectura se lee por I²C y OUT/PGO quedan sin usar. Fijar DIR a GND en cada módulo, mediante su resistencia existente si se verifica o un puente externo; no dejar DIR flotante. No hace falta quemar OTP para leer el ángulo.

**Verificar el módulo antes de conectarlo a J110x.** En modo 3,3 V el AS5600 requiere VDD5V y VDD3V3 unidos y desacoplo local. No basta con que el chip admita 3,3 V: comprobar la conexión de ambas patillas, el puente de selección del módulo y que sus pull-ups vayan a los mismos 3,3 V. La fotografía no permite validar su circuito completo. Adaptar el módulo si viene preparado exclusivamente para 5 V; no conectar pull-ups a 5 V a estas ramas de 3,3 V. Los condensadores deben estar en el módulo junto al AS5600.

La fotografía muestra dos resistencias «103», probablemente los pull-ups de 10 kΩ: confirmar su función y valor. Si son 10 kΩ, se ponen en paralelo con los 4,7 kΩ de cada rama; con una única rama seleccionada y los 2,2 kΩ principales, la resistencia equivalente nominal es unos **1,30 kΩ**, unos **2,54 mA** a nivel bajo con 3,3 V. Verificar tolerancias, VOL, capacitancia y tiempos de subida con el módulo y cable reales; ajustar/no montar los pull-ups de rama si hace falta. El comprobador solo contrasta los valores nominales, no mide estas condiciones.

El multiplexor limita el bus a **400 kHz** aunque el sensor admita más. Objetivo inicial de software: **500 lecturas por segundo de cada uno de los ocho motores**, pendiente de validar. Una secuencia de selección más lectura de dos bytes consume idealmente 63 pulsos por sensor, **1,26 ms para ocho a 400 kHz**, sin contar pausas, comprobaciones del imán, reintentos ni procesamiento. Leer `RAW_ANGLE` (0x0C/0x0D) y revisar el estado del imán; ajustar el filtro interno según ruido y retardo. El ángulo de 12 bits cubre una vuelta: contar las vueltas en firmware y realizar homing de los ejes lineales después de encender. Una pérdida de muestras o un giro mayor de media vuelta entre muestras hace ambiguo el contador por desenvolvimiento; validar velocidad máxima y latencia.

I²C se debe validar en el cableado real junto a los motores: usar recorridos cortos, retorno GND próximo y separar señales de potencia. Las nuevas ramas no incorporan protección ESD externa; elegirla con baja capacidad si el ensayo y montaje lo requieren. Estos cambios no validan todavía integridad de señal, alcance de cable ni la fijación/alineación del imán.

## Finales de carrera

Hay **ocho entradas mecánicas FC1…FC8**, conservando los cuatro adaptadores modificados por el usuario y añadiendo otros cuatro con el mismo circuito. Los inversores U1/U2 y C3/C4 siguen retirados. Los encoders angulares no dan por sí solos una referencia absoluta de traslación tras encender. La correspondencia entre finales y ejes se asignará al definir el montaje.

J901…J908 son conectores genéricos de **dos pines, paso 2,54 mm**, huella `Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical`. **Pin 1 = GND → COM; pin 2 = señal → NC**. Dejar NO sin conectar y aislado. Los interruptores se montan en la mecánica, unidos a la base mediante cable.

| Entrada / red | Conector | GPIO | Patilla MCU (LQFP144) | Línea EXTI |
| --- | --- | --- | --- | --- |
| LIMIT_1 | J901 | PD0 | 114 | 0 |
| LIMIT_2 | J902 | PD1 | 115 | 1 |
| LIMIT_3 | J903 | PD2 | 116 | 2 |
| LIMIT_4 | J904 | PD3 | 117 | 3 |
| LIMIT_5 | J905 | PD7 | 123 | 7 |
| LIMIT_6 | J906 | PF8 | 20 | 8 |
| LIMIT_7 | J907 | PF9 | 21 | 9 |
| LIMIT_8 | J908 | PD10 | 79 | 10 |

El contacto va **directamente al GPIO mediante el filtro**, sin conversor ni inversor: pull-up externo de **2,2 kΩ a 3,3 V en el lado del contacto**, resistencia serie de **1 kΩ** y condensador de **100 nF / 16 V a GND en el lado del MCU**. No colocar el pull-up después de la resistencia serie: formaría un divisor con el contacto cerrado. Con NC cerrado circulan aproximadamente 1,5 mA. Las constantes de tiempo nominales son 0,10 ms al cerrar y 0,32 ms al abrir. R y C son 0805. Los adaptadores modificados por el usuario no llevan los diodos ESD D901…D904; los cuatro nuevos replican esa versión, sin protección ESD externa. Es una entrada para contacto seco; no conectar 5/12 V ni una salida de sensor alimentado.

**La polaridad cambia respecto al esquema anterior:** GPIO = **0 en reposo, NC cerrado; 1 al pulsar o abrirse un cable**. Las redes se renombran de `LIMIT_x_N` a `LIMIT_x` para reflejarlo. Configurar entrada digital sin pulls internos y, si se usa interrupción, **EXTI por flanco ascendente**, con antirrebote por firmware. Las ocho líneas EXTI son distintas. PD4 se mantiene para RESET del multiplexor; PD7, PF8, PF9 y PD10 se asignan a los nuevos finales.

**Limitación del Omron SS-01GL2:** su ficha da una carga mínima de referencia de **5 V / 1 mA**. La simplificación solicitada trabaja a **3,3 V** y queda fuera de esa referencia de tensión; no se garantiza la fiabilidad de contacto a partir de esa ficha. Probar repetibilidad y fiabilidad con los interruptores reales o elegir uno especificado para 3,3 V. Si se mantiene el requisito de operar dentro de la referencia Omron de 5 V, habrá que recuperar una interfaz de 5 V compatible con el MCU y su secuencia de alimentación.

Implementar y probar antirrebote, velocidad de aproximación y distancia de frenado. Un cable abierto y un final pulsado son indistinguibles; un cortocircuito a GND no se detecta. Estos finales no sustituyen una parada de emergencia.

## Validación y trabajo pendiente

`python check_design.py` exporta una netlist nueva, ejecuta ERC y comprueba: ATX/EPS y las tres redes separadas del selector JP1; divisor y datos USB; regulador y alimentaciones del H733; BOOT, RESET (sin C1033), VCAP y cristal; los ocho TMC y motores, ocho finales directos y ocho puertos AS5600 con TCA9548A; ST-LINK, UART, Raspberry Pi UART8/SPI4 y correspondencia de patillas con pads de huellas. Contrasta las **65 asignaciones de GPIO/AF** con los XML oficiales de STM32CubeMX para STM32H733ZGTx. Las diez asignaciones Ethernet son reservas sin PHY. Admite `--kicad-cli` y `--cube-db` para otras instalaciones.

Resultado del cambio: **209 componentes físicos, 0 errores y 10 avisos ERC**. Los diez avisos corresponden exclusivamente a etiquetas Ethernet conectadas a una sola patilla; el comprobador los identifica expresamente y rechaza cualquier aviso distinto. J10-2 y J10-4 reciben ATX_5V, separado de los 5 V seleccionados para el MCU y de sus 3,3 V. Se conserva el resto de conexiones, incluidos UART8 y SPI4. PB6/PB7 se asignan a I2C1 y PD4 al RESET del multiplexor. Se conservan los cuatro adaptadores modificados por el usuario y se añaden FC5…FC8 con pull-up de 3,3 V y polaridad de apertura activa alta. La vista exportada contiene una página A1. Netlist/ERC no validan impedancias, consumo USB/Pi, integridad de señal, térmica ni el montaje físico.

La PCB conserva el archivo anterior y todavía no tiene colocación, apilado ni ruteado para esta arquitectura. Importar desde el esquema con F8 al empezar esa fase. Dimensionar cobre, retornos, vías y conectores para la corriente real. Mantener la potencia de motor alejada del MCU, cristal y comunicaciones.

La capacidad y la TVS del bus son valores iniciales para un prototipo a 12 V. Comprobar corriente conjunta, arranque, temperatura, par/velocidad, alimentación y disipación de los ocho drivers. La ATX no se ha validado para absorber frenado regenerativo: **la TVS no es un freno continuo**. Medir los transitorios y dimensionar almacenamiento/frenado antes de movimientos rápidos. La parada de emergencia y otros sensores siguen pendientes; estos finales no constituyen un circuito de seguridad. No insertar ni retirar drivers o motores con alimentación.

No se ha generado ni probado firmware, USB CDC, UART, SPI, I²C, lectura AS5600, movimiento, antirrebote o homing en hardware. La carga completa del proyecto en la interfaz de CubeMX sigue pendiente; la validación entregada usa sus bases de datos oficiales.

El proyecto Nucleo completo, incluida su huella y plantilla, queda conservado en `.history/pre-return-h733-2026-11-22.zip`. El MCU/USB/Ethernet original se conserva en `.history/pre-nucleo-backplane-2026-11-21.zip`; el esquema del usuario anterior al selector queda en `.history/pre-usb-power-selector-2026-11-22.zip`; el diseño anterior a los encoders y a las cuatro entradas directas queda en `.history/pre-eight-as5600-four-direct-limits-2026-11-22.zip`; se mantienen las demás copias históricas. El CSV y la plantilla de la Nucleo se retiran de los archivos activos para evitar confundirlos con el diseño actual.

La versión del usuario anterior a la ampliación a ocho finales queda conservada en `.history/pre-eight-direct-limits-2026-11-23.zip`.

## Referencias

- [ST STM32H733, DS13314: pines, alimentación, VCAP y USB](https://www.st.com/resource/en/datasheet/stm32h733vg.pdf).
- [ST LD39200: pinout, capacidad de salida y térmica](https://www.st.com/resource/en/datasheet/ld39200.pdf).
- [ST AN4879: USB y divisor de VBUS](https://www.st.com/resource/en/application_note/an4879-usb-hardware-and-pcb-guidelines-using-stm32-mcus-stmicroelectronics.pdf).
- [ST UM2910: señales SWD, VTref, NRST y VCP](https://www.st.com/resource/en/user_manual/um2910-stlinkv3minie-debuggerprogrammer-tiny-probe-for-stm32-microcontrollers-stmicroelectronics.pdf).
- [Raspberry Pi: GPIO de 40 pines, niveles y funciones UART](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html#gpio).
- [BIGTREETECH TMC2209 V1.3, esquema oficial](https://github.com/bigtreetech/BIGTREETECH-Stepper-Motor-Driver/blob/master/TMC2209/V1.3/Schematic/TMC2209%20V1.3-SCH.pdf).
- [Omron SS y SS-01GL2](https://omronfs.omron.com/en_US/ecb/products/pdf/en-ss.pdf).
- [TI TCA9548A: pinout, dirección, RESET, selección de canales y pull-ups](https://www.ti.com/lit/ds/symlink/tca9548a.pdf).
- [ams AS5600: ficha del fabricante, alojada por Mouser](https://www.mouser.com/pdfdocs/AMS_AS5600_Datasheet_EN.PDF).
- [TI TPD1E10B06](https://www.ti.com/lit/ds/symlink/tpd1e10b06.pdf).
- [Intel ATX, conector principal](https://edc.intel.com/content/www/us/en/design/ipla/software-development-platforms/client/platforms/alder-lake-desktop/atx-version-3-0-multi-rail-desktop-platform-power-supply-design-guide/2.0/2.01/main-power-connector-required/).
