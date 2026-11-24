# Estado

_Última actualización: 2026-11-24_

## Resumen
Robot que juega al futbolín **madrileño** (mesa plana, figuras con pie de taco) contra una persona.
El robot maneja las cuatro barras de un equipo con ocho motores paso a paso. El agente se entrena
en un simulador propio (`Modelo/`) pensado para pasar a la mesa real.

## Medidas de la mesa (2026-11-24)
- Fuente única: `Medidas/mesa.json`. La leen el simulador y el 3D; ver [[2026-11-24-medidas-fuente-unica]].
- Topes reales y carrera distinta por barra: ver [[2026-11-24-topes-y-carrera-por-barra]].
  Con los valores de ahora: portero 239,6 · defensa 388,6 · medios 158,6 · delanteros 274,6 mm.
  **Supuesto** en parte: distancia entre guías (748,6, del 3D), ancho de la figura en la barra (30)
  y radio del pasador (2,5) no están medidos. **Comprobado**: goma 20 mm y muelle 30 mm (los dio el usuario).
- Referencia de posición: desde el final de carrera del lado de los motores, ver [[2026-11-24-referencia-desde-tope-motores]].
- **Comprobado** 2026-11-24: `Medidas/comprobar.py` da 48 de 48 coincidencias entre 3D, simulador y `mesa.json`.
- Vista interactiva de los recorridos (privada): ver [[recorridos-artifact]].

## Simulador (`Modelo/`)
- Física con subpasos, motores paso a paso, cámara Brio con latencia y compensación de latencia
  por defecto ([[2026-11-23-compensacion-latencia]]), observación v3 (actor 186 + 54 privilegiadas).
- Desde 2026-11-24 cada barra tiene su propia carrera (`BarState.linear_range`).
- `FUTBOLIN_MESA=mesa` (por defecto, lee `mesa.json`) o `sim` (mesa antigua, para evaluar modelos viejos).
- Pruebas: 57 en `Modelo/test_training.py`. **Comprobado** 2026-11-24: pasan todas con la mesa de
  `mesa.json` (goma 20, muelle 30 sin comprimir) y con `FUTBOLIN_MESA=sim`.

## Entrenamiento
- Método que funciona: profesor heurístico + imitación DAgger. PPO empeora al alumno ([[2026-11-22-dagger-sin-ppo]]).
- Mejor resultado (mesa `sim`): profesor `base_90` +0,35 goles/min; alumno `compensado_s3` −0,05
  (media de 3 semillas −0,14); techo con la percepción del robot −0,06. Ver [[historial-entrenamientos]].
- **Todos los modelos guardados se entrenaron con la mesa `sim`**: no valen para la mesa nueva.
  Hay que reoptimizar el profesor y reentrenar el alumno.

## 3D (`Mecanica/3d/`)
- Carpeta organizada el 2026-11-24 ([[2026-11-24-organizacion-carpeta-3d]]): `piezas/` (build123d, un
  `.py` por pieza con su STEP/STL), `montajes/`, `referencias/` (STEP descargados), `futbolin_actualizado/`
  (mesa), `futbolin_con_motores/` (robot sobre la mesa) y `copias/`.
- `futbolin_actualizado/`: mesa completa en Blender. Desde 2026-11-24 sigue `mesa.json`
  (`aplicar_medidas.py`); `.blend`, `.glb` y `.step` regenerados. El render y el vídeo son de la mesa anterior.
- `futbolin_con_motores/`: regenerado el 2026-11-24 sobre la mesa nueva, con una guía por barra
  (400 / 570 / 340 / 460 mm); ver [[montaje-con-motores]]. Con los topes reales el actuador no cabe en el
  tope interior: el robot solo podría usar portero 182,3 · defensa 349,8 · medios 119,8 · delanteros
  239,8 mm. **Falta decisión del usuario.** El vídeo del montaje es todavía de la mesa anterior.

## Hardware (`Hardware/MCU-RT/`)
- Placa propia con STM32H733ZGT6, ocho TMC2209, ocho AS5600 por TCA9548A y ocho finales de
  carrera. Documentación y evidencias: [[hardware-mcu-rt]], con enlaces al README técnico y KiCad.
- **Comprobado** 2026-11-24: C1033 restaurado, 100 nF / 16 V en 0805, entre la red NRST y GND;
  ver [[2026-11-24-condensador-reset]].
- **Comprobado**: 210 huellas colocadas, contorno provisional de 160 × 140 mm y cuatro capas.
  El usuario exige **cuatro capas como máximo** y fabricación económica en JLCPCB;
  ver [[2026-11-24-pcb-cuatro-capas-jlcpcb]]. El perfil de pedido aún no está cotizado.
- **Comprobado**: pasa `python Hardware/MCU-RT/check_design.py`; ERC con 0 errores y diez
  avisos Ethernet, sin diferencias esquema/PCB ni errores de colocación distintos de serigrafía.
  Quedan **499 conexiones sin rutear y 363 avisos de serigrafía**. No hay pistas ni planos de cobre.
  La PCB no está terminada ni validada para fabricar.
- Motor de referencia 17HS4401 a 12 V. `Firmware/Control/` existe pero está vacía.

## Git
- Rama de trabajo `train` (último commit subido: `7557467`). El trabajo del 2026-11-24 (medidas,
  carrera por barra, 3D, comprobación, limpieza y esta vault) **no está commiteado**.
