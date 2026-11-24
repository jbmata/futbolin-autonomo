# Motores: tema aparcado

- Desde: 2026-11-22
- **El usuario pidió explícitamente no tocarlo**: no editar `Modelo/hardware.py` ni `Modelo/actuators.py`
  por velocidades, polea, masa del carro o corriente del driver sin que lo pida.

## Lo que hay ahora en el simulador
- Polea GT2 de 30 dientes supuesta (radio 9,55 mm): 0,45 m/s y unos 10 m/s² en el deslizamiento.
- Velocidad de crucero = 0,5 × 94 rad/s = 47 rad/s en los dos ejes (`RobotSpec.firmware_vmax`).
- Carro de 0,45 kg estimado.

## Lo que dice el montaje
- El montaje lineal (`Mecanica/3d/montajes/montaje_soportes_lineales2020.py`) usa una GT2 de **20 dientes**
  (40 mm por vuelta). Con 20 dientes el simulador daría 0,30 m/s y unos 15 m/s².

## Al retomarlo
Confirmar con el usuario la polea real, la masa del carro y si los motores de giro y de deslizamiento
llevan límites distintos. Con las carreras nuevas (hasta unos 390 mm en la defensa), cruzar la barra
lleva casi 1 s a 0,45 m/s.
