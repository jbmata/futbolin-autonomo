# Decisión: compensar la latencia de la cámara por defecto

- Fecha: 2026-11-23
- Estado: adoptada

## Contexto
El alumno (DAgger) jugaba mucho peor que el profesor. La cámara entrega la bola con 70–90 ms de
retraso y el alumno decidía con esa posición antigua. Error mediano de la bola por encima de 1 m/s:
101 mm.

## Decisión
- La observación adelanta la bola y las barras rivales al instante actual con su velocidad y la
  latencia nominal (`vision.Camera.lead`, `observation.perceived_ball`). Error mediano: 58 mm.
- Activada por defecto. `FUTBOLIN_COMPENSAR=0` la apaga.
- Hay que usar el mismo valor al entrenar, evaluar y ejecutar. Los modelos anteriores no valen con ella.
- El robot real puede hacerlo porque conoce la hora de llegada de cada fotograma.

## Por qué (y alternativas descartadas)
- **Comprobado** (run 45124636693, 200 partidos): alumno de −0,43 a −0,08 goles/min, unos +0,35.
- Descartado: más fotogramas apilados (`FUTBOLIN_HISTORY=6`), más datos o más épocas. No mejoraron.
- Descartado: detectar golpes en el filtro α-β. No ayudó.
- Descartado: que el alumno imite a un profesor que ve con la cámara (`--teacher-view camera`). No ayudó.

Ver [[historial-entrenamientos]].
