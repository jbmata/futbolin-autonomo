# Decisión: las posiciones de las barras se miden desde el final de carrera del lado de los motores

- Fecha: 2026-11-24
- Estado: adoptada

## Contexto
Las distancias se daban desde el centro del campo o desde la pose inicial del 3D, con signos
negativos y referencias mezcladas (plancha, pared, pose del CAD). El usuario pidió poner la
referencia en el extremo del final de carrera del lado de los motores de cada barra.

## Decisión
- `s = 0`: la barra contra el tope del lado de **sus** motores (el lado del mango). En el 3D, equipo A en −y
  y equipo B en +y. `s = carrera`: contra el tope contrario.
- Las medidas de los jugadores son distancias a la banda (borde de la plancha) de ese lado, con la barra en `s = 0`.
  Todas positivas y medibles con una regla.
- Ahí irán los finales de carrera del robot.
- Internamente el simulador usa `linear_pos` con 0 a mitad de carrera (así normaliza la observación).
  Para pasar de una a otra: `world.linear_pos_from_s(barra, s)` y `world.s_from_linear_pos(barra, pos)`.
- La mesa es simétrica por rotación de 180°: el rival usa las mismas medidas desde su lado.

## Por qué (y alternativas descartadas)
- Es la referencia que tendrá el robot real (el final de carrera) y la que se puede medir en la mesa.
- Descartado: medir desde el centro del campo (salían valores negativos y confundían).
- Descartado: usar los 2 mm de diferencia que da el 3D entre las barras del robot y las del rival:
  rompía la simetría que necesita el agente, que juega con cualquiera de los dos equipos.

Relacionado: [[2026-11-24-topes-y-carrera-por-barra]].
