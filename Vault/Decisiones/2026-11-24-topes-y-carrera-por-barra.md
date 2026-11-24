# Decisión: topes reales de la mesa y carrera distinta por barra

- Fecha: 2026-11-24
- Estado: adoptada

## Contexto
El modelo genérico del 3D pone los mismos topes en las ocho barras y da 115,68 mm de carrera a todas.
El usuario señaló que eso no puede ser: los medios no pueden recorrer lo mismo que la defensa.

## Decisión
Los topes son los de la mesa real, según el usuario (**Comprobado** por él):
- En cada lado de cada barra, de dentro a la pared: el jugador del extremo, un taco de goma de 20 mm
  y un muelle de 30 mm que choca con la guía de la pared.
- En el portero, en lugar del jugador, un pasador en cada agujero de los extremos (a 40 y 80,4 cm de
  la punta del mango), con su goma y su muelle. El portero va en el agujero central (60,2 cm).
- Se considera que **el muelle no se comprime**: el tope es cuando toca la guía (lo decidió el usuario).

Con eso la carrera de cada barra sale del plano y de la mesa:
- Portero: distancia entre guías − 404 − 2 × (radio del pasador + goma + muelle).
- Resto: distancia entre guías − distancia entre los jugadores de los extremos
  − 2 × (medio ancho de la figura en la barra + goma + muelle).

Con los valores de 2026-11-24: portero 239,6 · defensa 388,6 · medios 158,6 · delanteros 274,6 mm.
**Supuesto**: la distancia entre guías (748,6, del 3D), el ancho de la figura en la barra (30) y el
radio del pasador (2,5) no están medidos.

El simulador tiene ahora una carrera por barra (`BarState.linear_range`); antes había una sola constante.

## Por qué (y alternativas descartadas)
- Así cuantos menos jugadores tiene la barra, más carrera, como en una mesa real.
- Descartado: 115,68 mm en todas (modelo genérico). Dejaba huecos sin cubrir y una mesa descentrada.
- Descartado: ±70 mm en todas (estimación inicial del simulador, mesa `sim`).
- Descartado: usar el muelle comprimido (estimado en 10 mm) como tope: daría unos 40 mm más de carrera
  por barra y obligaría a aplastar el muelle.

**Consecuencia**: el montaje con motores (`futbolin_con_motores`) se diseñó para 115,68 mm. La defensa
necesita unos 390 mm de carrera, así que hay que rehacer guías, correas y carros. Lo decide el usuario.

Relacionado: [[2026-11-24-medidas-fuente-unica]], [[plano-barras-futbolinmadrid]].
