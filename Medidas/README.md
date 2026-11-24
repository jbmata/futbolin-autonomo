# Medidas de la mesa

`mesa.json` es la única fuente de medidas de la mesa. La leen el simulador
(`Modelo/world.py`, mesa `FUTBOLIN_MESA=mesa`, la de por defecto) y el 3D de
`Mecanica/3d/futbolin_actualizado`. Cada valor lleva su fuente:

| Fuente | Significado |
|---|---|
| `real` | medido en la mesa |
| `plano` | plano de futbolinmadrid.com, «Medidas de las barras de un futbolín Madrid» (guardarlo aquí como `plano_barras_futbolinmadrid.jpg`) |
| `cad` | sacado del 3D actual |
| `ficha` | ficha de un fabricante |
| `estimado` | supuesto, sin medir |

Al medir algo, cambia `valor` y pon `"fuente": "real"`.

## Cómo se colocan las barras

Las cuatro barras del robot (y las del rival, igual desde su lado; la mesa es simétrica):

- **Agujeros de los jugadores**: los del plano, medidos desde la punta del mango (lado de los
  motores).
  - Portero 110 cm: 40 · 20,2 · 20,2 · 28. Jugador en el agujero central.
  - Defensa 124 cm: 58 · 23 · 44.
  - Delantero 115 cm: 47 · 17,2 · 17,2 · 34.
  - Media 105 cm: 35 · 11,5 · 11,5 · 11,5 · 11,5 · 23.
- **Topes**: en cada lado de cada barra, de dentro a la pared, va el jugador del extremo (en el
  portero, el pasador), un taco de goma de 20 mm y un muelle de 30 mm que toca la guía. Se
  considera que el muelle no se comprime.
  - Portero: pasadores en los agujeros de los extremos (a 40 y 80,4 cm).
    Carrera = distancia entre guías − 404 − 2 × (radio del pasador + goma + muelle).
  - Defensa, medios y delanteros: el jugador de cada extremo.
    Carrera = distancia entre guías − distancia entre los jugadores de los extremos
    − 2 × (medio ancho de la figura en la barra + goma + muelle).
- **s = 0**: la barra contra el tope del lado de los motores. El simulador mide todo desde ahí.

Con los valores de ahora: portero 239,6 · defensa 388,6 · medios 158,6 · delanteros 274,6 mm.

## Qué medir en la mesa real

| Medida | Cómo | Campo de `mesa.json` |
|---|---|---|
| Distancia entre guías | De la cara interior del casquillo (guía) de una pared a la de la otra, en una barra | `campo.distancia_entre_guias` |
| Ancho y largo de la plancha | Zona plana de juego, sin las bandas inclinadas | `campo.ancho_plancha`, `campo.largo_plancha` |
| Ancho de la portería | Entre postes, por dentro | `campo.ancho_porteria` |
| Taco de goma | Largo a lo largo de la barra | `topes.largo_taco_goma` (medido: 20) |
| Muelle | Largo sin comprimir | `topes.largo_muelle` (medido: 30) |
| Figura en la barra | Ancho de la figura a lo largo de la barra, donde la atraviesa | `figura.ancho_en_barra` |
| Pasador del portero | Diámetro del pasador | `topes.radio_pasador` (la mitad) |
| Carrera de cada barra (comprobación) | De tope a tope, con el muelle tocando sin comprimir. Debe salir igual que la calculada | — |
| Posición de cada barra a lo largo | Del eje de la barra a la línea central del campo | `barras.*.x_eje_desde_centro` |
| Figura | Del eje a la suela, del eje a la cabeza, ancho y largo del pie, alto del taco, holgura de la suela sobre el campo | `figura.*` |
| Bola | Diámetro y peso | `bola.*` |

## Cuando cambies una medida

1. Edita `mesa.json` (valor y fuente).
2. El simulador la toma solo (`Modelo/world.py`). Los modelos entrenados con otra mesa dejan de valer.
3. Aplícala al 3D y exporta el GLB (Blender 5.3):

       cd Mecanica/3d/futbolin_actualizado
       blender --background futbolin_madrid.blend --python aplicar_medidas.py -- --glb futbolin_madrid.glb

   Mueve cada barra (con sus guías) a su posición, coloca los jugadores a mitad de carrera,
   pone en cada extremo el taco de goma y el muelle (y los pasadores del portero) y fija el
   recorrido de cada barra. Si la plancha o la distancia entre guías del 3D no coinciden con
   `mesa.json`, avisa y no cambia nada: entonces hay que rehacer la mesa del 3D.
4. Regenera el STEP: `volcar_blend.py` y después `exportar_step.py` (ver sus cabeceras).
5. Comprueba que el 3D, el simulador y `mesa.json` coinciden:

       Modelo/.venv/Scripts/python.exe Medidas/comprobar.py

   Compara barra a barra la carrera, cada jugador en el tope de los motores, la posición de
   la barra y que en cada tope el muelle toque la guía. Sale con error si algo no coincide.

## Pendiente (no está en `mesa.json`)

- **Polea del motor lineal**: el montaje usa una GT2 de 20 dientes (40 mm por vuelta) y el
  simulador (`Modelo/hardware.py`) sigue con 30 dientes. Es parte del tema de motores, sin tocar.
- **Masas**: carro, motor de giro, campana, acople, varilla y figuras, para `hardware.py` y
  `world.py`. Pesarlas.
- **Figura**: la del 3D es la del modelo genérico (unos 111 mm de alto); el simulador usa la
  de las fichas (71 mm del eje a la suela, 48 a la cabeza). Medir una figura real.
- **Montaje con motores** (`Mecanica/3d/futbolin_con_motores`), a medio actualizar. Con estos topes,
  en el tope interior el mango queda a 11–15 mm de la pared (el portero la atravesaría) y el actuador
  necesita 50 mm: el robot solo podría recorrer portero 182,3, defensa 349,8, medios 119,8 y
  delanteros 239,8 mm de los 239,6 / 388,6 / 158,6 / 274,6 que permite la mesa. Falta decidir si se
  limita el recorrido del robot, se rediseña el soporte de pared o se cambian los topes. Si se limita,
  el simulador tendrá que distinguir la carrera de la mesa de la del robot.
- **Vídeo** de `futbolin_actualizado/video`: es de la mesa anterior.
