# Medidas del CAD y diferencias con el simulador

Comparación del 23/11/2026. Unidades en milímetros, salvo indicación contraria.

El montaje CAD está actualizado, pero estos cambios todavía no están trasladados
al simulador. Este documento registra las diferencias; no modifica sus parámetros.
Las medidas del CAD no sustituyen las mediciones de la mesa real.

## Diferencias principales

| Medida | Simulador actual | CAD actualizado |
|---|---:|---:|
| Plancha de juego | 1160 × 700 | 1150 × 680 |
| Separación entre defensas | 140 | 230 |
| Separación entre delanteros | 140 | 172 |
| Separación entre medios | 120 | 115 |
| Polea motriz GT2 | 30 dientes | 20 dientes |
| Avance lineal por vuelta | Aproximadamente 60 | 40 |
| Radio primitivo de la polea | 9,55 | 6,366 |
| Recorrido lineal total | 140, con límites uniformes de ±70 | 115,68 según los topes del CAD |
| Posición longitudinal de los medios | −50 y +50 | Aproximadamente −71 y +70 |

La plancha de juego no debe confundirse con la distancia entre paredes ni con
los límites de rebote de la bola. Antes de modificar esos límites en la física
hay que comprobar las bandas y las porterías de la mesa real.

La polea de 20 dientes, con paso GT2 de 2 mm, avanza 40 mm por vuelta. A las
mismas rpm produce un 33,3 % menos de velocidad lineal que la de 30 dientes.
Este es un cambio de conversión geométrica; no valida la velocidad disponible
bajo carga ni el par del motor.

## Límites de las cuatro barras del robot

Estos valores son desplazamientos respecto a la pose inicial del CAD. El signo
corresponde al eje Y de Blender; no son distancias desde un final de carrera.

| Barra del equipo A | Mínimo | Máximo | Carrera total |
|---|---:|---:|---:|
| Portero, barra 01 | −37,40 | +78,29 | 115,68 |
| Defensa, barra 02 | −78,01 | +37,67 | 115,68 |
| Medios, barra 04 | −56,68 | +59,00 | 115,68 |
| Delanteros, barra 06 | −56,68 | +59,00 | 115,68 |

No basta con sustituir el límite uniforme de ±70 por ±57,84: hay que definir
el cero y los límites de cada barra, junto con la posición de sus jugadores.
Los recorridos anteriores se obtienen de los topes del modelo 3D original,
por lo que siguen pendientes de contrastar con los recorridos reales.

## Geometría que no debe copiarse automáticamente desde Blender

Las figuras del CAD son genéricas. La malla mide aproximadamente 111,1 mm de
altura total, mientras el simulador usa una figura de 119 mm, con 71 mm desde
el eje al pie y 48 mm desde el eje a la cabeza.

Esta discrepancia no demuestra que el simulador esté mal: hay que medir la figura
real y ajustar ambos modelos a ella. Lo mismo se aplica a la altura del eje,
la holgura del pie sobre el campo y los límites de giro. El giro del vídeo es
ilustrativo y no sirve para calibrar los topes mecánicos.

La barra del CAD tiene diámetro de 14 mm. El simulador conserva una masa estimada
de varilla de 0,70 kg y una inercia de 3,7 × 10⁻⁵ kg·m², cuyo comentario de origen
supone una varilla hueca de unos 16 mm. No existe un parámetro de diámetro de
varilla que baste con cambiar: hay que comprobar su masa, longitud y sección.

## Masas e inercias pendientes

El simulador estima 0,45 kg para el conjunto desplazado con la barra, además de
la varilla y las figuras. Hay que pesar el conjunto actual de motor de giro,
carro, campana, abrazadera, acople y tornillería, y revisar esa estimación.

También falta contabilizar la inercia de giro de la campana, la abrazadera y el
acople. Las funciones actuales consideran la varilla, las figuras y el rotor
del motor, pero no representan explícitamente esos componentes añadidos.

## Archivos de referencia

- [world.py](../world.py): campo, separación de jugadores, posiciones de barras,
  recorrido uniforme, geometría de figuras, masas e inercias de las barras.
- [hardware.py](../hardware.py): radio de polea y masa desplazada estimada.
- [actuators.py](../actuators.py): conversión de órdenes y límites de movimiento.
- [Construcción del futbolín CAD](../../Mecanica/3d/futbolin_actualizado/construir_blend.py):
  dimensiones de la plancha, diámetro de barra y posiciones de jugadores.
- [Comprobaciones del montaje](../../Mecanica/3d/futbolin_con_motores/comprobaciones.json):
  carrera de las cuatro barras y márgenes del carro.
- [Montaje lineal](../../Mecanica/3d/montaje_soportes_lineales2020.py):
  referencia GT2 de 20 dientes y disposición de poleas y correa.
- [Referencia del simulador](REFERENCIA.md): hipótesis y valores de partida.

Pendiente antes de calibrar o volver a entrenar: medir geometría, ceros, topes,
masas y respuesta de los motores en la mesa real, y trasladar esos valores de
forma coherente a la física, los actuadores y la normalización de observaciones.
