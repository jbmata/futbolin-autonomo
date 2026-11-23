# Futbolín autónomo

Proyecto de investigación para desarrollar un robot que juegue al futbolín **madrileño**
contra una persona, con un agente entrenado en simulación y preparado para pasar a la mesa real.

## Componentes

- [Modelo](./Modelo/README.md): simulador físico del sistema real (motores paso a paso,
  cámara, figuras con taco), entorno Gymnasium, profesor heurístico, imitación con DAgger,
  PPO y evaluación con intervalo de confianza. La memoria técnica (valores y su fuente,
  resultados, lecciones y trabajo futuro) está en [`Modelo/docs/REFERENCIA.md`](./Modelo/docs/REFERENCIA.md).
- [Hardware/MCU-RT](./Hardware/MCU-RT/README.md): esquema KiCad con STM32H733ZGT6,
  ocho drivers TMC2209, ocho puertos AS5600 mediante TCA9548A y ocho finales
  mecánicos, con instrucciones de montaje y validación.
- `Mecanica/3d/`: piezas 3D en [build123d](https://github.com/gumyr/build123d) (un `.py` por pieza: `carrovsolt.py`, `carrovsolt_nema17.py`,
  `motor_17hs4401.py`, `vslot2020.py` y `montaje.py`). Cada uno genera su `.step`/`.stl` y admite `--ver` (OCP CAD Viewer).
  Los soportes lineales están separados: `soporte_lineal_motor2020.py` y
  `soporte_lineal_tensor2020.py`, cada uno con su STEP, STL y PNG individuales,
  sin perfil, motor ni poleas. `montaje_soportes_lineales2020.py` comprueba
  interferencias y exporta el conjunto de referencia con perfil en la subcarpeta
  `montajes/`, aparte de las piezas para fabricar.

La versión con fijación a pared mediante cuatro tornillos está conservada en
[`Mecanica/3d/copias/soporte_motor2020_pared_4_tornillos/`](./Mecanica/3d/copias/soporte_motor2020_pared_4_tornillos/),
con fuentes, dependencias, STEP, STL, imágenes y comprobaciones SHA256 de los archivos copiados.

[`Mecanica/3d/nema17_40mm_descargado.step`](./Mecanica/3d/nema17_40mm_descargado.step)
es el modelo NEMA 17 genérico de cuerpo de 40 mm descargado de
[Xem Studio](https://www.xemstudio.com/objects/nema-17-stepper/) el 23/11/2026,
licencia [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), sin modificar.
No es un plano certificado del 17HS4401 ni sustituye la envolvente actual de los
montajes; comprobarlo con el motor real antes de usarlo para ajustar piezas.

El vídeo del futbolín actual está en
[`Mecanica/3d/futbolin_actualizado/video/futbolin_animado.mp4`](./Mecanica/3d/futbolin_actualizado/video/futbolin_animado.mp4):
12 segundos a 24 fps, 1280 × 720, con giro y desplazamiento de las ocho barras y
recorrido de cámara. El `.blend` animado y `crear_video.py` están en esa misma carpeta.
Se reutiliza `futbolin_madrid.blend`, sin añadir carros ni modificar el original.
Es una animación ilustrativa, con el deslizamiento dentro de los límites del modelo;
no simula contactos ni dinámica del juego. Para volver a renderizar:
`blender --background Mecanica/3d/futbolin_actualizado/video/futbolin_animado.blend --render-anim`.

El **soporte lineal del motor** tiene un encaje frontal en U de **50 mm**, con tope
axial, para **un único perfil V-Slot 2020**, abrazado por dos caras y por debajo.
La vista superior sigue el croquis: perfil centrado entre dos mejillas, dos alas
de fijación a pared y refuerzos de transición a ambos lados. El **cuerpo del motor queda
por fuera de la mejilla derecha** y solo el resalte y el eje pasan hacia la polea.
El eje está **a la altura del centro del perfil, Z=0**, perpendicular a su longitud.
La correa discurre **por las ranuras superior e inferior del V-Slot, en el plano
central Y=0**; ambas poleas comparten altura y plano. Las mejillas del motor y del
encaje tienen **20,4 mm de separación interior**, sin canales laterales adicionales.
Lleva **dos M5 con
tuerca T en cada cara: cuatro anclajes independientes**, no tornillos que atraviesen
el perfil. El alojamiento nominal es de 20,4 mm, con 0,2 mm de holgura por cara;
comprobar ajuste de impresión y usar calces si hace falta antes de apretar.
La salida de correa tiene una **escotadura central abierta por arriba**. Cuatro nervios
triangulares laterales de 6 mm, **dos arriba y dos abajo**, enlazan las paredes altas con el encaje, desde
X=-10 hasta X=20 mm, dejando libre el canal entre ambas caras. Desde X=20 mm las
paredes llegan solo a Z=14 mm, conservando los cuatro anclajes M5.
Se ha eliminado el fondo central entre las alas traseras y el tope del perfil:
el espacio queda abierto también por debajo en esa zona. El tope axial de 2 mm
se divide en dos apoyos laterales, con un paso central de 8 mm para la correa.
Se conserva el fondo bajo el encaje del perfil.

Tres nervios de **4 mm**, dos superiores y uno inferior del lado del motor,
refuerzan las dos alas de pared de **40 × 80 × 8 mm**. Los cuatro taladros de
**6,6 mm para M6** forman un patrón de **72 × 59 mm**, con acceso frontal para
apretarlos. La pared se supone perpendicular al perfil y queda 68 mm detrás de su
extremo. Es una disposición provisional: faltan las medidas y el material de la
pared real para definir tornillería, respaldo y resistencia de la fijación.
Se ha eliminado la cartela grande inferior exterior del lado opuesto al motor.
El hueco central del encaje aloja el perfil 2020; se entrega vacío en el STEP/STL
individual porque el perfil de aluminio se coloca después. Las paredes del soporte
abrazan directamente el 2020, sin paredes interiores duplicadas ni canales laterales.

El soporte tensor es una horquilla de **90 × 28 × 32,4 mm**, con dos mejillas de
**6 mm** separadas **20,4 mm** y unidas en la punta por un puente de **6 mm**.
Abraza ambos lados del perfil con dos M5 y dos tuercas T por cara: cuatro anclajes.
La unión frontal evita que las mejillas se retuerzan de forma independiente.
El motor de traslación tiene el eje perpendicular al perfil; el soporte del motor
de giro del carro, con eje paralelo al perfil, es otra pieza.

El extremo motor admite NEMA 17: cuatro pasos M3 de 3,4 mm, patrón 31 × 31 mm y
apertura central de 22,4 mm en **ambas mejillas**. Los cuatro agujeros M3 están
**solo en la mejilla del motor**. La opuesta lleva **cuatro accesos de 8 mm**,
alineados con esos tornillos, para introducir el destornillador desde fuera.
El recorrido recto se comprueba con un vástago de 7 mm; verificar la herramienta
real y usar una punta larga si su portapuntas no cabe. Para una placa de 6 mm, M3×10 con arandela de 1 mm
deja unos 3 mm de rosca dentro del motor: verificar la profundidad del motor real.
Para los anclajes del encaje, M5×12 con arandela de 1 mm atraviesa los 6 mm de
pared y la holgura nominal de 0,2 mm; confirmar enganche en la tuerca T y que no
toquen fondo antes de apretar. Los pasos M5 son de 5,4 mm, accesibles directamente
desde el exterior. Los cuatro M6 de fijación a pared necesitan longitud según
su espesor real, que sigue sin definir.

El tensor tiene dos ranuras M5 alineadas con **10 mm de recorrido**, de tensado manual.
La referencia provisional es GT2 de 6 mm, motriz de 20 dientes/eje de 5 mm y loca
con agujero de 5 mm, pista Ø16 mm, pestañas Ø18 mm y ancho total de 10 mm. El montaje
usa **dos separadores metálicos de 5,2 mm**, uno entre cada mejilla y la pista
interior del rodamiento, contando las arandelas interiores, para centrar la polea.
El puente frontal queda al menos a **5,6 mm** de la envolvente de la correa,
incluso con el tensor en su posición más adelantada.
Su plano central es Y=0 y el eje está a Z=0 mm en el montaje; el plano central
de la polea motriz se ajusta a 16,2 mm hacia dentro desde la cara del motor,
con el cubo orientado hacia el motor. Son referencias
provisionales que dependen de las poleas reales. Usar arandelas y una
tuerca autoblocante en el eje; apretar sobre las pistas interiores, sin frenar el
rodamiento. La longitud del tornillo del eje se elige con la polea, arandelas y
tuerca reales (M5×45 es un punto de partida para la horquilla doble, no una longitud validada).

El STL del motor se orienta **con la cara de sus alas sobre la cama, Z=0**; el STEP
mantiene las coordenadas de montaje. Revisar soportes en el laminador para el
interior del encaje. El STL del tensor apoya la punta unida sobre la cama y deja
las dos mejillas hacia arriba. Cada STEP/STL
contiene únicamente su soporte: el perfil y motor solo están en el montaje aparte.
La imagen `montajes/motor_exterior2020.png` muestra soporte, motor exterior y polea.
`montajes/vista_superior2020.png` muestra el perfil central en azul y la correa
en verde. Son vistas
de referencia, no piezas para imprimir.
`montajes/cara_opuesta2020.png` muestra los cuatro accesos para el destornillador.
`montajes/refuerzos_inferiores2020.png` permite revisar los dos nervios inferiores.
`montajes/detalle_encaje2020.png` muestra el 2020 en azul dentro de su alojamiento central.
`montajes/seccion_correa2020.png` muestra los dos ramales en las ranuras superior e inferior.
El montaje incluye la envolvente continua de la correa de 6 mm, sin detalle de
dientes, para comprobar su recorrido; tampoco es una pieza para fabricar.
La boca nominal de la ranura es de 6,2 mm: quedan 0,1 mm por lado para una correa
de 6 mm. Comprobar anchura real, alineación y ausencia de rozamiento antes del uso.

Antes de imprimir, medir motor, poleas, tuercas, tornillos y espacio de montaje;
las envolventes comerciales no son piezas para fabricar. El perfil de 300 mm es
solo una muestra. Reservar los 50 mm ocupados por el encaje al calcular el recorrido
del carro; su paso libre y la fijación de correa siguen pendientes de comprobar.
El encaje y los nervios rigidizan la unión, pero no garantizan flecha nula del
perfil: depende de longitud, sección, carga y apoyos. El tensor actual no tiene
anclaje a pared. Para apoyar ambos extremos hace falta definir y fijar también ese
extremo. No se ha calculado resistencia, flecha, fluencia ni temperatura en servicio,
ni probado el conjunto físico. [Referencia de cálculo de flecha de MISUMI](https://uk.misumi-ec.com/en/techblog/product-usage/aluminum-groove-profile-deflection-calculation/).

Referencia de disposición: [actuador V-Slot de OpenBuilds](https://builds.openbuilds.com/builds/v-slot%C2%AE-nema-17-linear-actuator-belt-driven.80/).

## Cómo empezar

```bash
cd Modelo
pip install -r requirements.txt
python visualize.py --teacher models/teacher/base_90.json     # un partido en el navegador
python -m unittest test_training                              # pruebas
```

## Ramas y entrenamiento en GitHub

- `train` lleva el código y los entrenamientos; `dev` se pone al día con él cuando se estabiliza.
  En `train` hay workflows que **solo se lanzan al cambiar un fichero de disparo**
  (`.github/entrenar.txt`, `.github/lanzar-entrenamiento.txt`, `.github/optimizar-profesor.txt`
  o `.github/alumnos.txt`), así los entrenamientos no ensucian el historial de `dev`.
- Los resultados (modelos, métricas, paneles y partidos en HTML) salen como artefactos de
  cada ejecución y no se escriben en el repositorio.

## Estado

El simulador todavía no controla la electrónica directamente. Los parámetros de motores,
cámara y mecánica son estimaciones; el diseño electrónico requiere validación eléctrica y
mecánica antes de fabricar o alimentar. Los modelos entrenados dependen de esas estimaciones
y habrá que repetirlos cuando se midan.
