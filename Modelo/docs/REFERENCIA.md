# Futbolín autónomo — documento de referencia

Estado a **23 de noviembre de 2026**. Recoge qué simula el proyecto, de dónde
sale cada número, qué se ha probado y qué falta por medir. El `README.md` es la
guía de uso; este documento es la memoria técnica.

---

## Estado actual (23 de noviembre de 2026)

**Qué hay entrenado.** Un alumno por imitación (DAgger, red de 256 neuronas, solo cámara y
firmware) que juega al nivel de un heurístico que viera con la misma cámara. Se evalúa con 200
partidos de 60 s contra el rival heurístico de nivel 0,7, en goles por minuto de diferencia:

| | Diferencia |
|---|---|
| heurístico por defecto | −0,32 ± 0,13 |
| **alumno (media de 3 semillas, con compensación de latencia)** | **−0,14** (de −0,05 a −0,30) |
| profesor con la percepción del robot (cámara) | −0,06 ± 0,16 |
| profesor con el estado real (referencia, no realizable) | +0,35 ± 0,13 |

- **El alumno ya está entrenado y no hay que seguir entrenándolo** con esta receta: más datos, más
  épocas, más memoria, otro profesor y PPO no han mejorado nada. Copia a su profesor tan bien como
  puede; lo que le falta ya es de **percepción** (la bola llega 70–90 ms tarde y con ruido).
- **Dónde está:** `models/alumno/compensado_s3.zip` (el mejor de tres semillas; la elección usa los
  mismos partidos de evaluación, así que el margen real es el de las semillas, ~±0,12). Se carga con
  `PPO.load("models/alumno/compensado_s3")`. El resto de modelos están solo como artefactos de
  GitHub (caducan a los 90 días).
- **Qué NO es:** no es un modelo final. Se entrenó con parámetros estimados (latencia 70 ms, polea,
  masas, corriente del driver) y hay que repetirlo cuando se midan. Con la compensación de latencia
  activada por defecto, los modelos anteriores al 23 de noviembre no sirven.
- **Siguiente:** medir la latencia real de la Brio y ver cuánto valen la latencia y los fps en
  el simulador (con el profesor con ojos del alumno); después, repetir el entrenamiento con los
  parámetros medidos. Lo demás (profesor por refuerzo, PPO) no ataca el cuello de botella actual.

---

## 1. Objetivo y sistema que se simula

Un robot juega al futbolín **Madrid** contra una persona.

| Elemento | Qué es |
|---|---|
| Mesa | Madrid, campo plano de 116 × 70 cm, 8 barras (4 por equipo) |
| Robot (equipo A) | 4 barras, 2 motores paso a paso por barra (giro y desplazamiento): 8 motores. Placa `Hardware/MCU-RT` (STM32H733 + TMC2209) |
| Rival (equipo B) | una persona |
| Sensor | cámara cenital Logitech Brio sobre la mesa |
| Control | el robot recibe 8 posiciones objetivo (lineal y ángulo por barra) a 50 Hz |

El agente es una red neuronal entrenada con **PPO** en simulación. Cada decisión
que toma debe poder reproducirse en el robot real con lo que este sabrá:
la imagen de la cámara y la cuenta de pasos de los motores.

---

## 2. Mapa del código (`Modelo/`)

| Fichero | Contenido |
|---|---|
| `world.py` | Campo, barras, geometría de la figura, masas, observaciones v1/v2 |
| `hardware.py` | Especificaciones del motor, la mecánica, la mano del rival y la cámara |
| `physics.py` | Contactos bola‑figura en 3D con masas, giro de la bola, bandas, gol |
| `actuators.py` | Motor paso a paso del robot, mano del rival, actuador ideal histórico |
| `vision.py` | Cámara: latencia, ruido, borrón, oclusión, filtro α‑β |
| `observation.py` | Observación v3: parte del actor y parte privilegiada del crítico |
| `policies.py` | Actor‑crítico asimétrico y exportación del actor a NumPy |
| `reward.py` | Recompensa con shaping de potencial |
| `env.py` | Entorno Gymnasium; rivales heurístico, ausente y de política |
| `imitation.py` | Arranque por imitación (profesor‑alumno) |
| `train.py` | PPO, currículo, self‑play, evaluación |
| `benchmark.py` | Comparación de agentes con las mismas semillas |
| `dashboard.py` | Panel web local de seguimiento (`http://127.0.0.1:8765`) |
| `example.py`, `visualize.py` | Agente heurístico y visualización |
| `test_training.py` | 34 pruebas (física, motores, cámara, entorno, entrenamiento) |

---

## 3. Modelo físico y de dónde sale cada valor

Leyenda de fiabilidad: **F** = de una ficha o medida publicada, **E** = estimado
(hay que medirlo), **D** = deducido de otros valores.

### 3.1 Mesa y bola

| Parámetro | Valor | |
|---|---|---|
| Campo interior | 116 × 70 cm | F (ficha Don Futbolín Madrid) |
| Mesa | plana | F (la pidió el usuario: la Madrid no está inclinada) |
| Bola | 35 mm, 24 g | E |
| Portería | 20 cm | E |
| Recorrido lineal de cada barra | ± 7 cm | E |
| Posiciones de las barras (x, m) | −0,50 · −0,36 · −0,20 · −0,05 · +0,05 · +0,20 · +0,36 · +0,50 | E |
| Figuras por barra | 1 · 2 · 3 · 5 · 5 · 3 · 2 · 1, separadas 12–14 cm | E |

### 3.2 Figura madrileña con pie de taco

En la vista lateral es la unión de tres rectángulos que gira con la barra:
cabeza y torso, pierna y taco macizo de suela plana. En Y mide 30 mm.

| Medida | Valor | |
|---|---|---|
| Centro de la barra → pies | 71 mm | F |
| Centro de la barra → cabeza | 48 mm | F |
| Anchura máxima | 30 mm | F |
| Taco: largo en X × alto | 30 × 20 mm | E |
| Semigrosor de pierna y de torso | 8 y 10 mm | E |
| Holgura suela‑campo | 3 mm (eje a 74 mm) | E |
| Masa de una figura | 40 g | E |
| Masa de la varilla | 0,70 kg | E |

Fuentes de las cotas F: fichas de jugadores madrileños de
[Billares Balsalobre](https://www.billaresbalsalobre.com/tienda/futbolin/jugador-futbolin/billaresbalsalobre-jugador-futbolin-madrileno-madera-natural/)
y [Billares y Juegos](https://www.billaresyjuegos.com/jugadores-munecos-futbolines-jugador-muneco-futbolin-madera-pasador-rojiblanco-estandar-medidas-p-318.html)
(71 / 48 / 119 mm, 30 mm de ancho; el largo y el alto del taco no vienen en
ninguna ficha).

**Consecuencias geométricas**
- Con el jugador girado más de ~71° la bola pasa por debajo del taco.
- Con la barra cerca de 180° baja la **cabeza** hasta 26 mm del campo y toca la
  bola. Por eso el golpe se arma a unos −110° y no a −180°.

### 3.3 Física de la bola (`physics.py`)

| Fenómeno | Valor | |
|---|---|---|
| Subpasos de física por paso de control (20 ms) | 6 | — |
| Resistencia de rodadura | 0,15 m/s² constante + 0,05/s viscosa | E |
| Restitución bola‑banda | 0,65 | E |
| Restitución bola‑figura | 0,70 | E |
| Fricción de Coulomb bola‑figura | 0,30 | E |
| Fricción bola‑campo deslizando | 0,25 | E |
| Fricción bola‑banda (frena el giro) | 0,20 | E |
| Rampas de esquina | 0,6 m/s² en los 5 cm junto al fondo, fuera de la boca | E |
| Velocidad máxima de la bola | 8 m/s | — |

Cómo se resuelve:
- **Contacto**: punto más cercano de cabeza, pierna o taco al centro de la bola,
  en 3D. El impulso incluye las masas de la bola y de la barra, así que una
  barra ligera retrocede y golpea con menos fuerza.
- **Giro de la bola**: tras un golpe seco desliza hasta rodar y pierde 2/7 de su
  velocidad. En las bandas el rozamiento frena parte del giro.
- **Pisar la bola**: si la suela presiona desde arriba, la frena y el motor se
  bloquea (el taco no atraviesa la bola).
- **Bola atrapada** entre dos figuras: se detiene y los motores se bloquean.
- **Simetría A/B**: cuando la bola toca varias barras a la vez se resuelve la
  más cercana, no la de menor índice. Hay un test que lo garantiza.
- **Bola parada**: si no se mueve 1 cm en 0,6 s fuera del alcance de las
  figuras (3 s si está a su alcance), recibe un empujón hacia el centro.

Aleatorización por partida: rodadura, restituciones, fricciones, masa de la
bola (±10 %), par del motor (×0,8–1,1), inercia (±15 %) y rozamiento (×0,5–2) de
los motores, latencia de la cámara (×0,8–1,3).

### 3.4 Motores del robot (`actuators.py`, `hardware.py`)

Cada eje es un NEMA17 con TMC2209. Se modela el motor, no un actuador ideal:

```
T = −T(ω)·sin(N·(q − q_cmd)) − c·(ω − ω_cmd) − rozamiento
```

El rotor sigue al campo magnético como un muelle sinusoidal. Si una carga lo
separa más de 2 pasos completos, el motor **salta y pierde pasos**. El firmware
sigue creyendo que está donde mandó (**lazo abierto**, sin encoder).

| Parámetro | Valor | |
|---|---|---|
| Par de retención | 0,40 N·m | E |
| Velocidad de codo / máxima | 19 / 94 rad/s | E |
| Inercia del rotor | 5,7·10⁻⁶ kg·m² | E |
| Dientes del rotor | 50 | F (motor de 1,8°) |
| Polea del eje lineal | GT2 de 30 dientes, radio 9,55 mm | E |
| Masa desplazada con la barra | 0,45 kg más figuras y varilla | E |
| Rozamiento de giro / lineal | 0,010 N·m / 4,0 N | E |
| Firmware: fracción de par que usa / de velocidad máxima | 50 % / 50 % | E |
| Latencia PC → placa | 10 ms | E |
| Encoders | no (`RobotSpec(encoders=True)` los activa) | — |

Capacidades resultantes (`python hardware.py`):

| Barra | Aceleración de giro | Velocidad de giro | Punta del pie | Lateral |
|---|---|---|---|---|
| GK (1 figura) | ~1.600 rad/s² | 47 rad/s | 3,3 m/s | 0,45 m/s |
| DEF (2) | ~950 rad/s² | 47 rad/s | 3,3 m/s | 0,45 m/s |
| ATK (3) | ~680 rad/s² | 47 rad/s | 3,3 m/s | 0,45 m/s |
| MID (5) | ~440 rad/s² | 47 rad/s | 3,3 m/s | 0,45 m/s |

Un golpe armado a unos −110° saca la bola a **~2,4–2,6 m/s** (medido en los
tests).

### 3.5 Rival humano

| Parámetro | Valor | |
|---|---|---|
| Velocidad / aceleración lateral | 1,5 m/s / 25 m/s² | E |
| Velocidad / aceleración de giro | 45 rad/s / 1.500 rad/s² | E |
| Inercia añadida por la mano | ×3 | E |
| Retardo de reacción | 60 ms (nivel 1) a 200 ms (nivel 0) | E |

El **nivel** (0–1) se sortea en cada partida dentro de un rango según la fase.
Con menos nivel el rival reacciona más tarde, se coloca con más error, golpea
menos y mueve las manos más despacio.

### 3.6 Cámara (`vision.py`)

Logitech Brio, 720p a 90 fps (~0,9 mm por píxel con 1.280 px sobre 1,16 m).

| Parámetro | Valor | |
|---|---|---|
| Latencia hasta tener la detección | 70 ms (×0,8–1,3 por partida) | **E, hay que medirla** |
| Ruido de posición de la bola | 1,5 mm | E |
| Exposición (borrón de la bola rápida) | 4 ms | E |
| Fallo de detección | 1 % + hasta 50 % si la bola va muy rápida | E |
| Ruido de las barras: posición / ángulo | 2 mm / 0,08 rad | E |
| Filtro α‑β | α = 0,6, β = 0,25 | E |

Además: la bola **no se ve cuando está bajo una figura** (vista desde arriba), y
la velocidad es la estimada por el filtro, no la real.

---

## 4. Observación, acción y recompensa

### Acción
8 dimensiones en [−1, 1]: `[lineal, ángulo] × (GK, DEF, MID, ATK)`, posiciones
objetivo. `ángulo > 0` lleva el pie hacia delante. Para el equipo B se invierten
Y y el ángulo, de modo que un mismo agente puede jugar con cualquiera.

### Observación v3 (240 dimensiones)

| Parte | Dims | Contenido |
|---|---|---|
| **Actor** | 186 = 62 × 3 fotogramas | por fotograma: bola según la cámara (x, y, vx, vy, visible, tiempo sin verla) 6 · barras propias según el firmware 16 · barras rivales según la cámara 16 · barras propias según la cámara 8 · bola relativa a cada barra propia 8 · acción anterior 8 |
| **Crítico** (solo en entrenamiento) | +54 | bola real con su giro 6 · las 8 barras reales 32 · bola relativa real 8 · error de la estimación propia 8 |

Las versiones v1 (36 dims) y v2 (44 dims) siguen disponibles con estado perfecto.

### Recompensa

| Componente | Valor |
|---|---|
| Gol marcado / recibido | +10 / −10 |
| Tiro a puerta propio / del rival | +0,5 / −0,25 |
| Shaping de potencial: posición y velocidad de la bola hacia la portería rival | pesos 1,0 y 0,5 |
| Shaping de potencial: proximidad jugador‑bola (solo fases 0 y 1) | 0,5 y 0,2 |
| Cambio brusco de acción | −0,001 × \|Δacción\| |

El shaping es de **potencial** (F = γΦ(s′) − Φ(s)): no cambia la política óptima y
el agente no puede acumularlo dando vueltas. Ya no se penaliza |acción|: la
acción es una posición objetivo y esa penalización empujaba las barras al centro.

---

## 5. Algoritmo y entrenamiento

- **PPO** (Stable Baselines3) con **actor‑crítico asimétrico**. El actor
  (186 → 256 → 256 → 8) solo ve lo que tendrá el robot; el crítico
  (240 → 256 → 256 → 1) ve además el estado real y solo existe en entrenamiento.
- **Memoria**: 3 fotogramas apilados y la acción anterior (fase del golpe,
  aceleraciones, latencia).
- **Arranque por imitación** (`imitation.py`): el heurístico juega con estado
  perfecto y sin retardo; el alumno lo imita viendo solo cámara y firmware, con
  ruido DART en las acciones. Sin él, PPO desde cero no descubre la secuencia
  armar‑golpear.
- **Currículo**: fase 0 rival ausente sin ruido (10 %); fase 1 heurístico de nivel
  0,2–0,6 (20 %); fase 2 nivel 0,4–1,0 (35 %); fase 3 self‑play con 50 %
  heurístico y 50 % copias congeladas (35 %).
- **Hiperparámetros**: 8 entornos, 1.024 pasos por entorno y actualización, lote
  512, 10 épocas, γ = 0,99, λ = 0,95, clip 0,2, `target_kl` 0,03, tasa de
  aprendizaje 1e‑4 → 3e‑5, std inicial 0,3, normalización solo de la recompensa.
- **Selección del mejor modelo**: diferencia de goles contra el heurístico de
  nivel 0,7, siempre con las mismas semillas.
- **Ritmo medido** en el PC del proyecto (4 núcleos, 8 hilos): ~620–700 pasos/s,
  unos 1 h 40 min para 4 M de pasos.

### Cómo reproducir

```bash
cd Modelo
.venv/Scripts/python.exe hardware.py                    # capacidades del robot
.venv/Scripts/python.exe imitation.py --out models/bc_init_v3 --episodes 360
.venv/Scripts/python.exe train.py --timesteps 4000000 --n-envs 8 --curriculum --self-play \
    --init-model models/bc_init_v3 --save-dir models/ppo_v4 --log-dir logs/ppo_v4 --eval-episodes 100
.venv/Scripts/python.exe dashboard.py --log-dir logs/ppo_v4 --total 4000000
.venv/Scripts/python.exe benchmark.py --episodes 200 --skill 0.7 models/ppo_v4/best/best_model
.venv/Scripts/python.exe -m unittest test_training
```

No se debe editar el código mientras entrena: cada fase vuelve a cargar los
módulos en los procesos de entorno.

---

## 6. Resultados hasta ahora

Todas las cifras son goles por partida contra el heurístico, con 60–100 partidas.
Con tan pocos goles por partida (~0,1) las diferencias pequeñas son ruido.

| Referencia | Marca | Recibe | Nota |
|---|---|---|---|
| Heurístico contra heurístico (nivel rival 0,7) | 0,12 | 0,12 | simulador realista |
| Imitación (`bc_init_v3`), solo cámara y firmware | 0,10 | 0,15 | arranque del PPO |
| Modelos PPO antiguos (`best`, `ppo_v2`), física anterior | 0,07 · 0,03 | 0,10 · 0,07 | no comparable: física y observación distintas |

**Evaluación final de `ppo_v4`** (4 M de pasos, 150 partidas nuevas, semillas
20000–20149, rival heurístico de nivel 0,7; con ~0,1 goles por partida, diferencias
de ±0,03 no son significativas):

| Agente | Marca | Recibe | Diferencia | Partidas con gol |
|---|---|---|---|---|
| Heurístico (el profesor) | 0,12 | 0,08 | +0,04 | 20 % |
| Imitación `bc_init_v3` (punto de partida) | 0,11 | 0,11 | 0,00 | 23 % |
| `ppo_v4/best/best_model` (mejor de la fase 3) | 0,11 | 0,11 | −0,01 | 22 % |
| `ppo_v4/phase2/best_model` (mejor de la fase 2) | 0,11 | 0,17 | −0,05 | 28 % |
| `ppo_v4/foosball_ppo_final` | 0,10 | 0,18 | −0,08 | 28 % |

**Conclusión: 4 M de pasos de PPO no mejoraron lo que la imitación ya daba.**
El mejor modelo guardado empata con el punto de partida y el modelo final es algo
peor (recibe más goles). Durante el entrenamiento la mejor evaluación parecía
mejor (+0,5 en la fase 2) porque se medía con solo 20 partidas, que es ruido.
Los tiros a puerta subieron de ~0,5 a ~0,8 por partida, pero se convertían en
gol casi igual de poco.

Qué no es la causa: la memoria de 3 fotogramas, el crítico asimétrico y la
exploración funcionaron (no hay colapso ni inestabilidad). La hipótesis más
probable es que el profesor es un techo bajo (apenas gana +0,04 a un rival de
nivel 0,7) y que marcar es tan raro que la recompensa por gol da muy poca señal.

**Cambio de planteamiento tras esos resultados (22 de noviembre)**

Con partidos que acababan en el primer gol (~0,1 goles por partida) no se podía
distinguir una mejora real del azar: el margen de error de 120 partidos era de
±0,09 goles, del mismo tamaño que las diferencias que se buscaban. Se cambió a:

- **Partido continuo**: el juego sigue tras cada gol y saca el portero que lo ha
  recibido. Un partido de 30 s tiene decenas de situaciones de gol. `benchmark.py`
  informa de goles por minuto con intervalo de confianza del 95 %.
- **Profesor optimizado con CMA‑ES** (`optimize_teacher.py`): 14 parámetros del
  heurístico del robot, 19 generaciones en 40 min, partidos de 30 s contra rivales
  de nivel 0,4 / 0,7 / 1,0. Validación con partidos nuevos (240 por conjunto, goles
  por minuto):

| Parámetros del heurístico | Marca | Recibe | Diferencia (IC 95 %) | Tiros a puerta |
|---|---|---|---|---|
| Por defecto | 0,24 | 0,48 | −0,24 ± 0,17 | 6,8 |
| **Optimizado** | 0,39 | 0,20 | **+0,19 ± 0,16** | **12,6** |

La mejora (+0,43 goles/min) es significativa. Con la validación pequeña de 72
partidos no lo era (+0,36 ± 0,41), y el «mejor candidato» de la búsqueda salió
peor que el de partida (−0,56): con 12 partidos por candidato, el mejor de 14 es
sobre todo el más afortunado. Los parámetros son `models/teacher/heuristic_optimized.json`.

- **Imitación con DAgger** (`dagger.py`) y PPO a partir de ese alumno, con las
  variantes entrenadas en paralelo en GitHub Actions (rama `train`). Los
  resultados de ese entrenamiento se añadirán aquí cuando existan.

**Resultado del entrenamiento en GitHub y de la optimización por roles (22 de noviembre)**

Evaluación con 100 partidos de 60 s contra el heurístico de nivel 0,7 (goles/min):

| Agente | Diferencia (IC 95 %) |
|---|---|
| Heurístico por defecto | −0,50 ± 0,22 |
| **Profesor optimizado (ve el estado real)** | **+0,16 ± 0,11** |
| Alumno DAgger, red 128 | −0,24 ± 0,15 |
| Alumno DAgger, red 256 (3 semillas) | −0,17 ± 0,16 · −0,32 ± 0,18 · −0,05 ± 0,21 |
| Alumno DAgger, red 512 | −0,27 ± 0,18 |
| Tras PPO (mejor guardado), todas las variantes | de −0,63 a −1,30 |
| Tras PPO (modelo final), todas las variantes | de −0,67 a −1,55 |

- **DAgger funciona** (mucho mejor que el heurístico por defecto) pero queda por debajo del
  profesor: lo que se pierde por ver solo cámara y firmware.
- **PPO empeora al alumno en las 5 variantes**: olvida lo copiado y la recompensa escasa no
  le da nada mejor. Habrá que anclarlo al alumno (pérdida de imitación o límite KL) o dejarlo fuera.
- **El tamaño de la red (128, 256, 512) no cambia el resultado**: las diferencias están dentro
  de la variación entre semillas.

Optimización del profesor por roles (4 trabajos de 150 min en GitHub, cada uno un rol): solo
los delanteros mejoran en su validación (+0,46 ± 0,28 frente a +0,21 ± 0,22); portero, defensa
y medios no superan a los parámetros de partida (con la búsqueda de la defensa sin converger).
Juntos y validados con 240 partidos de 30 s:

| Profesor | Marca | Recibe | Diferencia (IC 95 %) | Tiros/min |
|---|---|---|---|---|
| Anterior (mismo juego para los 4 roles) | 0,47 | 0,14 | +0,33 ± 0,16 | 12,3 |
| Por roles | 0,57 | 0,20 | +0,37 ± 0,16 | 14,1 |

La diferencia en goles (+0,04) no es significativa; los tiros suben un 15 %. Conclusión: con
esta forma de búsqueda el profesor está en una meseta de +0,2 a +0,4 goles/min. Sigue siendo
ruido de medida: dos validaciones del mismo profesor anterior dieron +0,19 y +0,33.

**Reglas de juego y del robot añadidas el 22 de noviembre** (todo lo anterior a esto se
entrenó sin ellas y hay que repetirlo):

- **Giro de las barras**: de −90° a +90° (0 = vertical), y el **portero de −10° a +90°**, que
  es un tope mecánico: apenas puede echar el pie atrás. En el equipo B los ángulos están
  girados 180°.
- **Levantar es siempre a −90°** (pie atrás: la bola pasa por debajo, con 59 mm libres);
  el ángulo positivo se usa solo para golpear. El portero no se levanta atrás: queda en
  vertical. Un profesor anterior levantaba a −159°, donde la **cabeza** de la figura tapa
  la bola, y bloqueaba sus propios pases y el saque.
- **Giro en lazo cerrado** (encoder): el MCU corrige los pasos perdidos en cada ciclo. El
  eje lateral va en lazo abierto. Sin nada, una barra estaba desfasada más de 5° el 66 % del
  tiempo y llegaba a 156°; con final de carrera, el 32 %; con encoder, el 0 %.
- **Huecos hacia la portería** (`aim_on`): cualquier barra, también defensa y medios, busca
  un hueco entre los rivales y tira recto o en diagonal. Se apunta descentrando la figura
  respecto a la bola y barriendo la barra al golpear; los ajustes por rol salen de medir el
  simulador (`AIM_TABLE`: de unos 5° a 12°, con ±1,5° a ±3° de dispersión). El portero tira
  siempre recto.
- **Motor 17HS4401 con TMC2209 a 12 V**: el par sale del modelo eléctrico (1,7 A, 1,5 Ω,
  2,8 mH, 40 N·cm), no de una curva supuesta: 0,28 N·m parado y 0,18 N·m al golpear
  (47 rad/s). A 24 V serían 0,28 N·m al golpear.
- **Optimizador**: los tiros a puerta pesan 0,04 (antes 0,10, que los hacía valer unas seis
  veces más que un gol) y cuentan los del rival restados.
- **Visor** (`visualize.py`): reproductor HTML con planta y alzado de cada partido.

Comparación hecha con esas reglas (90 partidos de 30 s, mismo profesor, con y sin buscar
huecos): sin huecos +0,67 ± 0,28 goles/min, con huecos +0,56 ± 0,29. Sin diferencia clara, y
con huecos tira menos (11,6 frente a 14,6 tiros/min): sus parámetros (`gap_wait`,
`lane_margin`) aún no están optimizados.

**Cadena completa con las reglas nuevas (23 de noviembre, run 45064712873, 3 h 50 min)**.
100 partidos de 60 s contra el heurístico de nivel 0,7, goles por minuto (IC 95 %):

| Agente | Marca | Recibe | Diferencia |
|---|---|---|---|
| heurístico por defecto | 0,17 | 0,57 | −0,40 ± 0,16 |
| profesor (estado real) | 0,61 | 0,24 | **+0,37 ± 0,19** |
| alumno DAgger, cámara (3 semillas) | 0,30–0,37 | 0,62–0,74 | −0,32 / −0,37 / −0,32 (± 0,20–0,23) |
| alumno DAgger, sensores ideales | 0,47 | 0,32 | **+0,15 ± 0,20** |
| tras PPO con poco ruido (`best`) | 0,37–0,48 | 0,59–0,73 | −0,35 / −0,25 / −0,20 |
| tras PPO con poco ruido (final) | 0,29–0,42 | 0,81–0,93 | −0,52 / −0,51 / −0,47 |
| tras PPO con ruido normal (`best` / final) | 0,37–0,40 | 0,89–1,03 | −0,49 / −0,66 |

- **El optimizador no mejoró al profesor.** Con las reglas nuevas, de las tres semillas de
  CMA‑ES, dos acabaron devolviendo el punto de partida (`base_90.json`) y la tercera quedó peor
  (+0,32 frente a +0,49 con 240 partidos). El profesor usado es `base_90.json`, ajustado a mano.
  La meseta de +0,2 a +0,4 se confirma.
- **La cámara es lo que más cuesta.** El alumno con sensores ideales queda en +0,15 y los tres con
  cámara en −0,32 a −0,37: unos 0,5 goles/min por la percepción (latencia, ruido, oclusión) y unos
  0,2 por la imitación en sí. Es una sola semilla en la variante ideal, y los intervalos se tocan.
- **PPO no ayuda.** Con poco ruido (log‑std −2,5) el mejor punto no se separa del alumno inicial
  (todas las diferencias caen dentro del margen), y el modelo final empeora siempre. Con el ruido
  normal empeora más (−0,66). El ruido de exploración explica parte del daño, pero no todo.
  Además `best_model` se elige con pocas partidas de evaluación, así que su mejora aparente
  es en parte sesgo de selección.
- **Defensa**: los alumnos reciben 2,5–3 veces más goles que el profesor; es donde se pierde.

**Causa probable y siguiente prueba (lanzada el 23 de noviembre, workflow `alumnos.yml`).** La
cámara entrega la bola con 70–90 ms de retraso (la latencia varía en cada partida) y la
observación no la adelanta al instante actual: con la bola a más de 1 m/s, la que ve el alumno
está a **101 mm** (mediana) de la real; las figuras miden 30 mm. Con `FUTBOLIN_COMPENSAR=1` se
adelantan bola y barras rivales con su velocidad y la latencia **nominal** (lo que podrá hacer el
sistema real: conoce la hora de llegada de cada fotograma) y el error baja a **58 mm**. Lo que
queda es casi todo irreducible: durante ~70 ms tras cada golpe la cámara aún no lo ha visto.
Detectar golpes en el filtro α‑β apenas cambia nada (49 mm de mediana, la misma media), así que no se usa.
`FUTBOLIN_HISTORY=N` cambia los fotogramas apilados (3 por defecto). Las dos variables cambian lo
que ve la red: hay que usar el mismo valor al entrenar, evaluar y ejecutar el modelo. Variantes:
referencia, compensado, compensado con 6 fotogramas, compensado con 3× partidos y todo junto,
evaluadas con 200 partidos (margen ±0,14). PPO queda fuera.

**Resultado (run 45124636693, 200 partidos de 60 s, nivel 0,7, goles/min):**

| Alumno | Marca | Recibe | Diferencia |
|---|---|---|---|
| profesor (estado real) | 0,58 | 0,23 | +0,35 ± 0,13 |
| referencia (sin compensar) | 0,30 | 0,73 | −0,43 ± 0,16 |
| **compensado** | 0,46 | 0,54 | **−0,08 ± 0,16** |
| compensado, 6 fotogramas | 0,32 | 0,62 | −0,31 ± 0,13 |
| compensado, 3× partidos | 0,39 | 0,49 | −0,10 ± 0,14 |
| compensado, 6 fotogramas, 3× partidos, más épocas | 0,50 | 0,56 | −0,07 ± 0,16 |

La compensación sube al alumno unos **0,35 goles/min** (los márgenes no se tocan) y **es ahora el
valor por defecto**. Más datos, más épocas o más memoria no añaden nada; con 6 fotogramas, solo, empeora.
Quedan ~0,4 goles/min hasta el profesor.

**Techo con esta percepción (run 45142565146, 200 partidos de 60 s, nivel 0,7, goles/min):**

| Agente | Marca | Recibe | Diferencia |
|---|---|---|---|
| profesor (estado real) | 0,58 | 0,23 | +0,35 ± 0,13 |
| **profesor con los ojos del alumno** (bola y barras rivales por la cámara) | 0,47 | 0,53 | **−0,06 ± 0,16** |
| alumno `compensado`, semillas 1 / 2 / 3 | 0,46 / 0,41 / 0,41 | 0,54 / 0,71 / 0,46 | −0,08 / −0,30 / −0,05 |
| alumno que imita al profesor con ojos del alumno, semillas 1 / 2 / 3 | 0,38 / 0,43 / 0,42 | 0,66 / 0,65 / 0,62 | −0,28 / −0,22 / −0,20 |

- **El hueco con el profesor es de percepción, no de imitación.** El mismo heurístico pasa de
  +0,35 a −0,06 solo por ver la bola y las barras rivales con la cámara; el alumno (media de
  tres semillas: −0,14) queda en ese techo, o algo por debajo. La diferencia se nota sobre todo en
  defensa (recibe 0,53 frente a 0,23).
- **Imitar al profesor con ojos del alumno no ayuda** (media −0,23, dentro del ruido de −0,14).
- **El ruido entre semillas es grande** con la misma receta (−0,05 a −0,30): la cifra de −0,08 de
  la primera ronda era una semilla afortunada. Una sola semilla no sirve para comparar variantes.
- Lo que queda por mejorar es **percibir mejor** (latencia, fotogramas por segundo, ruido y
  oclusión de la cámara) o un profesor/política que juegue bien con información retrasada
  (PPO no lo ha conseguido).

---

## 7. Lecciones aprendidas

- Antes de entrenar, medir heurístico contra heurístico **con el agente en A y en
  B**. Hubo dos asimetrías: el orden de resolución de contactos por índice de
  barra y un comando lineal que el heurístico del equipo B invertía dos veces.
- Un heurístico sin retardo de percepción defiende casi perfecto (~0,03 goles por
  partida): se añadió retardo de 60–200 ms.
- Contra un rival estático (pie abajo) el portero y la defensa cubren toda la
  portería: la fase 0 usa un rival *ausente* (figuras levantadas).
- Armar el golpe con la bola acercándose hace que el taco que se retira la
  arrastre hacia atrás (autogoles).
- Con la cabeza modelada, armar a 180° hace que la cabeza toque la bola.
- Una barra ligera (el portero) absorbe los tiros en vez de rebotarlos.
- Un motor paso a paso sin encoder **pierde pasos** al pisar la bola o recibir un
  tiro fuerte (unos 80–100 por partida en simulación): conviene un encoder
  magnético por eje (p. ej. AS5600).
- PPO desde cero no descubre la secuencia armar‑golpear; la imitación sí la
  arranca.
- Con la bola casi siempre sin marcar, la selección del mejor modelo era ruido:
  se evalúa contra un rival de nivel 0,7 y no contra el máximo.

---

## 8. Qué medir en el hardware real

Por orden de impacto:

1. **Latencia real de la Brio** (extremo a extremo, filmando un LED que se enciende
   desde el PC). Se ha supuesto 70 ms.
2. **Masa de una figura y de la varilla**. Si las figuras son de aluminio (~150 g
   en lugar de 40 g) la inercia sube mucho y el robot es mucho más lento.
3. **Motor y tensión de alimentación**: par a la velocidad de trabajo, polea,
   reducción de la barra de giro. Se ha supuesto 24 V y 1,5 A.
4. **Largo y alto del taco**, anchura de la portería, recorrido de cada barra,
   distancias entre figuras.
5. **Velocidad de la bola** tras un tiro y su deceleración (con vídeo a 90 fps o
   más): sirve para calibrar restitución y rozamientos.

Cada valor se cambia en `world.py` o `hardware.py` (marcados con `shortcut:` los
estimados) y el entrenamiento se repite.

---

## 9. Líneas pendientes

- **Mejorar el profesor** (el heurístico apenas supera a un rival de nivel 0,7) y
  repetir la imitación, idealmente con DAgger. Es la primera prioridad tras los
  resultados del apartado 6.
- **Recompensa más guiada** hacia la ocasión de gol (pases entre barras).
- **Entrenar varias configuraciones en paralelo** en GitHub Actions (repositorio
  público, 4 vCPU, sin cuota): semillas, redes de 128/256/512, PPO contra SAC.
  Sin repetir semillas no se distingue mejora real de azar.
- **Portar el actor al robot**: `policies.actor_weights()` y `actor_forward()`
  usan solo NumPy (~100 k pesos, < 1 ms por decisión); también cabe en el
  STM32H733 (~316 kB en float32, ~80 kB en int8).
- Calibrar el simulador con vídeo de la mesa real y, más adelante, con partidas
  grabadas del robot (identificación del sistema).
- Descartado por ahora: entrenar con vídeos de internet. Habría que reconstruir
  el estado desde cámaras laterales y las personas no se mueven como los motores.

### 9.1 Futuro: profesor por refuerzo (no implementado)

**Estado**: idea acordada el 22 de noviembre de 2026; se hará **después** de ver los resultados
de la cadena que corre esa noche en GitHub (rama `train`, workflow `pipeline.yml`). No hay
código ni se ha probado nada.

**Por qué**: hoy el profesor es un programa escrito a mano con 14 parámetros por rol
(`env.HeuristicParams`) y está en una meseta de +0,2 a +0,4 goles/min: ajustar sus
parámetros con CMA‑ES ya no la rompe. Como el alumno lo imita, su techo es el del profesor.
Un profesor que aprende por refuerzo podría jugar cosas que su estructura no contempla:
pases entre barras, cerrar ángulos, esperar el momento del tiro.

**Cómo**
1. **Información privilegiada**: el profesor ve el estado real de la mesa (sin cámara, sin
   retardo, sin oclusión), así que le es mucho más fácil aprender que al alumno. Después el
   alumno lo copia con DAgger viendo solo cámara y firmware (el esquema habitual de
   «profesor‑alumno» de la robótica con patas).
2. **Partir del heurístico**: primero se imita al heurístico actual con ese estado real y
   luego se afina con PPO. Hereda así las reglas de buen juego sin describirlas.
3. **Anclarlo al heurístico** mientras afina (pérdida de imitación auxiliar o límite de
   cambio de la política). La prueba de PPO sobre el alumno empeoró en las 5 variantes; la
   hipótesis (no demostrada) es que olvida lo copiado, y la otra posible causa es un ruido de
   exploración demasiado alto (0,3 en unidades de acción son ~54° en el ángulo de la barra).
4. **Rivales variados**: heurísticos de distinto nivel y copias de sí mismo, para no explotar
   las manías de un único rival.

**Reglas**
- *Físicas del robot* (giro de ±90°, portero de −10° a +90°, topes, motor a 12 V, lazo
  cerrado del giro, saque del portero): las impone el entorno solas.
- *De buen juego* (levantar a −90°, ángulo positivo solo al golpear, buscar huecos): el
  entorno no las impone. Se heredan del heurístico (punto 2), se vigilan con el anclaje
  (punto 3) y, para lo más grave, con una penalización medible: que una figura propia pare la
  bola de su propio equipo.
- Recompensas de **defensa** por rol (parada: toque propio que desvía un tiro a puerta rival;
  despeje: sacar la bola de la zona de peligro). El shaping nuevo debe ser de potencial para
  no cambiar la política óptima.

**Qué NO resolvería**
- La diferencia alumno‑profesor por la cámara (en la ronda anterior, 0,2–0,4 goles/min por
  debajo del profesor). Tiene su propio arreglo: retardo y memoria del alumno, y la variante
  con sensores ideales de la cadena de esta noche dirá cuánto de esa diferencia es la cámara.
- Los límites físicos: con 0,45 m/s de movimiento lateral y 0,28 N·m parado, ningún
  aprendizaje defiende como una persona rápida.

**Riesgos**: puede aprender a ganar a nuestro rival heurístico y no a una persona; puede
empeorar como PPO sobre el alumno; y cada comparación necesita muchos partidos (el margen de
error de 240 partidos es de ±0,16 goles/min).

**Prueba barata antes de comprometerse** (unas 2–3 h, una vez terminada la cadena actual):
imitar al heurístico con el estado real, afinar con PPO anclado durante ~500 k pasos y medirlo
con 240 partidos de 30 s contra el profesor actual (rivales de nivel 0,4, 0,7 y 1,0). Si no lo
supera con el margen de error claramente por encima de 0, se abandona.

**Efecto secundario útil**: entrenarlo con distintas especificaciones (12 V frente a 24 V,
otra polea, figuras de otro peso) dice cuántos goles por minuto cuesta cada decisión de
hardware antes de comprar nada.

**Demostraciones de personas** (vídeos, o partidas en tu mesa con encoders y la Brio): no
como profesor único, sino como punto de partida y freno del aprendizaje. Las partidas en tu
mesa real dan estado y acciones exactos; los vídeos exigen reconstruir el estado y casi todos
son de otras mesas. Una persona mueve las barras a ~1,5 m/s y el motor a 0,45 m/s: sirve
copiar la colocación, no las velocidades. Herramienta intermedia posible: jugar contra el
simulador con el ratón, una barra cada vez y las demás en automático.

---

## 10. Referencias

**Nota de honestidad**: de los trabajos sobre futbolín solo he visto el resumen
que devuelve un buscador, no los he leído completos. Los métodos de aprendizaje
por refuerzo (apartados 10.1) son los que implementa el código; las citas son
las de los artículos originales.

### 10.1 Métodos de aprendizaje usados

| Qué | Dónde se usa | Referencia |
|---|---|---|
| **PPO** | `train.py` | Schulman et al., *Proximal Policy Optimization Algorithms*, 2017. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347) |
| **Stable Baselines3** (implementación de PPO) | `train.py`, `policies.py` | Raffin et al., *Stable-Baselines3: Reliable Reinforcement Learning Implementations*, JMLR 22, 2021. [Documentación de PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) |
| **Shaping basado en potencial** | `reward.py` | Ng, Harada y Russell, *Policy invariance under reward transformations: Theory and application to reward shaping*, ICML 1999 |
| **Actor‑crítico asimétrico** (el crítico ve el estado real, el actor solo lo observable) | `policies.py` | Pinto et al., *Asymmetric Actor Critic for Image-Based Robot Learning*, RSS 2018. [arXiv:1710.06542](https://arxiv.org/abs/1710.06542) |
| **Imitación con ruido en las acciones (DART)** | `imitation.py` | Laskey et al., *DART: Noise Injection for Robust Imitation Learning*, CoRL 2017. [arXiv:1703.09327](https://arxiv.org/abs/1703.09327) |
| **DAgger** (propuesto, no implementado) | línea pendiente | Ross, Gordon y Bagnell, *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*, AISTATS 2011. [arXiv:1011.0686](https://arxiv.org/abs/1011.0686) |
| **Aleatorización de dominio** | `physics.py`, `actuators.py`, `vision.py` | Tobin et al., *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World*, IROS 2017. [arXiv:1703.06907](https://arxiv.org/abs/1703.06907) |

### 10.2 Trabajos sobre futbolín

| Trabajo | Qué es (según su resumen) | Relevancia |
|---|---|---|
| [*Learning to Play Foosball: System and Baselines*](https://arxiv.org/pdf/2407.16606) | sistema y líneas base para aprender a jugar al futbolín | Es lo más cercano a este proyecto; conviene leerlo entero antes de la siguiente iteración |
| [Conjunto de datos de futbolín para aprendizaje por imitación](https://arxiv.org/html/2404.05357v1) (arXiv:2404.05357) | datos etiquetados para regresar con una CNN el desplazamiento y el giro de cada barra y la bola, con vistas a imitar a humanos | Posible fuente para estimar el estado desde vídeo; no he visto que publique los datos |
| [Informe de Joseph Lundy, Univ. de Toronto (curso 2529, 2024)](https://www.cs.toronto.edu/~lindell/teaching/2529/past_projects/2024/report/joseph-lundy.pdf) | seguimiento de la bola y las figuras con YOLOv7 + SORT y marcadores ArUco para el ángulo real | Referencia para la parte de visión del robot real |
| [Tesis de David Hagens, h‑da (2023)](https://fbmn.h-da.de/fileadmin/Dokumente/Studium/DS/WS2023_MDS_Thesis_David_Hagens_THE.pdf) | trabajo de máster sobre futbolín (no leído) | Pendiente de revisar |
| [*Foosball tracking*, Virginia Tech](https://computing.ece.vt.edu/~steflee/pdfs/foosball-tracking.pdf) | seguimiento desde vídeo cenital | Antecedente de visión |
| [kickercam](https://github.com/moritzschaefer/kickercam/blob/master/README.md) | cámara de futbolín con detección de gol y repetición lenta | Ejemplo práctico de cámara |

### 10.3 Medidas y componentes

| Dato | Fuente |
|---|---|
| Campo de 116 × 70 cm | [Ficha del Don Futbolín Madrid](https://www.donfutbolin.com/product/futbolin-modelo-don-futbolin) |
| Figura madrileña: 71 mm al pie, 48 mm a la cabeza, 119 mm en total, 30 mm de ancho | [Billares Balsalobre](https://www.billaresbalsalobre.com/tienda/futbolin/jugador-futbolin/billaresbalsalobre-jugador-futbolin-madrileno-madera-natural/), [Billares y Juegos](https://www.billaresyjuegos.com/jugadores-munecos-futbolines-jugador-muneco-futbolin-madera-pasador-rojiblanco-estandar-medidas-p-318.html) |
| Otras cotas de figuras (no madrileñas; solo contraste) | [Billares y Juegos, aluminio](https://www.billaresyjuegos.com/jugadores-munecos-futbolines-jugador-muneco-futbolin-aluminio-azulgrana-para-barra-14mm-delgado-medidas-aaltura-cabeza-centro-barra54mm-baltura-pies-centro-barra65mm-haltura-total119mm-incluye-p-1151.html) |
| Driver TMC2209 | hoja de datos de Trinamic en `.codex-work-backplane/tmc2209.pdf` |
| Cámara Logitech Brio | sin fuente verificada: resolución y fps son los de su ficha comercial (720p a 90 fps), la latencia es una estimación |

La mecánica de la bola (deslizamiento hasta rodar, pérdida de 2/7 de la
velocidad tras un golpe seco, factor 5/7 de la rodadura) es mecánica clásica de
la esfera maciza, sin referencia específica. El modelo del motor paso a paso
(muelle sinusoidal con amortiguamiento y par que cae con la velocidad) es el
modelo estándar de motor híbrido; sus parámetros son estimaciones.
