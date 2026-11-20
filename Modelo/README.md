# Futbolín Autónomo — Simulador RL

Simulador de futbolín tipo **Madrid** (estilo español tradicional) diseñado para entrenar agentes de **Reinforcement Learning** y transferirlos a hardware real.

El enfoque es **control e inteligencia**, no gráficos. Solo Python + NumPy.

![Simulador en acción](docs/screenshot.png)

---

## Motivación

El futbolín es un entorno de control con características interesantes para RL:
- Dinámica rápida y no lineal (bola, rebotes, rotación de barras)
- Acción continua en dos ejes (posición lineal + ángulo de rotación)
- Recompensas escasas (gol) complementadas con señales densas
- Sim-to-real: el simulador está diseñado para sustituirse por hardware real sin cambiar el agente

---

## Estructura del proyecto

```
futbolin-autonomo/
├── world.py       — Estado del sistema (bola, barras, campo)
├── physics.py     — Motor de física 2D
├── actuators.py   — Modelo de actuadores (motores reales)
├── reward.py      — Función de recompensa
├── env.py         — Entorno tipo Gymnasium
├── train.py       — Entrenamiento PPO con currículo
├── test_training.py — Pruebas de simulación y entrenamiento
├── example.py     — Agente heurístico de referencia
├── visualize.py   — Visualización del agente y repeticiones
├── models/        — Modelos entrenados (.zip)
└── logs/          — Logs TensorBoard
```

---

## El campo

Campo interior de referencia: **Don Futbolín Madrid**, de
[116 × 70 cm según su ficha](https://www.donfutbolin.com/product/futbolin-modelo-don-futbolin).

| Parámetro | Valor |
|---|---|
| Largo | 116 cm |
| Ancho | 70 cm |
| Bola (diámetro) | 35 mm (estimado) |
| Ancho de portería | 20 cm (estimado) |
| Barras totales | 8 (4 por equipo) |

### Distribución de barras

```
  [Portería A]                                          [Portería B]
  x=-0.58                                               x=+0.58

  GK-A    DEF-A   ATK-B   MID-A   MID-B   ATK-A   DEF-B   GK-B
  (1j)    (2j)    (3j)    (5j)    (5j)    (3j)    (2j)    (1j)
  -0.50  -0.36   -0.20   -0.05   +0.05   +0.20   +0.36   +0.50
```

Cada barra tiene **dos grados de libertad**:
- **Lineal**: desplazamiento lateral ±7 cm (eje Y)
- **Angular**: rotación completa ±180° (golpeo)

Las posiciones de las barras, separaciones de figuras, recorridos y dimensiones
de los jugadores siguen siendo aproximaciones pendientes de medir en la mesa real.
La física todavía simplifica el giro y la altura de los pies; cambiar el tamaño
del campo no completa esa calibración. Los modelos guardados se pueden cargar,
pero necesitan nueva evaluación y entrenamiento con esta geometría.

---

## Arquitectura de módulos

### `world.py` — Estado del sistema

Define todos los estados del mundo como dataclasses:

```python
BallState      # x, y, vx, vy
BarState       # linear_pos, angle, linear_vel, angular_vel
FieldConfig    # dimensiones del campo
WorldState     # bola + 8 barras + campo + marcador
```

El vector de observación tiene **36 dimensiones** normalizadas ≈ [-1, 1]:

```
[ball_x, ball_y, ball_vx, ball_vy,  bar0×4, bar1×4, ..., bar7×4]
```

### `physics.py` — Motor de física

| Fenómeno | Modelo |
|---|---|
| Movimiento bola | Integración de Euler, dt=0.02 s |
| Fricción rodante | Decaimiento por paso 1 − 0.05 × dt |
| Rebote en pared | Laterales y fondos fuera de portería; restitución 0.65 + ruido angular |
| Impacto jugador | Impulso = ω × L × cos(θ) + transferencia vel. lineal |
| Zona de contacto | \|ángulo\| < 60° (pie en posición de golpeo) |
| Detección de gol | La bola cabe entre los postes en el instante de cruzar el fondo |
| Zona sin jugador | La bola pasa libremente (la varilla no bloquea) |

El ruido en rebotes e impactos es configurable y reduce el **sim-to-real gap**.

### `actuators.py` — Modelo de motores

Simula la dinámica real de servomotores con **perfil trapezoidal de velocidad**:

| Parámetro | Lineal | Angular |
|---|---|---|
| Recorrido | ±7 cm | ±180° |
| Vel. máxima | 0.8 m/s | ~3 giros/s |
| Aceleración máx. | 6 m/s² | 60 rad/s² |
| Ruido posición | ±2 mm | ±1.1° |
| Slippage | 0.5% prob./step | 0.5% prob./step |

El movimiento **no es instantáneo**: el agente debe aprender a anticipar la inercia.

### `reward.py` — Función de recompensa

La configuración final y el entrenamiento directo premian los goles. Las ayudas
para aprender a tocar la bola se reducen durante el currículo:

| Componente | Fase 0 | Fase 1 | Fase 2 / directo |
|---|---|---|---|
| Gol marcado | +5 | +10 | +10 |
| Gol recibido | -1 | -5 | -10 |
| Toque propio | +0.2 | +0.05 | 0 |
| Bono por golpe hacia delante | ≤ +0.3 | ≤ +0.1 | 0 |
| Dirección de la bola, escala por paso | 0.03 | 0.01 | 0 |
| Proximidad, máximo por paso | +0.03 | +0.003 | 0 |
| Bola en zona de ataque, por paso | +0.005 | 0 | 0 |
| Posición en el campo, por paso | 0 | 0 | 0 |
| Penalización por acción, escala | 0 | 0.002 | 0.005 |
| Penalización por cambio de acción, escala | 0.002 | 0.002 | 0.002 |

La proximidad se calcula sobre **todas las barras del equipo** (GK, DEF, MID, ATK),
durante las fases iniciales. En la fase final, mantener la bola cerca o acumular
toques no genera recompensa. Se conservan penalizaciones pequeñas por acción y
cambios bruscos; su coste total máximo en 750 pasos es menor que el premio por gol.

### `env.py` — Entorno Gymnasium

```python
env = FoosballEnv(
    dt=0.02,              # 50 Hz
    max_steps=750,        # 15 s por episodio
    controlled_team='A',  # equipo controlado por el agente
    opponent='heuristic', # oponente: None | 'heuristic' | objeto propio
    noise=True,           # incertidumbre física activada
)

obs, info   = env.reset()
obs, r, terminated, truncated, info = env.step(action)
```

**Espacio de acción (8 dimensiones, [-1, 1]):**
```
[lin_GK, ang_GK,  lin_DEF, ang_DEF,  lin_MID, ang_MID,  lin_ATK, ang_ATK]
```
Cada par controla la posición lineal y el ángulo de una barra del equipo A.

### Para conectar hardware real

Sobreescribir estos dos métodos en una subclase de `FoosballEnv`:

```python
class HardwareFoosballEnv(FoosballEnv):

    def _apply_agent_commands(self, action):
        # Escribir aquí el comando al driver del motor (serial, CAN, etc.)
        for k, bar_idx in enumerate(self._team_bar_idxs):
            linear_cmd  = float(action[2 * k])
            angular_cmd = float(action[2 * k + 1])
            my_motor_driver.send(bar_idx, linear_cmd, angular_cmd)

    def _get_obs(self):
        # Leer aquí los sensores reales (encoders, cámara, etc.)
        return my_sensor_driver.read_normalized()
```

---

## Entrenamiento RL

### Algoritmo

**PPO** (Proximal Policy Optimization) con red neuronal MLP [256, 256].

El actor (acciones) y el crítico (valor del estado) conservan sus dos capas de
256 neuronas. La exploración usa **gSDE**, con ruido dependiente del estado que
se renueva cada 4 pasos, y una salida `tanh` que respeta los límites de las barras.
La tasa de aprendizaje baja de `3e-4` hacia `3e-5` a lo largo del entrenamiento,
sin reiniciarse al cambiar de fase; `target_kl=0.02`
detiene las épocas de una actualización cuando la política cambia demasiado.
Estos ajustes siguen las opciones de [PPO en Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html).

Cada entorno tiene su propio generador aleatorio: `--seed` controla también el
ruido de física, actuadores y relanzamientos. Las métricas cuentan los toques
del equipo A por separado para cada entorno y se guardan en CSV y TensorBoard.
Los modelos anteriores siguen siendo compatibles con evaluación y visualización.
Las recompensas y resultados históricos corresponden a la física y configuración
anteriores y no se comparan directamente con los nuevos.

`best/best_model.zip` se elige por estos criterios, en orden: puntos por partida
(victoria = 1, empate = 0.5, derrota = 0), diferencia media de goles, goles marcados
y recompensa media. Así, más recompensa auxiliar no desplaza un modelo con mejores
resultados. Cada checkpoint se valida con las mismas semillas: un entorno por
episodio, 15 episodios en la fase final y 10 en las anteriores. Conviene reservar
otras semillas para la comparación final entre entrenamientos.

TensorBoard registra `eval/win_rate`, `eval/draw_rate`, `eval/loss_rate`,
`eval/points`, `eval/goals_scored`, `eval/goals_conceded` y `eval/goal_difference`.
Junto a `evaluations.npz` se guarda `evaluations_goals.npz`, con los goles A/B de
cada episodio, los pasos de entrenamiento y la semilla base de evaluación.

### Currículo de 3 fases

El entrenamiento progresa en dificultad para acelerar el aprendizaje:

| Fase | Configuración | Objetivo |
|---|---|---|
| 0 | Sin oponente, sin ruido, 300 pasos | Aprender a tocar la bola |
| 1 | Oponente heurístico, con ruido, 500 pasos | Adaptarse al juego real |
| 2 | Configuración completa, 750 pasos, ayudas desactivadas | Marcar y evitar goles |

### Uso

```bash
# Instalar dependencias
pip install stable-baselines3[extra] gymnasium torch

# Entrenar con currículo (recomendado)
python train.py --timesteps 300000 --curriculum

# Entrenar la nueva configuración sin sobrescribir los modelos existentes
python train.py --timesteps 1000000 --curriculum --seed 42 --save-dir models/ppo_v3 --log-dir logs/ppo_v3

# Entrenar directo sin currículo
python train.py --timesteps 500000

# Más entornos paralelos (más rápido si tienes CPU)
python train.py --timesteps 500000 --n-envs 8 --curriculum

# Evaluar un modelo guardado
python train.py --eval-only models/foosball_ppo_final

# Comparar ambos modelos en los mismos episodios (semillas 10042 a 10061)
python train.py --eval-only models/foosball_ppo_final --seed 42
python train.py --eval-only models/ppo_v3/best/best_model --seed 42

# Ver curvas de aprendizaje
tensorboard --logdir logs

# Pruebas de simulación, métricas y entrenamiento corto (directo y curricular)
python -m unittest -v test_training
```

PPO completa bloques de `n_steps × n_envs`; el número real de pasos puede superar
el solicitado, especialmente en ejecuciones cortas. La evaluación muestra recompensa,
goles, toques propios, victorias, empates, derrotas y puntos por partida. Los cambios de configuración necesitan
un entrenamiento nuevo; las pruebas cortas verifican el funcionamiento, pero la mejora
de juego debe comprobarse con varias semillas y el mismo presupuesto de entrenamiento.

### Resultados históricos (configuración anterior; 1M pasos, ~24 min en GitHub Actions)

Entrenamiento en la nube mediante el workflow `train.yml` (Ubuntu, CPU).
Cada run guarda el modelo automáticamente en el repositorio.

| Métrica | Valor |
|---|---|
| Timesteps | 1.000.000 |
| Currículo | 3 fases (sin oponente → heurístico → completo) |
| Duración (GitHub Actions) | ~24 min |
| Modelo guardado en | `models/foosball_ppo_final.zip` |

---

## Uso rápido

```python
from env import FoosballEnv
import numpy as np

env = FoosballEnv(dt=0.02, max_steps=750)
obs, _ = env.reset()

for _ in range(750):
    action = env.action_space.sample()         # acción aleatoria
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

print(f"Score: A={info['score']['A']}  B={info['score']['B']}")
```

```python
# Cargar y usar el agente entrenado
from stable_baselines3 import PPO
from env import FoosballEnv

model = PPO.load("models/foosball_ppo_final")
env   = FoosballEnv()
obs, _ = env.reset()

done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, r, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

---

## Dependencias

```
numpy>=2.0
stable-baselines3[extra]>=2.0
gymnasium>=0.29
torch>=2.0
matplotlib>=3.7
```

```bash
pip install -r requirements.txt
```

> **Nota:** Se requiere `numpy>=2.0` porque los modelos entrenados se serializan con esa versión. Usar numpy 1.x provocará error al cargar el modelo.

---

## Próximos pasos sugeridos

- **Self-play**: que el equipo B también sea un agente RL que mejora junto al A
- **Observación con cámara**: sustituir la obs. perfecta por detección de bola con visión
- **Múltiples agentes**: un agente por barra en lugar de uno para todo el equipo
- **Hardware real**: implementar `HardwareFoosballEnv` con los drivers del sistema

---

## Licencia

Proyecto de investigación y desarrollo. Uso libre para fines educativos y de investigación.
