"""
world.py
--------
Estado completo del sistema — futbolín tipo Madrid.
Medidas reales: campo 120 x 68 cm, bola 35 mm de diámetro.
8 barras totales (4 por equipo), con posición lineal Y y ángulo de rotación.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List

# ---------------------------------------------------------------------------
# Constantes físicas del futbolín tipo Madrid
# ---------------------------------------------------------------------------

FIELD_LENGTH  = 1.20    # m  (eje X, largo)
FIELD_WIDTH   = 0.68    # m  (eje Y, ancho)
BALL_RADIUS   = 0.0175  # m  (diámetro 35 mm)
GOAL_WIDTH    = 0.20    # m  (abertura de portería, centrada en Y=0)

# Recorrido lineal máximo de cada barra (±7 cm)
LINEAR_RANGE  = 0.07    # m

# Límites de actuador (usados también en physics y reward)
LINEAR_VMAX       = 0.8         # m/s
LINEAR_AMAX       = 6.0         # m/s²
ANGULAR_OMEGA_MAX = 6.0 * np.pi # rad/s  (~3 giros/s, golpeo realista)
ANGULAR_ALPHA_MAX = 60.0        # rad/s²  (aceleración alta para golpes rápidos)

# Geometría del jugador (simplificado como prisma)
FOOT_LENGTH       = 0.075   # m  (radio del arco de giro: eje barra → pie)
PLAYER_HALF_WIDTH = 0.025   # m  (semiancho del pie en Y)
PLAYER_DEPTH      = 0.008   # m  (profundidad del pie en X)
KICK_ANGLE_MAX    = np.pi / 3   # rad (±60°): el pie puede golpear dentro de este rango

# Velocidad máxima estimada de la bola (para normalización)
MAX_BALL_SPEED = 3.5    # m/s

# ---------------------------------------------------------------------------
# Configuración de las 8 barras — Futbolín tipo Madrid
#
# Equipo A defiende portería izquierda (x_min), ataca hacia la derecha.
# Equipo B defiende portería derecha  (x_max), ataca hacia la izquierda.
#
# Layout real (de izquierda a derecha):
#   GK-A | DEF-A | ATK-B | MID-A | MID-B | ATK-A | DEF-B | GK-B
#
# Equipo A: GK(-0.50)  DEF(-0.36)  MID(-0.05)  ATK(+0.20)
# Equipo B: ATK(-0.20) MID(+0.05)  DEF(+0.36)  GK(+0.50)
# ---------------------------------------------------------------------------
#  idx  x_pos   n_players  equipo  rol
BAR_DEFINITIONS = [
    (-0.50, 1, 'A', 'GK'),    # 0  Portero A       — guarda portería izquierda
    (-0.36, 2, 'A', 'DEF'),   # 1  Defensa A        — protege su propia portería
    (-0.20, 3, 'B', 'ATK'),   # 2  Delanteros B     — B ataca hacia la izquierda
    (-0.05, 5, 'A', 'MID'),   # 3  Medios A
    ( 0.05, 5, 'B', 'MID'),   # 4  Medios B
    ( 0.20, 3, 'A', 'ATK'),   # 5  Delanteros A     — A ataca hacia la derecha
    ( 0.36, 2, 'B', 'DEF'),   # 6  Defensa B        — protege su propia portería
    ( 0.50, 1, 'B', 'GK'),    # 7  Portero B        — guarda portería derecha
]

# Separación Y entre jugadores de la misma barra
_PLAYER_SPACING = {1: 0.0, 2: 0.14, 3: 0.14, 5: 0.12}


def _compute_offsets(n: int) -> List[float]:
    """Distribuye N jugadores simétricamente en Y."""
    s = _PLAYER_SPACING.get(n, 0.13)
    half = (n - 1) / 2.0 * s
    return [round(-half + i * s, 4) for i in range(n)]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BallState:
    """Estado de la bola en 2D."""
    x:  float = 0.0
    y:  float = 0.0
    vx: float = 0.5
    vy: float = 0.2

    def reset(self, x=0.0, y=0.0, vx=0.5, vy=0.2):
        self.x, self.y, self.vx, self.vy = x, y, vx, vy

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.vx, self.vy], dtype=np.float32)


@dataclass
class BarState:
    """
    Estado de una barra del futbolín.

    Dos grados de libertad:
      - linear_pos [m]: desplazamiento lateral (eje Y) dentro de ±LINEAR_RANGE
      - angle [rad]:    ángulo de rotación de la barra, 0 = pie hacia abajo
    """
    bar_x:          float       = 0.0
    n_players:      int         = 1
    team:           str         = 'A'
    role:           str         = 'GK'
    player_offsets: List[float] = field(default_factory=list)

    # Estado dinámico
    linear_pos:  float = 0.0   # [-LINEAR_RANGE, +LINEAR_RANGE]
    angle:       float = 0.0   # [-π, +π]
    linear_vel:  float = 0.0   # m/s
    angular_vel: float = 0.0   # rad/s

    def get_player_abs_y(self) -> List[float]:
        """Posición Y absoluta de cada jugador en el campo."""
        return [self.linear_pos + off for off in self.player_offsets]

    def foot_tip_vel_x(self) -> float:
        """
        Velocidad tangencial de la punta del pie en X cuando angle ≈ 0.
        Se usa para calcular el impulso en golpeos.
        """
        # Velocidad del pie: ω × R, proyectada en X (componente de golpeo)
        return self.angular_vel * FOOT_LENGTH * np.cos(self.angle)

    def is_in_kick_zone(self) -> bool:
        """True si el pie está en posición de contacto con la bola."""
        return abs(self.angle) < KICK_ANGLE_MAX

    def as_array(self) -> np.ndarray:
        """Vector normalizado de estado de la barra."""
        return np.array([
            self.linear_pos  / LINEAR_RANGE,
            self.angle       / np.pi,
            self.linear_vel  / LINEAR_VMAX,
            self.angular_vel / ANGULAR_OMEGA_MAX,
        ], dtype=np.float32)

    def reset(self):
        self.linear_pos  = 0.0
        self.angle       = 0.0
        self.linear_vel  = 0.0
        self.angular_vel = 0.0


@dataclass
class FieldConfig:
    """Geometría del campo."""
    length:     float = FIELD_LENGTH
    width:      float = FIELD_WIDTH
    goal_width: float = GOAL_WIDTH

    @property
    def x_min(self)      -> float: return -self.length / 2.0
    @property
    def x_max(self)      -> float: return  self.length / 2.0
    @property
    def y_min(self)      -> float: return -self.width  / 2.0
    @property
    def y_max(self)      -> float: return  self.width  / 2.0
    @property
    def goal_y_min(self) -> float: return -self.goal_width / 2.0
    @property
    def goal_y_max(self) -> float: return  self.goal_width / 2.0


@dataclass
class WorldState:
    """Estado completo del mundo."""
    ball:  BallState      = field(default_factory=BallState)
    bars:  List[BarState] = field(default_factory=list)
    cfg:   FieldConfig    = field(default_factory=FieldConfig)
    time:  float          = 0.0
    score: List[int]      = field(default_factory=lambda: [0, 0])  # [A, B]

    @property
    def field(self) -> FieldConfig:
        """Alias para compatibilidad con el resto de modulos."""
        return self.cfg

    def reset(self, ball_vx: float = 0.5, ball_vy: float = 0.2):
        self.ball.reset(x=0.0, y=0.0, vx=ball_vx, vy=ball_vy)
        for bar in self.bars:
            bar.reset()
        self.time = 0.0

    def get_observation(self) -> np.ndarray:
        """
        Vector de observación normalizado.
        [ball_x, ball_y, ball_vx, ball_vy,  bar0×4, bar1×4, ..., bar7×4]
        Total: 4 + 8×4 = 36 dimensiones, todos ≈ [-1, 1].
        """
        f = self.field
        obs = [
            self.ball.x  / (f.length / 2.0),
            self.ball.y  / (f.width  / 2.0),
            self.ball.vx / MAX_BALL_SPEED,
            self.ball.vy / MAX_BALL_SPEED,
        ]
        for bar in self.bars:
            obs.extend(bar.as_array().tolist())
        return np.array(obs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Fábrica de mundo estándar
# ---------------------------------------------------------------------------

def create_standard_world() -> WorldState:
    """Crea un mundo con las 8 barras estándar del futbolín tipo Madrid."""
    bars = []
    for bar_x, n_players, team, role in BAR_DEFINITIONS:
        offsets = _compute_offsets(n_players)
        bars.append(BarState(
            bar_x=bar_x,
            n_players=n_players,
            team=team,
            role=role,
            player_offsets=offsets,
        ))
    return WorldState(bars=bars)


def get_team_bar_indices(team: str) -> List[int]:
    """Devuelve los índices de las barras de un equipo ('A' o 'B')."""
    return [i for i, (_, _, t, _) in enumerate(BAR_DEFINITIONS) if t == team]
