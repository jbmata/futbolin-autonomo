"""
world.py
--------
Estado completo del sistema — futbolín tipo Madrid.
Geometría de la mesa según FUTBOLIN_MESA (mesa: Medidas/mesa.json; sim: estimación inicial).
8 barras totales (4 por equipo), con posición lineal Y y ángulo de rotación.

Convenciones geométricas
~~~~~~~~~~~~~~~~~~~~~~~~
- Plano del campo: X a lo largo (A defiende x_min, B defiende x_max), Y a lo ancho.
- Z vertical. La varilla está a ROD_HEIGHT sobre el campo.
- Ángulo de barra θ: 0 = jugador vertical, pie abajo. La punta del pie está en
      x = bar_x + FOOT_LENGTH · sin θ,   z = ROD_HEIGHT − FOOT_LENGTH · cos θ
  Con ω > 0 el pie (abajo) avanza hacia +x. El equipo A chuta con ω > 0 y el B
  con ω < 0.

Observación por equipo
~~~~~~~~~~~~~~~~~~~~~~
`WorldState.observation(team)` devuelve la observación vista por ese equipo: para
el B se gira el campo 180° (x→−x, y→−y, θ→−θ), así que un mismo agente siempre
"ataca hacia +x". Esto permite jugar con cualquiera de los dos equipos y hacer
self-play con la misma red.
"""

import math
import os
import numpy as np
from dataclasses import dataclass, field
from typing import List

# ---------------------------------------------------------------------------
# Constantes físicas del futbolín tipo Madrid
# ---------------------------------------------------------------------------

# Geometría de la mesa. FUTBOLIN_MESA elige el juego de medidas:
#   mesa (por defecto): Medidas/mesa.json, la misma fuente que usa el 3D de
#        Mecanica/3d/futbolin_actualizado. Cada dato lleva su fuente (real, plano, cad, ficha,
#        estimado); ver Medidas/README.md. FUTBOLIN_MEDIDAS=ruta usa otro fichero.
#   sim: estimaciones iniciales (campo de 116 x 70 cm, recorrido ±70 mm, jugadores centrados).
#        Los modelos entrenados antes del 23/11/2026 se entrenaron con esta mesa.
# Cambia el comportamiento de la mesa: usar el mismo valor al entrenar, evaluar y ejecutar.
#
# Referencia de las barras (mesa): cada barra se mide desde el final de carrera del lado de SUS
# motores (el lado del puño; equipo A en −y, B en +y). s = 0 es la barra contra ese tope y
# s = carrera, contra el contrario. Los jugadores van en los agujeros del plano, medidos desde la
# punta del mango; con la barra en s = 0 cada uno queda a (agujero − punta_mango_a_banda_en_s0)
# de la banda de ese lado. La mesa es simétrica por rotación de 180°: el rival usa las mismas
# medidas desde su lado. Internamente linear_pos = 0 es la mitad de la carrera de cada barra, con
# +y del mundo (así se normaliza la observación); ver linear_pos_from_s().
_DEFAULT_FIGURE = dict(eje_a_suela=71.0, eje_a_cabeza=48.0, ancho_pie=30.0, largo_pie=30.0,
                       alto_taco=20.0, holgura_suela_campo=3.0)
_ROLES = ("GK", "DEF", "MID", "ATK")
# equipo y rol de cada barra, en el orden de BAR_DEFINITIONS
_BAR_ROLES = (('A', 'GK'), ('A', 'DEF'), ('B', 'ATK'), ('A', 'MID'),
              ('B', 'MID'), ('A', 'ATK'), ('B', 'DEF'), ('B', 'GK'))
_MEDIDAS_PATH = os.environ.get("FUTBOLIN_MEDIDAS") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "Medidas", "mesa.json")


def motor_side(team: str) -> float:
    """Signo de y del lado de los motores (y de los puños) de un equipo: A en −y, B en +y."""
    return -1.0 if team == 'A' else 1.0


def load_measurements(path: str = None) -> dict:
    """Lee Medidas/mesa.json y devuelve la mesa en mm: campo, figura, bola y, por rol, x del eje,
    carrera y distancia de cada jugador a la banda de los motores con la barra en s = 0."""
    import json
    with open(path or _MEDIDAS_PATH, encoding="utf-8") as fh:
        raw = json.load(fh)
    v = lambda d: d["valor"]
    guides, width = v(raw["campo"]["distancia_entre_guias"]), v(raw["campo"]["ancho_plancha"])
    guide_to_band = (guides - width) / 2.0           # cara de la guía → borde de la plancha
    # En cada lado de cada barra, de dentro a la pared: jugador del extremo (o pasador del portero),
    # taco de goma, muelle (sin comprimir: el tope es cuando toca) y guía. Lo que ocupan quita carrera
    # por los dos lados.
    t = raw["topes"]
    cushion = v(t["largo_taco_goma"]) + v(t["largo_muelle"])
    roles = {}
    for role in _ROLES:
        b = raw["barras"][role]
        holes = v(b["agujeros_desde_punta_mango"])
        players = sorted(holes[k] for k in v(b["jugadores_en_agujeros"]))
        if v(b["tope"]) == "pasadores":
            # Pasadores en dos agujeros: en s = 0 el del lado del mango, con su goma y su muelle,
            # toca la guía de los motores.
            p0, p1 = sorted(holes[k] for k in v(b["pasadores_en_agujeros"]))
            stack = v(t["radio_pasador"]) + cushion      # centro del pasador → guía
            first = players[0] - p0 + stack              # primer jugador → guía de los motores
            travel = guides - (p1 - p0) - 2.0 * stack
        elif v(b["tope"]) == "jugador":
            # El jugador de cada extremo, con su goma y su muelle, toca la guía de su lado.
            stack = v(raw["figura"]["ancho_en_barra"]) / 2.0 + cushion   # centro del jugador → guía
            first = stack
            travel = guides - (players[-1] - players[0]) - 2.0 * stack
        else:
            raise ValueError(f"{role}: tope {v(b['tope'])!r} desconocido (pasadores o jugador)")
        if travel <= 0:
            raise ValueError(f"{role}: la carrera sale {travel:.1f} mm; revisa las medidas")
        home = [first - guide_to_band + (p - players[0]) for p in players]
        roles[role] = dict(x=v(b["x_eje_desde_centro"]), travel=travel, home=home)
    return dict(length=v(raw["campo"]["largo_plancha"]), width=v(raw["campo"]["ancho_plancha"]),
                goal=v(raw["campo"]["ancho_porteria"]), roles=roles,
                figure={k: v(d) for k, d in raw["figura"].items()},
                ball_diameter=v(raw["bola"]["diametro"]), ball_mass=v(raw["bola"]["masa"]))


def _table_from_measurements(m: dict) -> dict:
    half_w = m["width"] / 2.0
    offsets, ranges = [], []
    for team, role in _BAR_ROLES:
        r = m["roles"][role]
        # y de cada jugador (desde el centro del campo) con la barra a mitad de carrera
        offsets.append(sorted(motor_side(team) * (half_w - d - r["travel"] / 2.0) / 1000.0
                              for d in r["home"]))
        ranges.append(r["travel"] / 2000.0)
    return dict(length=m["length"] / 1000.0, width=m["width"] / 1000.0, goal=m["goal"] / 1000.0,
                bar_x=tuple(m["roles"][k]["x"] / 1000.0 for k in _ROLES),
                offsets=offsets, ranges=ranges, figure=m["figure"],
                ball_radius=m["ball_diameter"] / 2000.0, ball_mass=m["ball_mass"] / 1000.0)


MESA = os.environ.get("FUTBOLIN_MESA", "mesa")
if MESA == "mesa":
    _M = _table_from_measurements(load_measurements())
elif MESA == "sim":
    _M = dict(length=1.16, width=0.70, goal=0.20, bar_x=(0.50, 0.36, 0.05, 0.20),
              offsets=None, ranges=[0.07] * 8, spacing={2: 0.14, 3: 0.14, 5: 0.12},
              figure=_DEFAULT_FIGURE, ball_radius=0.0175, ball_mass=0.024)
else:
    raise ValueError(f"FUTBOLIN_MESA={MESA!r}: usa 'mesa' o 'sim'")

FIELD_LENGTH  = _M["length"]   # m  (eje X, largo)
FIELD_WIDTH   = _M["width"]    # m  (eje Y, ancho)
BALL_RADIUS   = _M["ball_radius"]   # m
GOAL_WIDTH    = _M["goal"]     # m  (abertura de portería, centrada en Y=0)

# Recorrido lineal: cada barra va de −linear_range a +linear_range alrededor de la mitad de su
# carrera (BarState.linear_range). LINEAR_RANGE es el mayor, para lo que necesita un solo valor.
BAR_LINEAR_RANGES = tuple(_M["ranges"])   # m, en el orden de BAR_DEFINITIONS
LINEAR_RANGE  = max(BAR_LINEAR_RANGES)     # m


def linear_pos_from_s(bar, s: float) -> float:
    """linear_pos interno [m] de una barra a partir de s [m] desde el final de carrera de sus motores."""
    return -motor_side(bar.team) * (s - bar.linear_range)


def s_from_linear_pos(bar, linear_pos: float) -> float:
    """s [m] desde el final de carrera del lado de los motores (0 .. carrera)."""
    return bar.linear_range - motor_side(bar.team) * linear_pos


# Giro de las barras: 0 = jugador vertical (pie abajo). Los motores van de -90° a +90°.
ANGULAR_RANGE = math.pi / 2     # rad, orden ±1 → ±90°

# Rango de giro de cada rol, en el marco del equipo (positivo = pie hacia delante). El portero
# gira de -10° a +90°: casi no puede echar el pie atrás. Es un tope mecánico de la mesa.
ANGLE_LIMITS_DEG = {"GK": (-10.0, 90.0)}


def angle_limits(role: str, team: str = 'A') -> tuple:
    """(mínimo, máximo) en radianes en el marco del MUNDO; el equipo B está girado 180°."""
    lo, hi = ANGLE_LIMITS_DEG.get(role, (-90.0, 90.0))
    lo, hi = math.radians(lo), math.radians(hi)
    return (lo, hi) if team == 'A' else (-hi, -lo)


def team_angle_limits(role: str) -> tuple:
    """(mínimo, máximo) en radianes en el marco del EQUIPO."""
    return angle_limits(role, 'A')


# Límites de actuador (usados también en physics y reward)
LINEAR_VMAX       = 0.8         # m/s
LINEAR_AMAX       = 6.0         # m/s²
ANGULAR_OMEGA_MAX = 6.0 * np.pi # rad/s  (~3 giros/s, golpeo realista)
ANGULAR_ALPHA_MAX = 60.0        # rad/s²  (aceleración alta para golpes rápidos)

# Geometría del jugador Madrid: pierna + pie de TACO (bloque macizo de suela plana).
# En el marco local de la figura, u va del eje hacia los pies y v hacia delante
# (+x con θ = 0). La figura es la unión de dos rectángulos (extruidos en Y):
#   cabeza/torso: u ∈ [−HEAD_LENGTH, 0], v ∈ [−BODY_HALF_DEPTH, +BODY_HALF_DEPTH]
#   pierna: u ∈ [0, L − FOOT_HEIGHT],  v ∈ [−PLAYER_DEPTH, +PLAYER_DEPTH]
#   taco:   u ∈ [L − FOOT_HEIGHT, L],  v ∈ [−FOOT_HALF_LENGTH, +FOOT_HALF_LENGTH]
# Cotas de fichas de jugadores madrileños: 71 mm del centro de la barra a los pies,
# 48 mm a la cabeza, 119 mm en total y 30 mm de anchura máxima (billaresbalsalobre.com,
# billaresyjuegos.com).
# shortcut: el largo y el alto del taco no figuran en las fichas; medir una figura real.
_F = _M["figure"]
FOOT_LENGTH       = _F["eje_a_suela"] / 1000.0          # m  (eje de la barra → suela del taco)
HEAD_LENGTH       = _F["eje_a_cabeza"] / 1000.0         # m  (eje de la barra → cabeza)
FOOT_HEIGHT       = _F["alto_taco"] / 1000.0            # m  (alto del taco)
FOOT_HALF_LENGTH  = _F["largo_pie"] / 2000.0            # m  (semilargo del taco en X: puntera / talón)
PLAYER_HALF_WIDTH = _F["ancho_pie"] / 2000.0            # m  (semiancho del taco en Y)
PLAYER_DEPTH      = 0.008   # m  (semigrosor de la pierna en X)
BODY_HALF_DEPTH   = 0.010   # m  (semigrosor del torso y la cabeza en X)
FOOT_ROUNDING     = 0.002   # m  (radio de redondeo de las aristas)
FOOT_CLEARANCE    = _F["holgura_suela_campo"] / 1000.0   # m  (holgura entre la suela y el campo)
ROD_HEIGHT        = FOOT_LENGTH + FOOT_CLEARANCE   # m  (altura del eje de la barra)


def _foot_lowest_z(theta: float) -> float:
    """Altura del punto más bajo del taco con la barra a ángulo θ."""
    s, c = math.sin(theta), math.cos(theta)
    return min(ROD_HEIGHT - u * c + v * s
               for u in (FOOT_LENGTH - FOOT_HEIGHT, FOOT_LENGTH)
               for v in (-FOOT_HALF_LENGTH, FOOT_HALF_LENGTH))


def _compute_kick_angle_max() -> float:
    """Mayor |θ| con el que el taco aún alcanza la bola (por encima pasa por debajo)."""
    top = 2.0 * BALL_RADIUS + FOOT_ROUNDING
    lo, hi = 0.0, math.pi / 2
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if _foot_lowest_z(mid) < top else (lo, mid)
    return lo


KICK_ANGLE_MAX = float(_compute_kick_angle_max())   # ≈ 70° con el taco

# Velocidad de referencia de la bola (normalización de las observaciones v1/v2)
MAX_BALL_SPEED = 3.5    # m/s

# Masas e inercias
# shortcut: bola de 35 mm de unos 24 g y figuras de ~40 g; pesar las piezas reales.
GRAVITY         = 9.81
BALL_MASS       = _M["ball_mass"]   # kg
FIGURE_MASS     = 0.040    # kg por figura
FIGURE_GYRATION = 0.045    # m, radio de giro de una figura respecto al eje de la barra
ROD_MASS        = 0.70     # kg, varilla hueca de acero de ~16 mm
ROD_INERTIA     = 3.7e-5   # kg·m², varilla respecto a su eje


def bar_inertia(n_players: int) -> float:
    """Momento de inercia de una barra (varilla + figuras) respecto a su eje [kg·m²]."""
    return ROD_INERTIA + n_players * FIGURE_MASS * FIGURE_GYRATION ** 2


def bar_mass(n_players: int) -> float:
    """Masa de una barra con sus figuras [kg]."""
    return ROD_MASS + n_players * FIGURE_MASS

# ---------------------------------------------------------------------------
# Configuración de las 8 barras — Futbolín tipo Madrid
#
# Equipo A defiende portería izquierda (x_min), ataca hacia la derecha.
# Equipo B defiende portería derecha  (x_max), ataca hacia la izquierda.
#
# Distribución (de izquierda a derecha):
#   GK-A | DEF-A | ATK-B | MID-A | MID-B | ATK-A | DEF-B | GK-B
# ---------------------------------------------------------------------------
#  idx  x_pos   n_players  equipo  rol
# shortcut: posiciones y separaciones de figuras sin medir en la mesa real (ver MESA).
_GK_X, _DEF_X, _MID_X, _ATK_X = _M["bar_x"]
BAR_DEFINITIONS = [
    (-_GK_X,  1, 'A', 'GK'),    # 0  Portero A
    (-_DEF_X, 2, 'A', 'DEF'),   # 1  Defensa A
    (-_ATK_X, 3, 'B', 'ATK'),   # 2  Delanteros B
    (-_MID_X, 5, 'A', 'MID'),   # 3  Medios A
    ( _MID_X, 5, 'B', 'MID'),   # 4  Medios B
    ( _ATK_X, 3, 'A', 'ATK'),   # 5  Delanteros A
    ( _DEF_X, 2, 'B', 'DEF'),   # 6  Defensa B
    ( _GK_X,  1, 'B', 'GK'),    # 7  Portero B
]
N_BARS = len(BAR_DEFINITIONS)

# Separación Y entre jugadores de la misma barra
_PLAYER_SPACING = {1: 0.0, **_M.get("spacing", {2: 0.14, 3: 0.14, 5: 0.12})}

# Dimensiones de la observación
OBS_DIM_V1 = 4 + N_BARS * 4          # 36: bola + 8 barras (formato histórico)
OBS_DIM_V2 = OBS_DIM_V1 + 4 * 2      # 44: + bola relativa a cada barra propia
OBS_REL_X_SCALE = 0.15               # m
OBS_REL_Y_SCALE = 0.10               # m


def _compute_offsets(n: int) -> List[float]:
    """Distribuye N jugadores simétricamente en Y."""
    s = _PLAYER_SPACING.get(n, 0.13)
    half = (n - 1) / 2.0 * s
    return [round(-half + i * s, 4) for i in range(n)]


def team_direction(team: str) -> float:
    """+1 si el equipo ataca hacia +x (A), −1 si ataca hacia −x (B)."""
    return 1.0 if team == 'A' else -1.0


def other_team(team: str) -> str:
    return 'B' if team == 'A' else 'A'


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BallState:
    """
    Estado de la bola en el plano.
    (wx, wy) es la velocidad que tendría la bola si rodara sin deslizar con su
    giro actual (ω × r). Rodando, w = v; tras un golpe, v cambia y w no, y la
    bola desliza hasta volver a rodar.
    """
    x:  float = 0.0
    y:  float = 0.0
    vx: float = 0.5
    vy: float = 0.2
    wx: float = 0.5
    wy: float = 0.2

    def reset(self, x=0.0, y=0.0, vx=0.5, vy=0.2):
        self.x, self.y, self.vx, self.vy = x, y, vx, vy
        self.wx, self.wy = vx, vy

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.vx, self.vy], dtype=np.float32)


@dataclass
class BarState:
    """
    Estado de una barra del futbolín.

    Dos grados de libertad:
      - linear_pos [m]: desplazamiento lateral (eje Y) dentro de ±linear_range
      - angle [rad]:    ángulo de rotación de la barra, 0 = pie hacia abajo
    """
    bar_x:          float       = 0.0
    n_players:      int         = 1
    team:           str         = 'A'
    role:           str         = 'GK'
    player_offsets: List[float] = field(default_factory=list)
    linear_range:   float       = LINEAR_RANGE   # m, media carrera de esta barra

    # Propiedades físicas (efectivas, incluyen motor o mano)
    inertia:     float = 1e-4  # kg·m² respecto al eje
    lin_mass:    float = 1.0   # kg en el desplazamiento lateral

    # Estado dinámico
    linear_pos:  float = 0.0   # [-linear_range, +linear_range]
    angle:       float = 0.0   # [-π, +π]
    linear_vel:  float = 0.0   # m/s
    angular_vel: float = 0.0   # rad/s

    def get_player_abs_y(self) -> List[float]:
        """Posición Y absoluta de cada jugador en el campo."""
        return [self.linear_pos + off for off in self.player_offsets]

    def foot_tip_x(self) -> float:
        """Posición X del centro de la suela del taco."""
        return self.bar_x + FOOT_LENGTH * math.sin(self.angle)

    def foot_tip_height(self) -> float:
        """Altura Z del punto más bajo del taco sobre el campo."""
        return _foot_lowest_z(self.angle)

    def figure_footprint_x(self) -> tuple:
        """Extensión en X (mín, máx) de toda la figura (cabeza, cuerpo y taco) vista desde arriba."""
        s, c = math.sin(self.angle), math.cos(self.angle)
        xs = [self.bar_x + u * s + v * c
              for u, vs in ((-HEAD_LENGTH, BODY_HALF_DEPTH), (0.0, BODY_HALF_DEPTH),
                            (FOOT_LENGTH - FOOT_HEIGHT, FOOT_HALF_LENGTH),
                            (FOOT_LENGTH, FOOT_HALF_LENGTH))
              for v in (-vs, vs)]
        return min(xs), max(xs)

    def foot_footprint_x(self) -> tuple:
        """Extensión en X (mín, máx) del taco, para dibujarlo en planta."""
        s, c = math.sin(self.angle), math.cos(self.angle)
        xs = [self.bar_x + u * s + v * c
              for u in (FOOT_LENGTH - FOOT_HEIGHT, FOOT_LENGTH)
              for v in (-FOOT_HALF_LENGTH, FOOT_HALF_LENGTH)]
        return min(xs), max(xs)

    def foot_tip_vel_x(self) -> float:
        """Velocidad tangencial de la punta del pie en X."""
        return self.angular_vel * FOOT_LENGTH * math.cos(self.angle)

    def is_in_kick_zone(self) -> bool:
        """True si el pie está lo bastante bajo para tocar la bola."""
        return abs(self.angle) < KICK_ANGLE_MAX

    def as_array(self) -> np.ndarray:
        """Vector normalizado de estado de la barra (marco del mundo)."""
        return np.array([
            self.linear_pos  / self.linear_range,
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

    def reset(self, ball_vx: float = 0.5, ball_vy: float = 0.2,
              ball_x: float = 0.0, ball_y: float = 0.0):
        self.ball.reset(x=ball_x, y=ball_y, vx=ball_vx, vy=ball_vy)
        for bar in self.bars:
            bar.reset()
        self.time = 0.0

    def get_observation(self) -> np.ndarray:
        """Observación histórica de 36 dims en el marco del equipo A."""
        return self.observation('A', version=1)

    def observation(self, team: str = 'A', version: int = 2,
                    bar_noise: np.ndarray = None,
                    ball_noise: np.ndarray = None) -> np.ndarray:
        """
        Vector de observación normalizado, visto por `team`.

        version=1 (36 dims): [bola×4, barra0×4, ..., barra7×4]
        version=2 (44 dims): lo anterior + para cada barra propia (GK, DEF, MID,
            ATK) la distancia de la bola en X a la barra y en Y al jugador más
            cercano. Son cálculos que la red podría aprender, pero darlos
            explícitamente acelera mucho el aprendizaje.

        bar_noise : (8, 2) opcional — error de encoder [m, rad] por barra.
        ball_noise: (2,) opcional   — error de la cámara [m].
        Para el equipo B el campo se gira 180°: el agente siempre ataca hacia +x.
        """
        f = self.cfg
        s = -1.0 if team == 'B' else 1.0
        bx, by = self.ball.x, self.ball.y
        if ball_noise is not None:
            bx += float(ball_noise[0])
            by += float(ball_noise[1])
        hx, hy = f.length / 2.0, f.width / 2.0
        obs = [s * bx / hx, s * by / hy,
               s * self.ball.vx / MAX_BALL_SPEED, s * self.ball.vy / MAX_BALL_SPEED]

        n = len(self.bars)
        order = range(n) if s > 0 else range(n - 1, -1, -1)
        lin_obs = [0.0] * n
        ang_obs = [0.0] * n
        for i in order:
            bar = self.bars[i]
            lin, ang = bar.linear_pos, bar.angle
            if bar_noise is not None:
                lin += float(bar_noise[i, 0])
                ang += float(bar_noise[i, 1])
            lin_obs[i], ang_obs[i] = lin, ang
            obs.extend((
                s * lin / bar.linear_range,
                s * ang / math.pi,
                s * bar.linear_vel / LINEAR_VMAX,
                s * bar.angular_vel / ANGULAR_OMEGA_MAX,
            ))

        if version >= 2:
            for i in get_team_bar_indices(team):
                bar = self.bars[i]
                rel_x = s * (bx - bar.bar_x) / OBS_REL_X_SCALE
                lin = lin_obs[i]
                rel_y = min((by - (lin + off) for off in bar.player_offsets), key=abs)
                rel_y = s * rel_y / OBS_REL_Y_SCALE
                obs.append(max(-2.0, min(2.0, rel_x)))
                obs.append(max(-2.0, min(2.0, rel_y)))
        return np.array(obs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Fábrica de mundo estándar
# ---------------------------------------------------------------------------

def create_standard_world() -> WorldState:
    """Crea un mundo con las 8 barras estándar del futbolín tipo Madrid."""
    bars = []
    for i, (bar_x, n_players, team, role) in enumerate(BAR_DEFINITIONS):
        if _M["offsets"] is not None:
            offsets = list(_M["offsets"][i])
            assert len(offsets) == n_players, (role, offsets)
        else:
            offsets = _compute_offsets(n_players)
        bars.append(BarState(
            bar_x=bar_x,
            n_players=n_players,
            team=team,
            role=role,
            player_offsets=offsets,
            linear_range=BAR_LINEAR_RANGES[i],
            inertia=bar_inertia(n_players),
            lin_mass=bar_mass(n_players),
        ))
    return WorldState(bars=bars)


def get_team_bar_indices(team: str) -> List[int]:
    """
    Índices de las barras de un equipo, ordenados GK, DEF, MID, ATK.
    A: [0, 1, 3, 5]   B: [7, 6, 4, 2]
    """
    idxs = [i for i, (_, _, t, _) in enumerate(BAR_DEFINITIONS) if t == team]
    return idxs if team == 'A' else idxs[::-1]
