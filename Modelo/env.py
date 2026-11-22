"""
env.py
------
Entorno tipo Gymnasium para el futbolín tipo Madrid.

Características:
  - 8 barras (4 por equipo), cada una con control lineal + angular
  - El agente controla un equipo (A por defecto): 4 barras × 2 acciones = 8 dims
  - El otro equipo usa un oponente: heurístico (con nivel `skill`), una política
    entrenada (self-play) o ninguno (barras quietas)
  - Física con subpasos: los actuadores se integran dentro de cada subpaso

Espacio de observación (marco del equipo controlado, siempre ataca hacia +x):
  obs_version=2 (44 dims, por defecto): bola×4, 8 barras×4, bola relativa a las
                 4 barras propias×2
  obs_version=1 (36 dims): formato histórico, para modelos antiguos

Espacio de acción (8 dims, [-1, 1]):
  Para cada barra propia, ordenadas GK, DEF, MID, ATK: [linear_cmd, angular_cmd]
  En el marco del equipo: angular_cmd > 0 lleva el pie hacia delante (ataque).

Para conectar hardware real:
  Sobreescribir _apply_agent_commands() y _get_obs().
"""

import dataclasses
import math
from collections import deque
from dataclasses import dataclass
import numpy as np
from world import (
    WorldState, BallState, create_standard_world, get_team_bar_indices,
    LINEAR_RANGE, BALL_RADIUS, PLAYER_HALF_WIDTH, OBS_DIM_V1, OBS_DIM_V2,
    BAR_DEFINITIONS, team_direction, other_team,
)
from physics import PhysicsEngine, PhysicsParams
from actuators import ActuatorBank
from reward import RewardConfig, compute_reward, potential
from hardware import DEFAULT_SPEC, SimSpec
from observation import ObservationBuilder, OBS_DIM_V3, ACTOR_DIM


try:
    from gymnasium import spaces
    GYM_VERSION = "gymnasium"
except ImportError:
    try:
        from gym import spaces
        GYM_VERSION = "gym"
    except ImportError:
        spaces = None
        GYM_VERSION = "none"


BALL_OBS_NOISE = 0.002     # m, error de la cámara (con noise=True)
STALL_RADIUS   = 0.01      # m, la bola "no se mueve" si no sale de este radio
STALL_UNREACHABLE_TIME = 0.6   # s parada fuera del alcance → se reactiva
STALL_MAX_TIME         = 3.0   # s parada en cualquier sitio → se reactiva


def action_to_commands(action: np.ndarray, team: str) -> dict:
    """
    Convierte una acción en el marco del equipo a comandos del mundo.
    Devuelve {bar_idx: (linear_cmd, angular_cmd)}.
    Para el B el campo está girado 180°: se invierten Y y el ángulo.
    """
    s = team_direction(team)
    return {bar_idx: (s * float(action[2 * k]), s * float(action[2 * k + 1]))
            for k, bar_idx in enumerate(get_team_bar_indices(team))}


# ---------------------------------------------------------------------------
# Oponente heurístico
# ---------------------------------------------------------------------------

@dataclass
class HeuristicParams:
    """
    Parámetros del heurístico. Los valores por defecto son los de un jugador
    razonable ajustado a mano; `optimize_teacher.py` los optimiza automáticamente.
    Los ángulos y órdenes están en el marco del equipo: orden > 0 lleva el pie
    hacia delante.
    """
    windup_cmd:          float = -0.6    # orden de ángulo al armar (pie atrás)
    strike_cmd:          float = 0.5     # orden de ángulo al golpear
    armed_theta:         float = -1.2    # rad; con el pie más atrás que esto, ya está armado
    strike_end_theta:    float = 1.0     # rad; golpe terminado al pasar de este ángulo
    windup_front_min:    float = 0.035   # m; distancia mínima delante de la barra para armar
    windup_front_max:    float = 0.09    # m; distancia máxima
    strike_trigger:      float = 0.035   # m; si la bola está más cerca que esto, golpear ya
    reach_max:           float = 0.10    # m; alcance máximo del golpe
    approach_vmin:       float = -0.15   # m/s; velocidad mínima de la bola hacia el pie para armar
    mode_timeout:        float = 15.0    # pasos; abortar si se bloquea
    lift_cmd:            float = -0.5    # pose levantada si la bola está detrás de la barra
    rest_cmd:            float = 0.0     # pose de defensa (pie abajo)
    aim_offset:          float = 0.0     # m; desplazamiento lateral hacia el centro de la portería rival
    stagger:             float = 0.5     # fracción: el delantero no arma si el medio ya está armado

    def to_array(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in PARAM_BOUNDS], dtype=float)

    @classmethod
    def from_array(cls, x) -> "HeuristicParams":
        p = cls()
        for (name, (lo, hi)), v in zip(PARAM_BOUNDS.items(), x):
            setattr(p, name, float(min(max(v, lo), hi)))
        return p

    def save(self, path: str):
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "HeuristicParams":
        import json
        with open(path, encoding="utf-8") as f:
            return cls(**json.load(f))


PARAM_BOUNDS = {
    "windup_cmd":       (-1.0, -0.25),
    "strike_cmd":       (0.2, 1.0),
    "armed_theta":      (-2.4, -0.6),
    "strike_end_theta": (0.4, 1.6),
    "windup_front_min": (0.015, 0.06),
    "windup_front_max": (0.06, 0.14),
    "strike_trigger":   (0.015, 0.06),
    "reach_max":        (0.07, 0.16),
    "approach_vmin":    (-0.8, 0.2),
    "mode_timeout":     (8.0, 30.0),
    "lift_cmd":         (-0.9, 0.9),
    "rest_cmd":         (-0.2, 0.3),
    "aim_offset":       (-0.03, 0.03),
    "stagger":          (0.0, 1.0),
}


class HeuristicOpponent:
    """
    Jugador heurístico para cualquier equipo.

    Cada barra:
      - Coloca en la trayectoria de la bola al jugador que mejor la alcanza
        (no el centro de la barra).
      - Si la bola está a su alcance delante del pie: prepara (pie atrás) y
        golpea hacia la portería rival.
      - Si la bola está detrás de la barra, la levanta hacia atrás para no tapar
        a sus compañeros ni arrastrar la bola al armar.
      - En otro caso deja el pie abajo para bloquear.

    skill ∈ [0, 1]: con menos nivel reacciona más tarde, se coloca con error y
    golpea con menos frecuencia. `rng` controla esa aleatoriedad.

    Percepción con retardo: decide con la bola de hace `reaction_delay` s
    (60 ms con skill=1, 200 ms con skill=0), como una persona o un sistema con
    cámara. Sin retardo la defensa es perfecta y casi no hay goles.
    """

    def __init__(self, skill: float = 1.0, rng: np.random.Generator = None,
                 reaction_delay: float = None, dt: float = 0.02,
                 params: HeuristicParams = None):
        self.rng = rng or np.random.default_rng()
        self.dt = dt
        self.params = params or HeuristicParams()
        self._fixed_delay = reaction_delay
        self._ball_hist = deque(maxlen=1)
        self.set_skill(skill)
        self._aim_error = {}
        self._kick_ok = {}
        self._striking = {}
        self._mode_time = {}
        self._t = 0

    def set_skill(self, skill: float):
        self.skill = float(np.clip(skill, 0.0, 1.0))
        delay = self._fixed_delay
        if delay is None:
            delay = 0.06 + 0.14 * (1.0 - self.skill)
        steps = max(int(round(delay / self.dt)), 0)
        if self._ball_hist.maxlen != steps + 1:
            self._ball_hist = deque(maxlen=steps + 1)

    def reset(self, rng: np.random.Generator = None):
        if rng is not None:
            self.rng = rng
        self._aim_error.clear()
        self._kick_ok.clear()
        self._striking.clear()
        self._mode_time.clear()
        self._ball_hist.clear()
        self._t = 0

    def _refresh(self, idxs):
        # Error de colocación y decisión de golpeo, renovados cada ~0.3 s
        sigma = (1.0 - self.skill) * 0.03
        p_kick = 0.35 + 0.65 * self.skill
        for i in idxs:
            self._aim_error[i] = float(self.rng.normal(0.0, sigma)) if sigma > 0 else 0.0
            self._kick_ok[i] = bool(self.rng.random() < p_kick)

    def get_commands(self, state: WorldState, team: str = 'B') -> dict:
        P = self.params
        b = state.ball
        self._ball_hist.append((b.x, b.y, b.vx, b.vy))
        ball  = BallState(*self._ball_hist[0])      # bola percibida (con retardo)
        field = state.field
        d     = team_direction(team)
        idxs  = get_team_bar_indices(team)
        if self._t % 15 == 0 or not self._aim_error:
            self._refresh(idxs)
        self._t += 1

        cmds = {}
        armed_bars = 0
        for idx in idxs:
            bar = state.bars[idx]
            approaching = ball.vx * (bar.bar_x - ball.x) > 0
            if approaching and abs(ball.vx) > 0.05:
                target_y = PhysicsEngine.predict_ball_y(ball, bar.bar_x, field)
            else:
                target_y = ball.y
            target_y += self._aim_error.get(idx, 0.0)
            # Desplazamiento lateral hacia el centro de la portería rival: el
            # golpe sale algo diagonal gracias al rozamiento con el pie en movimiento
            target_y -= P.aim_offset * math.copysign(1.0, target_y) if abs(target_y) > 0.02 else 0.0

            # Jugador que mejor alcanza target_y
            best_lin, best_err = 0.0, float("inf")
            for off in bar.player_offsets:
                lin = min(max(target_y - off, -LINEAR_RANGE), LINEAR_RANGE)
                err = abs(lin + off - target_y) + 0.05 * abs(lin - bar.linear_pos)
                if err < best_err:
                    best_lin, best_err = lin, err
            linear_cmd = best_lin / LINEAR_RANGE

            # Golpeo
            front = (ball.x - bar.bar_x) * d          # >0: bola delante del pie
            fwd_v = ball.vx * d                       # >0: bola va hacia la portería rival
            aligned = min(abs(ball.y - (bar.linear_pos + off))
                          for off in bar.player_offsets) < PLAYER_HALF_WIDTH + BALL_RADIUS
            theta = bar.angle * d                     # ángulo en el marco del equipo

            # Modo por barra: idle → windup (pie atrás) → strike (golpe adelante)
            mode = self._striking.get(idx, "idle")
            reach = aligned and -0.005 < front < P.reach_max
            timer = self._mode_time.get(idx, 0) + 1
            if mode in ("strike", "windup") and timer > P.mode_timeout:
                mode = "idle"                         # bloqueado demasiado tiempo: abortar
            elif mode == "strike" and (theta > P.strike_end_theta or not reach):
                mode = "idle"                         # golpe terminado
            elif mode == "windup":
                if not reach:
                    mode = "idle"                     # la bola se fue
                elif theta < P.armed_theta or front < P.strike_trigger:
                    mode = "strike"                   # armado, o la bola se acerca: golpear ya
            elif mode == "idle":
                # Armar solo con la bola delante, separada del taco y sin venir
                # deprisa hacia el pie (si no, el pie que se retira la arrastra atrás).
                if (aligned and P.windup_front_min < front < P.windup_front_max
                        and fwd_v > P.approach_vmin and self._kick_ok.get(idx, True)):
                    mode = "windup"

            if mode == "strike":
                angular = P.strike_cmd
            elif mode == "windup":
                angular = P.windup_cmd
            elif front < -0.03:
                # Bola detrás de esta barra: levantarla hacia atrás. Así no tapa los
                # pases de los compañeros y, al armar luego, el pie no pasa sobre la bola.
                angular = P.lift_cmd
            else:
                angular = P.rest_cmd                  # pie abajo, bloquear
            if mode != self._striking.get(idx, "idle"):
                timer = 0
            self._mode_time[idx] = timer
            self._striking[idx] = mode
            # linear_cmd ya está en el marco del mundo; el ángulo, en el del equipo
            cmds[idx] = (linear_cmd, d * angular)
        return cmds


class AbsentOpponent:
    """Rival "ausente": mantiene todos sus jugadores levantados (la bola pasa por debajo)."""

    def reset(self, rng=None):
        pass

    def get_commands(self, state: WorldState, team: str = 'B') -> dict:
        d = team_direction(team)
        return {i: (0.0, d * 0.5) for i in get_team_bar_indices(team)}


class PolicyOpponent:
    """
    Oponente controlado por una política entrenada (self-play).
    `model` es cualquier objeto con predict(obs, deterministic) -> (action, _),
    o una ruta a un modelo PPO. Ve el campo en su propio marco; con modelos v3
    usa su propia cámara simulada (el entorno la enlaza con `bind`).
    """

    def __init__(self, model, deterministic: bool = False):
        if isinstance(model, str):
            from stable_baselines3 import PPO
            model = PPO.load(model, device="cpu")
        self.model = model
        self.deterministic = deterministic
        shape = getattr(getattr(model, "observation_space", None), "shape", None)
        self.obs_version = {(OBS_DIM_V1,): 1, (OBS_DIM_V2,): 2}.get(shape, 3)
        self._builder = None
        self._last_action = np.zeros(8, dtype=np.float32)

    def bind(self, env, team: str):
        if self.obs_version == 3:
            self._builder = env.add_observer(team)

    def reset(self, rng=None):
        self._last_action = np.zeros(8, dtype=np.float32)

    def get_commands(self, state: WorldState, team: str = 'B') -> dict:
        if self._builder is not None:
            obs = self._builder.build(self._last_action)
        else:
            obs = state.observation(team, version=min(self.obs_version, 2))
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        action = np.clip(action, -1.0, 1.0)
        self._last_action = action.astype(np.float32)
        return action_to_commands(action, team)


# ---------------------------------------------------------------------------
# Entorno principal
# ---------------------------------------------------------------------------

class FoosballEnv:
    """
    Entorno de simulación del futbolín tipo Madrid.

    Parámetros
    ----------
    dt : paso de control [s].
    max_steps : pasos máximos por episodio.
    controlled_team : 'A' o 'B' (el equipo del robot).
    opponent : "heuristic" | "absent" | None | objeto con get_commands(state, team)
               (None = barras rivales quietas con el pie abajo; "absent" = levantadas)
    opponent_skill : nivel del heurístico (0..1), o (min, max) para sortearlo
               en cada partida.
    noise : ruido de sensores y actuadores + aleatorización de dominio.
    randomize : aleatorización de dominio por episodio. Por defecto igual que `noise`.
    realistic : True (por defecto): robot con motores paso a paso y lazo abierto,
               rival humano, choques con masas reales y bola con giro.
               False: actuadores ideales y física simplificada (modelo anterior).
    obs_version : 3 (por defecto con realistic): cámara + firmware + memoria +
               parte privilegiada para el crítico. 2 / 1: estado perfecto (44 / 36
               dims) para modelos antiguos.
    spec : hardware.SimSpec con motores, mecánica, rival y cámara.
    continue_after_goal : True (por defecto): el partido sigue tras un gol, como en
               la mesa real: saca el portero del equipo que lo ha recibido, y el
               episodio dura `max_steps` pasos. False: el episodio termina en el
               primer gol (formato antiguo).
    """

    metadata = {"render_modes": ["text"]}

    def __init__(
        self,
        dt:               float = 0.02,
        max_steps:        int   = 750,
        controlled_team:  str   = 'A',
        opponent               = "heuristic",
        noise:            bool  = True,
        reward_config:    RewardConfig = None,
        sparse_reward:    bool  = False,
        opponent_skill          = 1.0,
        randomize:        bool  = None,
        substeps:         int   = 6,
        obs_version:      int   = None,
        realistic:        bool  = True,
        spec:             SimSpec = None,
        continue_after_goal: bool = True,
    ):
        if controlled_team not in ('A', 'B'):
            raise ValueError("controlled_team debe ser 'A' o 'B'")
        if obs_version is None:
            obs_version = 3 if realistic else 2
        if obs_version not in (1, 2, 3):
            raise ValueError("obs_version debe ser 1, 2 o 3")
        self.dt              = dt
        self.max_steps       = max_steps
        self.controlled_team = controlled_team
        self.opponent_team   = other_team(controlled_team)
        self.noise           = noise
        self.randomize       = noise if randomize is None else randomize
        self.sparse_reward   = sparse_reward
        self.obs_version     = obs_version
        self.realistic       = realistic
        self.spec            = spec or DEFAULT_SPEC
        self.reward_config   = reward_config or RewardConfig(sparse_only=sparse_reward)
        self.opponent_skill  = opponent_skill
        self.continue_after_goal = continue_after_goal

        self._rng = np.random.default_rng()
        if opponent == "heuristic":
            self.opponent = HeuristicOpponent(skill=self._draw_skill(), rng=self._rng, dt=dt)
        elif opponent == "absent":
            self.opponent = AbsentOpponent()
        else:
            self.opponent = opponent

        self.state = create_standard_world()
        params = PhysicsParams() if realistic else PhysicsParams(
            finite_inertia=False, spin=False, max_speed=4.0)
        self.physics = PhysicsEngine(dt=dt, noise=noise, substeps=substeps,
                                     params=params, randomize=self.randomize)
        if realistic:
            kinds = ["stepper" if t == controlled_team else "human"
                     for _, _, t, _ in BAR_DEFINITIONS]
        else:
            kinds = ["ideal"] * len(BAR_DEFINITIONS)
        self.actuators = ActuatorBank(
            n_bars=8, dt=dt, add_noise=noise, randomize=self.randomize, kinds=kinds,
            n_players=[n for _, n, _, _ in BAR_DEFINITIONS], spec=self.spec)

        self._team_bar_idxs = get_team_bar_indices(controlled_team)
        self._n_agent_bars  = len(self._team_bar_idxs)   # 4
        self.action_dim = self._n_agent_bars * 2          # 8

        self._observers = []
        self._builder = self.add_observer(controlled_team) if obs_version == 3 else None
        self.obs_dim = {1: OBS_DIM_V1, 2: OBS_DIM_V2, 3: OBS_DIM_V3}[obs_version]
        self.actor_dim = ACTOR_DIM if obs_version == 3 else self.obs_dim

        if spaces is not None:
            lim = 2.0
            self.observation_space = spaces.Box(
                low=-lim, high=lim, shape=(self.obs_dim,), dtype=np.float32)
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)

        self._step_count:     int   = 0
        self._episode_reward: float = 0.0
        self._prev_actions:   np.ndarray = np.zeros(self.action_dim, dtype=np.float32)
        self._score_info:     dict  = {"A": 0, "B": 0}
        self._stall_time:     float = 0.0
        self._stall_anchor:   tuple = (0.0, 0.0)
        self._phi:            float = 0.0
        self._last_touch:     str   = None
        self._pending = []        # comandos del robot en camino (latencia)
        self._t_step = 0.0

        def bar_update(h):
            # Comandos del robot que llegan a la placa en este subpaso
            if self._pending:
                self._t_step += h
                while self._pending and self._pending[0][0] <= self._t_step + 1e-9:
                    _, cmds = self._pending.pop(0)
                    for bar_idx, (lc, ac) in cmds.items():
                        self.actuators.set_commands(bar_idx, lc, ac)
            for obs in self._observers:
                obs.record()
            self.actuators.apply_to(self.state.bars, h)

        def on_bar_change(bar_idx):
            self.actuators.sync_from_bar(bar_idx, self.state.bars[bar_idx])

        self._bar_update = bar_update
        self._on_bar_change = on_bar_change

        if hasattr(self.opponent, "bind"):
            self.opponent.bind(self, self.opponent_team)

    # ------------------------------------------------------------------
    # Observadores (cámara simulada por equipo)
    # ------------------------------------------------------------------

    def add_observer(self, team: str) -> ObservationBuilder:
        builder = ObservationBuilder(self, team)
        self._observers.append(builder)
        return builder

    def _draw_skill(self) -> float:
        s = self.opponent_skill
        if isinstance(s, (tuple, list)):
            return float(self._rng.uniform(s[0], s[1]))
        return float(s)

    # ------------------------------------------------------------------
    # Interfaz Gymnasium
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        actuator_seed = None
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.physics.rng = np.random.default_rng(self._rng.integers(2**32))
            actuator_seed = int(self._rng.integers(2**32))

        # Nivel del rival para esta partida (también limita la velocidad de sus manos)
        skill = self._draw_skill()
        if hasattr(self.opponent, "set_skill"):
            self.opponent.set_skill(skill)
        for i in get_team_bar_indices(self.opponent_team):
            act = self.actuators[i]
            if getattr(act, "kind", "") == "human" and self.realistic:
                act.skill_scale = 0.6 + 0.4 * skill
        if hasattr(self.opponent, "reset"):
            self.opponent.reset(rng=np.random.default_rng(self._rng.integers(2**32)))

        # Saque desde el centro hacia un lado aleatorio
        vx0 = self._rng.uniform(0.4, 1.0) * self._rng.choice([-1, 1])
        vy0 = self._rng.uniform(-0.5, 0.5)
        self.state.reset(ball_vx=vx0, ball_vy=vy0)
        self.state.score = [0, 0]
        self.actuators.reset(seed=actuator_seed)
        self.actuators.write_physical_properties(self.state.bars)
        self.physics.randomize_params()

        self._step_count     = 0
        self._episode_reward = 0.0
        self._prev_actions   = np.zeros(self.action_dim, dtype=np.float32)
        self._score_info     = {"A": 0, "B": 0}
        self._stall_time     = 0.0
        self._stall_anchor   = (self.state.ball.x, self.state.ball.y)
        self._last_touch     = None
        self._pending        = []
        self._phi = potential(self.state, self.controlled_team, self.reward_config)
        for obs in self._observers:
            obs.reset(rng=np.random.default_rng(self._rng.integers(2**32)))

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        """
        action : (8,) — [lin, ang] × (GK, DEF, MID, ATK) del equipo controlado,
                 en el marco del equipo.
        """
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        # 1. Comandos del agente (con latencia) y del oponente
        self._apply_agent_commands(action)
        if self.opponent is not None:
            for bar_idx, (lc, ac) in self.opponent.get_commands(
                    self.state, team=self.opponent_team).items():
                self.actuators.set_commands(bar_idx, lc, ac)

        # 2. Física con actuadores integrados en cada subpaso
        self._t_step = 0.0
        ball_before = (self.state.ball.vx, self.state.ball.vy)
        events = self.physics.step(self.state, self._bar_update, self._on_bar_change)
        # Lo que quede en camino llega antes del siguiente paso
        if self._pending:
            for _, cmds in self._pending:
                for bar_idx, (lc, ac) in cmds.items():
                    self.actuators.set_commands(bar_idx, lc, ac)
            self._pending = []
        self.actuators.end_of_step()

        # 3. Último equipo en tocar + tiro a puerta
        touch_teams = {self.state.bars[b].team for b, _ in events["touches"]}
        if len(touch_teams) == 1:
            self._last_touch = touch_teams.pop()
        own = self.controlled_team
        events["shot_on_target"] = None
        for team in (own, self.opponent_team):
            if any(self.state.bars[b].team == team for b, _ in events["touches"]) \
                    and PhysicsEngine.shot_on_target(self.state.ball, self.state.field, team):
                events["shot_on_target"] = team
        events["ball_speed_change"] = math.hypot(
            self.state.ball.vx - ball_before[0], self.state.ball.vy - ball_before[1])

        # 4. Marcador
        if events.get("goal") == 'A':
            self._score_info["A"] += 1
            self.state.score[0]   += 1
        elif events.get("goal") == 'B':
            self._score_info["B"] += 1
            self.state.score[1]   += 1

        events["serve"] = None
        if events.get("goal") is not None and self.continue_after_goal:
            # Saca el portero del equipo que ha recibido el gol
            self._serve(other_team(events["goal"]))
            events["serve"] = other_team(events["goal"])
        terminated = bool(events.get("ball_out", False)) and not self.continue_after_goal
        self._step_count += 1
        truncated = (not terminated) and self._step_count >= self.max_steps

        # 5. Recompensa (con shaping basado en potencial)
        # Un gol cierra un "tramo" del partido: Φ(terminal) = 0 y el siguiente tramo
        # empieza en la bola ya recolocada para el saque, sin contar ese salto.
        goal_scored = events.get("goal") is not None
        phi_next = 0.0 if (terminated or goal_scored) else potential(self.state, own, self.reward_config)
        actions_2d = action.reshape(self._n_agent_bars, 2)
        prev_2d    = self._prev_actions.reshape(self._n_agent_bars, 2)
        reward, breakdown = compute_reward(
            state=self.state, team=own, events=events,
            actions=actions_2d, prev_actions=prev_2d, config=self.reward_config,
            phi_prev=self._phi, phi_next=phi_next,
        )
        self._phi = (potential(self.state, own, self.reward_config)
                     if goal_scored and not terminated else phi_next)
        self._prev_actions    = action.copy()
        self._episode_reward += reward

        # 6. Bola muerta
        if not terminated and events["serve"] is None:
            self._handle_stall()

        lost = [self.actuators[i].lost_steps() for i in self._team_bar_idxs]
        info = {
            "step":           self._step_count,
            "episode_reward": self._episode_reward,
            "score":          self._score_info.copy(),
            "events":         events,
            "reward_bd":      breakdown,
            "last_touch":     self._last_touch,
            "lost_steps":     int(sum(a + b for a, b in lost)),
        }
        return self._get_obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _serve(self, team: str):
        """
        Saque de portería: la bola se deja parada delante del portero de `team`,
        dentro del ancho de la portería, y él la pone en juego.
        """
        gk = self.state.bars[get_team_bar_indices(team)[0]]
        d = team_direction(team)
        x = gk.bar_x + d * float(self._rng.uniform(0.045, 0.06))
        y = float(self._rng.uniform(-0.05, 0.05))
        self.state.ball.reset(x=x, y=y, vx=0.0, vy=0.0)
        self._stall_time = 0.0
        self._stall_anchor = (x, y)
        self._last_touch = None

    def _handle_stall(self):
        # Por desplazamiento, no por velocidad: una bola encajada puede vibrar.
        ball = self.state.ball
        ax, ay = self._stall_anchor
        if math.hypot(ball.x - ax, ball.y - ay) > STALL_RADIUS:
            self._stall_anchor = (ball.x, ball.y)
            self._stall_time = 0.0
            return
        self._stall_time += self.dt
        reachable = PhysicsEngine.ball_reachable(self.state)
        limit = STALL_MAX_TIME if reachable else STALL_UNREACHABLE_TIME
        if self._stall_time < limit:
            return
        self._stall_time = 0.0
        self._stall_anchor = (ball.x, ball.y)
        # Empujón hacia el centro del campo con algo de aleatoriedad
        ang = math.atan2(-ball.y, -ball.x) + self._rng.uniform(-0.6, 0.6)
        speed = self._rng.uniform(0.3, 0.6)
        ball.vx = ball.wx = speed * math.cos(ang)
        ball.vy = ball.wy = speed * math.sin(ang)

    def _apply_agent_commands(self, action: np.ndarray):
        """Acción en el marco del equipo → comandos de los actuadores (tras la latencia)."""
        cmds = action_to_commands(action, self.controlled_team)
        latency = self.spec.robot.command_latency if self.realistic else 0.0
        if latency <= 0.0:
            for bar_idx, (lc, ac) in cmds.items():
                self.actuators.set_commands(bar_idx, lc, ac)
        else:
            self._pending.append((latency, cmds))

    def _get_obs(self) -> np.ndarray:
        """Observación normalizada en el marco del equipo controlado."""
        if self._builder is not None:
            return np.clip(self._builder.build(self._prev_actions), -2.0, 2.0)
        bar_noise = ball_noise = None
        if self.noise:
            bar_noise = self.actuators.encoder_noise()
            ball_noise = self._rng.uniform(-BALL_OBS_NOISE, BALL_OBS_NOISE, size=2)
        obs = self.state.observation(self.controlled_team, self.obs_version,
                                     bar_noise=bar_noise, ball_noise=ball_noise)
        return np.clip(obs, -2.0, 2.0)

    # ------------------------------------------------------------------
    # Render ASCII
    # ------------------------------------------------------------------

    def render(self, mode: str = "text"):
        ball  = self.state.ball
        field = self.state.field
        COLS, ROWS = 50, 12

        col = int((ball.x - field.x_min) / field.length * (COLS - 1))
        row = int((ball.y - field.y_min) / field.width  * (ROWS - 1))
        col = max(0, min(COLS - 1, col))
        row = max(0, min(ROWS - 1, row))

        grid = [["·"] * COLS for _ in range(ROWS)]
        grid[ROWS - 1 - row][col] = "O"

        for bar in self.state.bars:
            bx = int((bar.bar_x - field.x_min) / field.length * (COLS - 1))
            bx = max(0, min(COLS - 1, bx))
            for py in bar.get_player_abs_y():
                br = int((py - field.y_min) / field.width * (ROWS - 1))
                br = max(0, min(ROWS - 1, br))
                sym = "A" if bar.team == 'A' else "B"
                if not bar.is_in_kick_zone():
                    sym = sym.lower()
                if grid[ROWS - 1 - br][bx] != "O":
                    grid[ROWS - 1 - br][bx] = sym

        print(f"\nStep {self._step_count:4d} | t={self.state.time:.2f}s | "
              f"Score A:{self._score_info['A']} B:{self._score_info['B']}")
        print("+" + "-" * COLS + "+")
        for row_data in grid:
            print("|" + "".join(row_data) + "|")
        print("+" + "-" * COLS + "+")
        print(f"Ball: ({ball.x:+.3f}, {ball.y:+.3f})  "
              f"v=({ball.vx:+.3f}, {ball.vy:+.3f})")

    @property
    def score(self) -> dict:
        return self._score_info.copy()
