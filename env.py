"""
env.py
------
Entorno tipo Gymnasium para el futbolín tipo Madrid.

Características:
  - 8 barras (4 por equipo), cada una con control lineal + angular
  - El agente controla el equipo A (4 barras × 2 acciones = 8 dims)
  - El equipo B usa un oponente heurístico configurable
  - Compatible con stable-baselines3 / gymnasium

Espacio de observación (36 dims, todos normalizados ≈ [-1, 1]):
  [ball_x, ball_y, ball_vx, ball_vy,  bar0×4, ..., bar7×4]

Espacio de acción (8 dims, [-1, 1]):
  Para cada barra del equipo A: [linear_cmd, angular_cmd]
  Orden: GK(0), ATK(2), MID(4), DEF(6)

Para conectar hardware real:
  Sobreescribir _apply_commands() y _read_sensors().
"""

import numpy as np
from world import (
    WorldState, create_standard_world, get_team_bar_indices,
    LINEAR_RANGE, MAX_BALL_SPEED,
)
from physics import PhysicsEngine
from actuators import ActuatorBank
from reward import RewardConfig, compute_reward


# Intentar importar gymnasium; si no, usar gym clásico como fallback
try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_VERSION = "gymnasium"
except ImportError:
    try:
        import gym
        from gym import spaces
        GYM_VERSION = "gym"
    except ImportError:
        gym    = None
        spaces = None
        GYM_VERSION = "none"


# ---------------------------------------------------------------------------
# Oponente heurístico (equipo B)
# ---------------------------------------------------------------------------

class HeuristicOpponent:
    """
    Oponente heurístico simple para el equipo B.
    Cada barra intenta alinearse con la bola en Y y bloquear.
    La barra de ataque intenta golpear cuando la bola está cerca.
    """

    def get_commands(self, state: WorldState, team: str = 'B') -> dict:
        """
        Devuelve {bar_idx: (linear_cmd, angular_cmd)} para las barras del equipo.
        """
        cmds    = {}
        ball    = state.ball
        field   = state.field
        bar_idxs = get_team_bar_indices(team)

        for idx in bar_idxs:
            bar = state.bars[idx]

            # Predecir posición Y de la bola en la barra
            pred_y = PhysicsEngine.predict_ball_y(ball, bar.bar_x, field)

            # Normalizar al rango de acción
            linear_cmd = float(np.clip(pred_y / LINEAR_RANGE, -1.0, 1.0))

            # Ángulo: si la bola está cerca, golpear (rotar rápido)
            dist_x      = abs(ball.x - bar.bar_x)
            angular_cmd = -1.0 if dist_x < 0.15 else 0.0  # kick si bola cercana

            cmds[idx] = (linear_cmd, angular_cmd)

        return cmds


# ---------------------------------------------------------------------------
# Entorno principal
# ---------------------------------------------------------------------------

class FoosballEnv:
    """
    Entorno de simulación del futbolín tipo Madrid.

    Parámetros
    ----------
    dt : float
        Paso de tiempo [s].
    max_steps : int
        Pasos máximos por episodio.
    controlled_team : str
        Equipo controlado por el agente ('A' o 'B').
    opponent : objeto con .get_commands(state) o None (oponente estático)
    noise : bool
        Activar incertidumbre física.
    reward_config : RewardConfig, opcional
    sparse_reward : bool
        Usar solo recompensas de evento.
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
    ):
        self.dt              = dt
        self.max_steps       = max_steps
        self.controlled_team = controlled_team
        self.noise           = noise
        self.sparse_reward   = sparse_reward
        self.reward_config   = reward_config or RewardConfig(sparse_only=sparse_reward)

        # Oponente
        if opponent == "heuristic":
            self.opponent = HeuristicOpponent()
        else:
            self.opponent = opponent   # puede ser None o un objeto con get_commands()

        # Construir mundo
        self.state   = create_standard_world()
        self.physics = PhysicsEngine(dt=dt, noise=noise)
        self.actuators = ActuatorBank(n_bars=8, dt=dt, add_noise=noise)

        # Índices de barras del equipo controlado
        self._team_bar_idxs = get_team_bar_indices(controlled_team)
        self._n_agent_bars  = len(self._team_bar_idxs)   # 4

        # Dimensiones de espacios
        self.obs_dim    = 4 + 8 * 4   # 36
        self.action_dim = self._n_agent_bars * 2   # 8

        # Espacios Gym (si está disponible)
        if spaces is not None:
            self.observation_space = spaces.Box(
                low=-2.0, high=2.0,
                shape=(self.obs_dim,),
                dtype=np.float32,
            )
            self.action_space = spaces.Box(
                low=-1.0, high=1.0,
                shape=(self.action_dim,),
                dtype=np.float32,
            )

        # Estado interno
        self._step_count:     int   = 0
        self._episode_reward: float = 0.0
        self._prev_actions:   np.ndarray = np.zeros(self.action_dim)
        self._score_info:     dict  = {"A": 0, "B": 0}

    # ------------------------------------------------------------------
    # Interfaz Gymnasium
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        vx0 = np.random.uniform(0.4, 1.0) * np.random.choice([-1, 1])
        vy0 = np.random.uniform(-0.5, 0.5)
        self.state.reset(ball_vx=vx0, ball_vy=vy0)
        self.actuators.reset()

        self._step_count     = 0
        self._episode_reward = 0.0
        self._prev_actions   = np.zeros(self.action_dim)
        self._score_info     = {"A": 0, "B": 0}

        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray):
        """
        Ejecuta un paso de simulación.

        action : ndarray shape (8,) — [lin0, ang0, lin1, ang1, lin2, ang2, lin3, ang3]
                 para las 4 barras del equipo controlado (GK, ATK, MID, DEF).
        """
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # 1. Aplicar acciones del agente
        self._apply_agent_commands(action)

        # 2. Aplicar acciones del oponente
        if self.opponent is not None:
            opp_cmds = self.opponent.get_commands(
                self.state,
                team='B' if self.controlled_team == 'A' else 'A',
            )
            for bar_idx, (lc, ac) in opp_cmds.items():
                self.actuators.set_commands(bar_idx, lc, ac)

        # 3. Actualizar todos los actuadores → sincronizar con estado del mundo
        act_results = self.actuators.update()
        for i, (lin_pos, angle, lin_vel, ang_vel) in enumerate(act_results):
            bar             = self.state.bars[i]
            bar.linear_pos  = lin_pos
            bar.angle       = angle
            bar.linear_vel  = lin_vel
            bar.angular_vel = ang_vel

        # 4. Actualizar física
        events = self.physics.step(self.state)

        # 5. Actualizar marcador
        if events.get("goal") == 'A':
            self._score_info["A"] += 1
            self.state.score[0]   += 1
        elif events.get("goal") == 'B':
            self._score_info["B"] += 1
            self.state.score[1]   += 1

        # 6. Recompensa
        actions_2d = action.reshape(self._n_agent_bars, 2)
        prev_2d    = self._prev_actions.reshape(self._n_agent_bars, 2)
        reward, breakdown = compute_reward(
            state        = self.state,
            team         = self.controlled_team,
            events       = events,
            actions      = actions_2d,
            prev_actions = prev_2d,
            config       = self.reward_config,
        )

        self._prev_actions    = action.copy()
        self._step_count     += 1
        self._episode_reward += reward

        # 7. Relanzar bola si se queda parada (velocidad < umbral)
        ball = self.state.ball
        if np.hypot(ball.vx, ball.vy) < 0.08:
            rng = np.random.default_rng()
            ball.vx = rng.choice([-1, 1]) * rng.uniform(0.4, 0.8)
            ball.vy = rng.uniform(-0.4, 0.4)

        # 8. Condiciones de fin
        terminated = events.get("ball_out", False)
        truncated  = self._step_count >= self.max_steps

        info = {
            "step":           self._step_count,
            "episode_reward": self._episode_reward,
            "score":          self._score_info.copy(),
            "events":         events,
            "reward_bd":      breakdown,
        }

        return self._get_obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Métodos internos (sobreescribir para hardware)
    # ------------------------------------------------------------------

    def _apply_agent_commands(self, action: np.ndarray):
        """
        Convierte el vector de acción en comandos por barra.
        action: [lin0, ang0, lin1, ang1, lin2, ang2, lin3, ang3]
        """
        for k, bar_idx in enumerate(self._team_bar_idxs):
            linear_cmd  = float(action[2 * k])
            angular_cmd = float(action[2 * k + 1])
            self.actuators.set_commands(bar_idx, linear_cmd, angular_cmd)

    def _get_obs(self) -> np.ndarray:
        """Vector de observación normalizado (36 dims)."""
        return self.state.get_observation()

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

    # ------------------------------------------------------------------
    # Propiedades
    # ------------------------------------------------------------------

    @property
    def score(self) -> dict:
        return self._score_info.copy()
