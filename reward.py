"""
reward.py
---------
Función de recompensa para el futbolín tipo Madrid.

Diseño por capas:
  1. Recompensas de evento (escasas): gol, toque de bola.
  2. Recompensas densas:  bola avanzando, proximidad barra-bola.
  3. Penalizaciones:      gol en contra, movimientos excesivos.

La separación en capas permite activar/desactivar componentes
y ajustar el balance sparse/dense para distintas fases del entrenamiento.
"""

import numpy as np
from dataclasses import dataclass
from world import WorldState, MAX_BALL_SPEED, get_team_bar_indices


@dataclass
class RewardConfig:
    # --- Eventos discretos ---
    goal_reward:       float = 10.0   # gol marcado a favor
    goal_penalty:      float = -10.0  # gol recibido
    touch_reward:      float =  0.5   # bola tocada por un jugador propio

    # --- Recompensas densas (por paso) ---
    ball_direction_scale: float = 0.3   # bola moviéndose hacia portería rival
    proximity_scale:      float = 0.15  # barra más cercana a la bola
    proximity_sigma:      float = 0.12  # anchura gaussiana [m]
    ball_in_atk_zone:     float = 0.05  # bola en zona de ataque
    ball_progress_scale:  float = 0.08  # bola avanzando por el campo (shaping)

    # --- Penalizaciones ---
    action_penalty_scale: float = 0.005 # penalización por |acción| (eficiencia energética)
    jerk_penalty_scale:   float = 0.002 # penalización por cambio brusco de acción

    # --- Modo ---
    sparse_only: bool = False   # si True, solo usa recompensas de evento


def compute_reward(
    state:      WorldState,
    team:       str,
    events:     dict,
    actions:    np.ndarray,
    prev_actions: np.ndarray = None,
    config:     RewardConfig = None,
) -> tuple:
    """
    Calcula la recompensa total para el equipo controlado.

    Parámetros
    ----------
    state      : WorldState — estado post-step
    team       : str        — equipo controlado ('A' o 'B')
    events     : dict       — eventos del paso (de PhysicsEngine)
    actions    : ndarray    — acciones aplicadas, shape (n_bars_team, 2)
    prev_actions: ndarray   — acciones del paso anterior (para jerk)
    config     : RewardConfig

    Devuelve
    --------
    total_reward : float
    breakdown    : dict   — desglose de componentes
    """
    cfg = config or RewardConfig()
    bd  = {}          # breakdown
    r   = 0.0

    opponent = 'B' if team == 'A' else 'A'

    # ------------------------------------------------------------------
    # 1. Eventos discretos
    # ------------------------------------------------------------------
    goal_event = events.get("goal")
    if goal_event == team:
        r += cfg.goal_reward
        bd["goal_scored"] = cfg.goal_reward
    elif goal_event == opponent:
        r += cfg.goal_penalty
        bd["goal_conceded"] = cfg.goal_penalty

    # Toques de bola propios
    team_bar_idxs = set(get_team_bar_indices(team))
    own_touches   = [(bi, pi) for bi, pi in events.get("touches", []) if bi in team_bar_idxs]
    if own_touches:
        touch_r  = cfg.touch_reward * len(own_touches)
        r       += touch_r
        bd["touch"] = touch_r

    if cfg.sparse_only:
        return float(r), bd

    # ------------------------------------------------------------------
    # 2. Recompensa densa: dirección de la bola
    # ------------------------------------------------------------------
    ball = state.ball
    # Equipo A ataca hacia x_max (+X), equipo B hacia x_min (-X)
    direction = 1.0 if team == 'A' else -1.0
    ball_dir_r = cfg.ball_direction_scale * np.tanh(direction * ball.vx / MAX_BALL_SPEED * 3.0)
    r += ball_dir_r
    bd["ball_direction"] = ball_dir_r

    # ------------------------------------------------------------------
    # 3. Recompensa densa: proximidad — barra más cercana en X a la bola
    # ------------------------------------------------------------------
    # Usamos TODAS las barras del equipo: cada barra tiene incentivo de
    # tocar la bola cuando está en su zona. Solo puntuamos la mejor.
    own_bars = [b for b in state.bars if b.team == team]
    if own_bars:
        prox_rewards = []
        for bar in own_bars:
            for player_y in bar.get_player_abs_y():
                dist_y = abs(ball.y - player_y)
                dist_x = abs(ball.x - bar.bar_x)
                dist   = np.hypot(dist_x, dist_y)
                prox   = cfg.proximity_scale * np.exp(-0.5 * (dist / cfg.proximity_sigma) ** 2)
                prox_rewards.append(prox)
        if prox_rewards:
            best_prox = max(prox_rewards)
            r        += best_prox
            bd["proximity"] = best_prox

    # ------------------------------------------------------------------
    # 4. Bola en zona de ataque + progreso por el campo
    # ------------------------------------------------------------------
    in_zone  = (ball.x * direction) > 0   # bola en la mitad del campo rival
    zone_r   = cfg.ball_in_atk_zone if in_zone else 0.0
    r       += zone_r
    bd["zone"] = zone_r

    # Shaping: recompensa proporcional a cuánto ha avanzado la bola
    # hacia la portería rival (normalizado a [-1, 1])
    progress_r = cfg.ball_progress_scale * (ball.x * direction) / 0.60
    r         += progress_r
    bd["ball_progress"] = progress_r

    # ------------------------------------------------------------------
    # 5. Penalización por acción (eficiencia energética)
    # ------------------------------------------------------------------
    if actions is not None and len(actions) > 0:
        action_arr = np.asarray(actions)
        act_pen    = -cfg.action_penalty_scale * float(np.mean(np.abs(action_arr)))
        r         += act_pen
        bd["action_penalty"] = act_pen

        # Penalización por jerk (cambio brusco)
        if prev_actions is not None:
            prev_arr   = np.asarray(prev_actions)
            jerk       = float(np.mean(np.abs(action_arr - prev_arr)))
            jerk_pen   = -cfg.jerk_penalty_scale * jerk
            r         += jerk_pen
            bd["jerk_penalty"] = jerk_pen

    bd["total"] = float(r)
    return float(r), bd
