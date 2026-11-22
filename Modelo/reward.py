"""
reward.py
---------
Función de recompensa para el futbolín tipo Madrid.

Diseño:
  1. Eventos: gol a favor / en contra, tiro a puerta propio (y del rival).
  2. Shaping basado en potencial  F = γ·Φ(s') − Φ(s)  (Ng et al., 1999).
     No cambia la política óptima, así que puede mantenerse durante todo el
     entrenamiento sin que el agente aprenda a "farmear" la ayuda:
       Φ = w_x · (posición de la bola hacia la portería rival, en [−1, 1])
         + w_v · (velocidad de la bola hacia la portería rival, normalizada)
         + w_p · (proximidad del jugador propio más cercano a la bola)
     Al terminar el episodio Φ(terminal) = 0.
  3. Penalización pequeña por cambios bruscos de acción (suavidad para el
     hardware). Ya no se penaliza |acción|: la acción es una posición objetivo
     y esa penalización empujaba las barras al centro.

Componentes antiguos (proximity_scale, ball_direction_scale, ball_in_atk_zone,
ball_progress_scale, forward_kick_reward, action_penalty_scale) se conservan
por compatibilidad, pero valen 0 por defecto: no son de potencial y el agente
puede explotarlos.
"""

import math
import numpy as np
from dataclasses import dataclass
from world import WorldState, MAX_BALL_SPEED, get_team_bar_indices, team_direction


@dataclass
class RewardConfig:
    # --- Eventos discretos ---
    goal_reward:        float = 10.0   # gol marcado a favor
    goal_penalty:       float = -10.0  # gol recibido
    shot_reward:        float = 0.5    # toque propio que deja la bola directa a puerta
    shot_conceded:      float = -0.25  # toque rival que deja la bola directa a nuestra puerta
    touch_reward:       float = 0.0    # por toque propio (solo para arrancar)

    # --- Shaping basado en potencial ---
    potential_x:        float = 1.0    # peso de la posición de la bola
    potential_v:        float = 0.5    # peso de la velocidad de la bola
    potential_prox:     float = 0.0    # peso de la proximidad jugador-bola
    proximity_sigma:    float = 0.08   # anchura gaussiana [m]
    gamma:              float = 0.99   # debe coincidir con el γ del algoritmo

    # --- Suavidad ---
    jerk_penalty_scale: float = 0.001  # cambio brusco de acción

    # --- Componentes heredados (no de potencial; 0 por defecto) ---
    ball_direction_scale: float = 0.0
    proximity_scale:      float = 0.0
    ball_in_atk_zone:     float = 0.0
    ball_progress_scale:  float = 0.0
    forward_kick_reward:  float = 0.0
    action_penalty_scale: float = 0.0

    # --- Modo ---
    sparse_only: bool = False   # si True, solo recompensas de evento


def _proximity(state: WorldState, team: str, sigma: float) -> float:
    """Proximidad (0..1) del jugador propio más cercano a la bola."""
    ball = state.ball
    best = 0.0
    inv = 1.0 / (2.0 * sigma * sigma)
    for bar in state.bars:
        if bar.team != team:
            continue
        dx = ball.x - bar.foot_tip_x()
        if abs(dx) > 4 * sigma:
            continue
        for py in bar.get_player_abs_y():
            dy = ball.y - py
            best = max(best, math.exp(-(dx * dx + dy * dy) * inv))
    return best


def potential(state: WorldState, team: str, config: RewardConfig = None) -> float:
    """Potencial Φ(s) del estado para `team`."""
    cfg = config or RewardConfig()
    if cfg.sparse_only:
        return 0.0
    d = team_direction(team)
    ball = state.ball
    phi = 0.0
    if cfg.potential_x:
        phi += cfg.potential_x * d * ball.x / (state.field.length / 2.0)
    if cfg.potential_v:
        phi += cfg.potential_v * math.tanh(d * ball.vx / MAX_BALL_SPEED * 2.0)
    if cfg.potential_prox:
        phi += cfg.potential_prox * _proximity(state, team, cfg.proximity_sigma)
    return phi


def compute_reward(
    state:        WorldState,
    team:         str,
    events:       dict,
    actions:      np.ndarray,
    prev_actions: np.ndarray = None,
    config:       RewardConfig = None,
    phi_prev:     float = None,
    phi_next:     float = None,
) -> tuple:
    """
    Recompensa del paso para el equipo controlado.

    phi_prev / phi_next: potencial antes y después del paso (el entorno los
    calcula; phi_next = 0 en estados terminales). Si faltan, no hay shaping.

    Devuelve (total_reward, breakdown).
    """
    cfg = config or RewardConfig()
    bd  = {}
    r   = 0.0
    opponent  = 'B' if team == 'A' else 'A'
    ball      = state.ball
    direction = team_direction(team)

    # 1. Goles
    goal_event = events.get("goal")
    if goal_event == team:
        r += cfg.goal_reward
        bd["goal_scored"] = cfg.goal_reward
    elif goal_event == opponent:
        r += cfg.goal_penalty
        bd["goal_conceded"] = cfg.goal_penalty

    # 2. Toques y tiros
    team_bar_idxs = set(get_team_bar_indices(team))
    touches = events.get("touches", [])
    own_touches = [t for t in touches if t[0] in team_bar_idxs]
    if own_touches and cfg.touch_reward:
        bd["touch"] = cfg.touch_reward * len(own_touches)
        r += bd["touch"]
    shot = events.get("shot_on_target")
    if shot == team and cfg.shot_reward:
        r += cfg.shot_reward
        bd["shot"] = cfg.shot_reward
    elif shot == opponent and cfg.shot_conceded:
        r += cfg.shot_conceded
        bd["shot_conceded"] = cfg.shot_conceded
    if own_touches and cfg.forward_kick_reward:
        fwd_speed = direction * ball.vx
        if fwd_speed > 0.3:
            bd["forward_kick"] = cfg.forward_kick_reward * math.tanh(fwd_speed / MAX_BALL_SPEED * 3.0)
            r += bd["forward_kick"]

    if cfg.sparse_only:
        bd["total"] = float(r)
        return float(r), bd

    # 3. Shaping de potencial
    if phi_prev is not None and phi_next is not None:
        shaping = cfg.gamma * phi_next - phi_prev
        r += shaping
        bd["shaping"] = shaping

    # 4. Componentes heredados (0 por defecto)
    if cfg.ball_direction_scale:
        bd["ball_direction"] = cfg.ball_direction_scale * math.tanh(direction * ball.vx / MAX_BALL_SPEED * 3.0)
        r += bd["ball_direction"]
    if cfg.proximity_scale:
        bd["proximity"] = cfg.proximity_scale * _proximity(state, team, 0.12)
        r += bd["proximity"]
    if cfg.ball_in_atk_zone and ball.x * direction > 0:
        bd["zone"] = cfg.ball_in_atk_zone
        r += bd["zone"]
    if cfg.ball_progress_scale:
        bd["ball_progress"] = cfg.ball_progress_scale * (ball.x * direction) / (state.field.length / 2.0)
        r += bd["ball_progress"]

    # 5. Suavidad de acción
    if actions is not None and len(actions) > 0:
        action_arr = np.asarray(actions)
        if cfg.action_penalty_scale:
            bd["action_penalty"] = -cfg.action_penalty_scale * float(np.mean(np.abs(action_arr)))
            r += bd["action_penalty"]
        if prev_actions is not None and cfg.jerk_penalty_scale:
            jerk = float(np.mean(np.abs(action_arr - np.asarray(prev_actions))))
            bd["jerk_penalty"] = -cfg.jerk_penalty_scale * jerk
            r += bd["jerk_penalty"]

    bd["total"] = float(r)
    return float(r), bd
