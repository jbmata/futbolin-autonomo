"""
observation.py
--------------
Observación realista (versión 3) para el equipo del robot.

El vector tiene dos partes:

  ACTOR (lo que el robot real tendrá, ACTOR_DIM = FRAME_DIM × HISTORY):
    por fotograma, en el marco del equipo (siempre ataca hacia +x):
      bola según la cámara: x, y, vx, vy (filtro α-β), visible, tiempo sin verla   6
      barras propias según el firmware (pasos enviados): lin, ang, v_lin, ω   ×4  16
      barras rivales según la cámara: lin, ang, v_lin, ω                     ×4  16
      barras propias según la cámara: lin, ang (delatan pasos perdidos)      ×4   8
      bola relativa a cada barra propia: dx, dy al jugador más cercano        ×4   8
      acción anterior                                                              8
    y se apilan los últimos HISTORY fotogramas (memoria corta: fase del golpe,
    aceleraciones, latencia).

  PRIVILEGIADA (solo para el crítico durante el entrenamiento, PRIV_DIM):
      bola real (posición, velocidad y giro), las 8 barras reales, bola
      relativa real y error de la estimación propia (pasos perdidos).

La política (actor) solo lee las primeras ACTOR_DIM componentes, así que en el
robot basta con construir esa parte. Ver policies.py.
"""

import os
from collections import deque
import numpy as np

from world import (ANGULAR_RANGE, LINEAR_RANGE, BALL_RADIUS, OBS_REL_X_SCALE, OBS_REL_Y_SCALE,
                   get_team_bar_indices, team_direction)
from vision import Camera

BALL_V_SCALE = 5.0     # m/s
LIN_V_SCALE  = 1.0     # m/s
ANG_V_SCALE  = 50.0    # rad/s
AGE_SCALE    = 0.25    # s

# Fotogramas apilados. Por defecto 3; FUTBOLIN_HISTORY=N lo cambia para experimentos (hay que usar el
# mismo valor al entrenar, evaluar y ejecutar el modelo, porque cambia el tamaño de la entrada).
HISTORY   = max(int(os.environ.get("FUTBOLIN_HISTORY", "3")), 1)
FRAME_DIM = 6 + 16 + 16 + 8 + 8 + 8
ACTOR_DIM = FRAME_DIM * HISTORY
PRIV_DIM  = 6 + 32 + 8 + 8
OBS_DIM_V3 = ACTOR_DIM + PRIV_DIM


def _advance(x, v, dt, lim, bounce):
    """Adelanta x con velocidad v durante dt; en las bandas rebota (y) o se queda en el límite (x)."""
    x += v * dt
    if bounce:
        if x > lim:
            x, v = 2 * lim - x, -v
        elif x < -lim:
            x, v = -2 * lim - x, -v
    return min(max(x, -lim), lim), v


def _clip(x, lim=2.0):
    return lim if x > lim else (-lim if x < -lim else x)


class ObservationBuilder:
    """Construye la observación v3 de `team` a partir del entorno."""

    def __init__(self, env, team: str, history: int = HISTORY):
        self.env = env
        self.team = team
        self.s = team_direction(team)
        self.own = get_team_bar_indices(team)
        self.opp = get_team_bar_indices('B' if team == 'A' else 'A')
        self.history = history
        self.camera = Camera(env.spec.camera, team, noise=env.noise)
        self._frames = deque(maxlen=history)

    def reset(self, rng=None):
        self.camera.reset(self.env.state, rng=rng)
        self._frames.clear()

    def record(self):
        self.camera.record(self.env.state)

    def _frame(self, last_action) -> list:
        env, s = self.env, self.s
        state = env.state
        f = state.field
        hx, hy = f.length / 2.0, f.width / 2.0
        cam = self.camera.observe(state.time)
        bx, by, bvx, bvy = cam.ball
        lead = cam.lead(state.time)
        if lead > 0.0:
            bx, bvx = _advance(bx, bvx, lead, hx - BALL_RADIUS, bounce=False)
            by, bvy = _advance(by, bvy, lead, hy - BALL_RADIUS, bounce=True)
        out = [s * bx / hx, s * by / hy,
               _clip(s * bvx / BALL_V_SCALE), _clip(s * bvy / BALL_V_SCALE),
               1.0 if cam.visible else 0.0,
               min(cam.age(state.time) / AGE_SCALE, 1.0)]
        est = []
        for i in self.own:
            lin, ang, lv, av = env.actuators[i].estimate()
            est.append((i, lin))
            out.extend((s * lin / LINEAR_RANGE, s * _clip(ang / ANGULAR_RANGE),
                        _clip(s * lv / LIN_V_SCALE), _clip(s * av / ANG_V_SCALE)))
        for lin, ang, lv, av in cam.opponent_bars():
            if lead > 0.0:
                lin = min(max(lin + lv * lead, -LINEAR_RANGE), LINEAR_RANGE)
                ang = ang + av * lead
            out.extend((s * lin / LINEAR_RANGE, s * _clip(ang / ANGULAR_RANGE),
                        _clip(s * lv / LIN_V_SCALE), _clip(s * av / ANG_V_SCALE)))
        for lin, ang in cam.own_bars():
            out.extend((s * lin / LINEAR_RANGE, s * ang / ANGULAR_RANGE))
        for i, lin in est:
            bar = state.bars[i]
            rel_x = s * (bx - bar.bar_x) / OBS_REL_X_SCALE
            rel_y = min((by - (lin + off) for off in bar.player_offsets), key=abs)
            out.append(_clip(rel_x))
            out.append(_clip(s * rel_y / OBS_REL_Y_SCALE))
        out.extend(float(a) for a in last_action)
        return out

    def _privileged(self) -> list:
        env, s = self.env, self.s
        state = env.state
        b = state.ball
        f = state.field
        hx, hy = f.length / 2.0, f.width / 2.0
        out = [s * b.x / hx, s * b.y / hy,
               _clip(s * b.vx / BALL_V_SCALE), _clip(s * b.vy / BALL_V_SCALE),
               _clip(s * b.wx / BALL_V_SCALE), _clip(s * b.wy / BALL_V_SCALE)]
        for i in self.own + self.opp:
            bar = state.bars[i]
            out.extend((s * bar.linear_pos / LINEAR_RANGE, s * bar.angle / ANGULAR_RANGE,
                        _clip(s * bar.linear_vel / LIN_V_SCALE),
                        _clip(s * bar.angular_vel / ANG_V_SCALE)))
        for i in self.own:
            bar = state.bars[i]
            rel_y = min((b.y - (bar.linear_pos + off) for off in bar.player_offsets), key=abs)
            out.append(_clip(s * (b.x - bar.bar_x) / OBS_REL_X_SCALE))
            out.append(_clip(s * rel_y / OBS_REL_Y_SCALE))
        for i in self.own:
            bar = state.bars[i]
            lin, ang, _, _ = env.actuators[i].estimate()
            out.append(_clip(s * (bar.linear_pos - lin) / LINEAR_RANGE * 10.0))
            out.append(_clip(s * (bar.angle - ang) / ANGULAR_RANGE * 10.0))
        return out

    def build(self, last_action) -> np.ndarray:
        frame = self._frame(last_action)
        if not self._frames:
            for _ in range(self.history):
                self._frames.append(frame)
        else:
            self._frames.append(frame)
        actor = [x for fr in reversed(self._frames) for x in fr]   # más reciente primero
        return np.asarray(actor + self._privileged(), dtype=np.float32)
