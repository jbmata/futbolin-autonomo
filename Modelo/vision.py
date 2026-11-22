"""
vision.py
---------
Modelo de la cámara cenital y del seguimiento de la bola.

El robot no conoce el estado real: lo ve con una cámara que
  - toma fotogramas a `fps` con una fase aleatoria,
  - entrega cada detección `latency` segundos después de la exposición,
  - mide con ruido, mayor cuanto más rápida va la bola (se emborrona durante
    la exposición), y a veces falla; si el emborronado es grande, falla más,
  - no ve la bola cuando está debajo de una figura (vista desde arriba, la
    figura y su taco la tapan),
  - estima la velocidad con un filtro α-β, que va retrasado tras un golpe.

Las barras también se miden con la cámara (posición y ángulo con ruido y con
la misma latencia): las rivales, porque no hay otra forma de saberlo, y las
propias, porque si el motor pierde pasos el firmware deja de saber dónde están.
"""

import math
from collections import deque
import numpy as np

from hardware import CameraSpec
from world import (BALL_RADIUS, PLAYER_HALF_WIDTH, FOOT_LENGTH, FOOT_HALF_LENGTH,
                   get_team_bar_indices)

BODY_HALF_X = 0.012       # m, semigrosor del cuerpo de la figura visto desde arriba
_HIDE_MARGIN = 0.004      # m, la bola está oculta si su centro queda tapado con este margen
_REACH_X = FOOT_LENGTH + FOOT_HALF_LENGTH


def ball_occluded(state) -> bool:
    """True si, vista desde arriba, el centro de la bola queda tapado por una figura."""
    ball = state.ball
    for bar in state.bars:
        dx = ball.x - bar.bar_x
        if dx > _REACH_X or dx < -_REACH_X:
            continue
        x0, x1 = bar.figure_footprint_x()
        x0 += _HIDE_MARGIN
        x1 -= _HIDE_MARGIN
        if not (x0 <= ball.x <= x1):
            continue
        for off in bar.player_offsets:
            if abs(ball.y - (bar.linear_pos + off)) < PLAYER_HALF_WIDTH - _HIDE_MARGIN:
                return True
    return False


class _AlphaBeta:
    """Filtro α-β de posición y velocidad en una coordenada."""

    __slots__ = ("x", "v", "alpha", "beta", "ok")

    def __init__(self, alpha, beta):
        self.alpha, self.beta = alpha, beta
        self.x = self.v = 0.0
        self.ok = False

    def reset(self, x=0.0, v=0.0):
        self.x, self.v, self.ok = x, v, True

    def predict(self, dt):
        self.x += self.v * dt

    def correct(self, z, dt):
        r = z - self.x
        self.x += self.alpha * r
        self.v += self.beta * r / dt


class Camera:
    """
    Cámara + seguimiento para un equipo observador.
    El entorno llama a `record(state)` en cada subpaso de física y a
    `observe(t)` en cada paso de control.
    """

    def __init__(self, spec: CameraSpec, team: str, noise: bool = True, rng=None):
        self.spec = spec
        self.team = team
        self.noise = noise
        self.rng = rng or np.random.default_rng()
        self.frame_dt = 1.0 / spec.fps
        self.opp_idxs = get_team_bar_indices('B' if team == 'A' else 'A')
        self.own_idxs = get_team_bar_indices(team)
        self.bar_idxs = self.own_idxs + self.opp_idxs
        self._hist = deque(maxlen=48)
        self.latency = spec.latency
        a, b = spec.tracker_alpha, spec.tracker_beta
        self._bx, self._by = _AlphaBeta(a, b), _AlphaBeta(a, b)
        self._bars = [(_AlphaBeta(a, b), _AlphaBeta(a, b)) for _ in self.bar_idxs]

    def reset(self, state, rng=None):
        if rng is not None:
            self.rng = rng
        self._hist.clear()
        # Latencia algo distinta en cada partida (carga del PC, exposición...)
        self.latency = self.spec.latency * (float(self.rng.uniform(0.8, 1.3)) if self.noise else 1.0)
        self._phase = float(self.rng.uniform(0.0, self.frame_dt))
        self._next_frame = self._phase
        self._last_seen_t = state.time
        self._frame_t = state.time
        self._visible = True
        b = state.ball
        self._bx.reset(b.x, b.vx)
        self._by.reset(b.y, b.vy)
        for (fl, fa), i in zip(self._bars, self.bar_idxs):
            bar = state.bars[i]
            fl.reset(bar.linear_pos, 0.0)
            fa.reset(bar.angle, 0.0)
        self.record(state)

    def record(self, state):
        bars = state.bars
        self._hist.append((
            state.time, state.ball.x, state.ball.y,
            self.spec.occlusion and ball_occluded(state),
            state.ball.speed,
            tuple((bars[i].linear_pos, bars[i].angle) for i in self.bar_idxs),
        ))

    def _sample(self, t):
        """Último estado registrado en o antes de t."""
        for rec in reversed(self._hist):
            if rec[0] <= t + 1e-9:
                return rec
        return self._hist[0]

    def observe(self, t_now: float):
        """Procesa los fotogramas disponibles en t_now. Devuelve el estado estimado."""
        spec = self.spec
        lat = self.latency
        rng = self.rng
        while self._next_frame + lat <= t_now + 1e-9:
            tf = self._next_frame
            self._next_frame += self.frame_dt
            if self._hist and tf < self._hist[0][0]:
                continue
            rec = self._sample(tf)
            dt = self.frame_dt
            self._bx.predict(dt)
            self._by.predict(dt)
            blur = rec[4] * spec.exposure            # m que recorre la bola durante la exposición
            p_miss = spec.miss_prob + min(max((blur - 0.02) / 0.03, 0.0), 0.5)
            seen = (not rec[3]) and not (self.noise and rng.random() < p_miss)
            if seen:
                zx, zy = rec[1], rec[2]
                if self.noise:
                    sigma = math.hypot(spec.ball_noise, 0.3 * blur)
                    zx += rng.normal(0.0, sigma)
                    zy += rng.normal(0.0, sigma)
                jump = math.hypot(zx - self._bx.x, zy - self._by.x)
                if tf - self._last_seen_t > 0.15 or jump > 0.12:
                    # Reaparece tras mucho tiempo oculta, o lejos de lo previsto
                    # (bola recolocada para un saque): reiniciar el seguimiento
                    self._bx.reset(zx, 0.0)
                    self._by.reset(zy, 0.0)
                else:
                    self._bx.correct(zx, dt)
                    self._by.correct(zy, dt)
                self._last_seen_t = tf
            else:
                # Oculta: se predice con velocidad que se va amortiguando
                self._bx.v *= 0.97
                self._by.v *= 0.97
            self._visible = seen
            for (fl, fa), (lin, ang) in zip(self._bars, rec[5]):
                fl.predict(dt)
                fa.predict(dt)
                if self.noise:
                    lin += rng.normal(0.0, spec.bar_lin_noise)
                    ang += rng.normal(0.0, spec.bar_ang_noise)
                fl.correct(lin, dt)
                fa.correct(ang, dt)
            self._frame_t = tf
        return self

    # Resultados (marco del mundo)
    @property
    def ball(self):
        return self._bx.x, self._by.x, self._bx.v, self._by.v

    @property
    def visible(self) -> bool:
        return self._visible

    def age(self, t_now: float) -> float:
        """Tiempo desde la última vez que se vio la bola."""
        return t_now - self._last_seen_t

    def opponent_bars(self):
        """[(lin, ang, lin_vel, ang_vel)] de las barras rivales, en el orden de opp_idxs."""
        n = len(self.own_idxs)
        return [(fl.x, fa.x, fl.v, fa.v) for fl, fa in self._bars[n:]]

    def own_bars(self):
        """[(lin, ang)] de las barras propias vistas por la cámara, en el orden de own_idxs."""
        n = len(self.own_idxs)
        return [(fl.x, fa.x) for fl, fa in self._bars[:n]]
