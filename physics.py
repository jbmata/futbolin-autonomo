"""
physics.py
----------
Motor de física 2D realista para futbolín tipo Madrid.

Modela:
  - Movimiento de la bola con fricción rodante
  - Rebotes en paredes con restitución realista
  - Detección de gol (bola cruza la línea de portería)
  - Colisión bola-jugador con swept detection (evita tunneling)
  - Transferencia de impulso angular de la barra a la bola
  - Incertidumbre: ruido en rebotes y contactos
  - Predicción de trayectoria para agente heurístico
"""

import numpy as np
from world import (
    WorldState, BallState, BarState, FieldConfig,
    BALL_RADIUS, FOOT_LENGTH, PLAYER_HALF_WIDTH, PLAYER_DEPTH,
    KICK_ANGLE_MAX, MAX_BALL_SPEED,
)


# ---------------------------------------------------------------------------
# Constantes físicas
# ---------------------------------------------------------------------------

FRICTION_ROLLING   = 0.05    # coeficiente de fricción rodante [1/s]
RESTITUTION_WALL   = 0.65    # rebote en pared
RESTITUTION_PLAYER = 0.90    # rebote en jugador (kick zone activa)
RESTITUTION_BLOCK  = 0.50    # rebote pasivo (jugador fuera de kick zone)
PLAYER_IMPULSE_K   = 0.70    # fracción de velocidad del pie transferida a bola
BAR_LINVEL_TRANSFER = 0.65   # fracción de vel. lineal de barra → bola (en Y)

# Zona de contacto en X: más generosa que PLAYER_DEPTH para evitar tunneling
CONTACT_ZONE_X     = BALL_RADIUS + PLAYER_DEPTH * 2   # ≈ 0.034 m

NOISE_RESTITUTION  = 0.05
NOISE_WALL_ANGLE   = 0.03


class PhysicsEngine:
    """
    Motor de física del futbolín.

    Parámetros
    ----------
    dt : float
        Paso de integración [s]. 0.02 s → 50 Hz.
    noise : bool
        Añade variabilidad a rebotes y contactos (sim-to-real gap).
    seed : int, opcional
        Semilla del RNG.
    """

    def __init__(self, dt: float = 0.02, noise: bool = True, seed: int = None):
        self.dt    = dt
        self.noise = noise
        self.rng   = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # API principal
    # ------------------------------------------------------------------

    def step(self, state: WorldState) -> dict:
        """
        Avanza la simulación un paso dt.

        Devuelve dict con eventos:
          'goal'        : None | 'A' | 'B'
          'ball_out'    : bool
          'wall_bounce' : bool
          'touches'     : list de (bar_idx, player_idx)
        """
        events = {
            "goal":        None,
            "ball_out":    False,
            "wall_bounce": False,
            "touches":     [],
        }

        ball  = state.ball
        field = state.field

        # --- Guardar posición ANTES de integrar (para swept detection) ---
        prev_x = ball.x
        prev_y = ball.y

        # 1. Integrar posición
        ball.x += ball.vx * self.dt
        ball.y += ball.vy * self.dt

        # 2. Fricción rodante
        decay   = max(1.0 - FRICTION_ROLLING * self.dt, 0.0)
        ball.vx *= decay
        ball.vy *= decay

        # 3. Rebotes en paredes Y
        if self._resolve_wall_bounce(ball, field):
            events["wall_bounce"] = True

        # 4. Colisiones con jugadores (swept detection)
        for bar_idx, bar in enumerate(state.bars):
            touches = self._resolve_bar_collisions(
                ball, bar, field, bar_idx, prev_x
            )
            events["touches"].extend(touches)

        # 5. Detección de gol
        goal = self._check_goal(ball, field)
        if goal is not None:
            events["goal"]     = goal
            events["ball_out"] = True
        elif ball.x < field.x_min - 0.05 or ball.x > field.x_max + 0.05:
            events["ball_out"] = True

        # 6. Limitar velocidad máxima
        speed = np.hypot(ball.vx, ball.vy)
        if speed > MAX_BALL_SPEED:
            ball.vx = ball.vx / speed * MAX_BALL_SPEED
            ball.vy = ball.vy / speed * MAX_BALL_SPEED

        state.time += self.dt
        return events

    # ------------------------------------------------------------------
    # Rebote en paredes
    # ------------------------------------------------------------------

    def _resolve_wall_bounce(self, ball: BallState, field: FieldConfig) -> bool:
        rest = RESTITUTION_WALL
        if self.noise:
            rest = float(np.clip(
                rest + self.rng.uniform(-NOISE_RESTITUTION, NOISE_RESTITUTION),
                0.3, 0.9
            ))
        bounced = False

        if ball.y - BALL_RADIUS <= field.y_min:
            ball.y  = field.y_min + BALL_RADIUS
            ball.vy = abs(ball.vy) * rest
            if self.noise:
                ball.vx += self.rng.uniform(-NOISE_WALL_ANGLE, NOISE_WALL_ANGLE) * abs(ball.vy)
            bounced = True

        elif ball.y + BALL_RADIUS >= field.y_max:
            ball.y  = field.y_max - BALL_RADIUS
            ball.vy = -abs(ball.vy) * rest
            if self.noise:
                ball.vx += self.rng.uniform(-NOISE_WALL_ANGLE, NOISE_WALL_ANGLE) * abs(ball.vy)
            bounced = True

        return bounced

    # ------------------------------------------------------------------
    # Colisión bola–barra  (swept detection)
    # ------------------------------------------------------------------

    def _resolve_bar_collisions(
        self,
        ball:    BallState,
        bar:     BarState,
        field:   FieldConfig,
        bar_idx: int,
        prev_x:  float,
    ) -> list:
        """
        Detecta colisión usando swept detection:
        comprueba si la bola CRUZÓ la barra entre el paso anterior y el actual,
        no solo si está dentro de una zona fina.

        Esto evita el tunneling cuando la bola va rápida.
        """
        dx_prev = prev_x  - bar.bar_x   # distancia en X antes del paso
        dx_curr = ball.x  - bar.bar_x   # distancia en X después del paso

        # ¿La bola cruzó la barra este paso?
        crossed = (dx_prev * dx_curr < 0)
        # ¿O se está acercando a la zona de contacto (y moviéndose HACIA la barra)?
        toward_bar = (ball.vx * (bar.bar_x - ball.x)) > 0
        near       = (abs(dx_curr) <= CONTACT_ZONE_X) and toward_bar

        if not (crossed or near):
            return []

        # Dirección de aproximación (desde qué lado venía la bola)
        approach_sign = np.sign(dx_prev) if crossed else np.sign(dx_curr)

        # Posición Y estimada en el momento del cruce
        if crossed and abs(dx_prev - dx_curr) > 1e-9:
            t_cross  = abs(dx_prev) / abs(dx_prev - dx_curr)
            check_y  = prev_y_approx = ball.y - ball.vy * (1.0 - t_cross) * self.dt
        else:
            check_y  = ball.y

        # ¿Está el jugador en zona de golpeo?
        in_kick_zone = bar.is_in_kick_zone()

        # Comprobar alineación Y con cada jugador de la barra
        for j, player_y in enumerate(bar.get_player_abs_y()):
            if abs(check_y - player_y) <= BALL_RADIUS + PLAYER_HALF_WIDTH:
                if in_kick_zone:
                    self._apply_player_impulse(
                        ball, bar, approach_sign, ball.y - player_y
                    )
                else:
                    # Bloqueo pasivo: el jugador está girado pero bloquea físicamente
                    self._apply_passive_block(ball, bar, approach_sign)
                return [(bar_idx, j)]

        # Sin alineación con ningún jugador: la bola pasa libremente.
        # En el futbolín real la varilla no bloquea donde no hay figura.
        return []

    # ------------------------------------------------------------------
    # Impulso activo (jugador en kick zone)
    # ------------------------------------------------------------------

    def _apply_player_impulse(
        self,
        ball:          BallState,
        bar:           BarState,
        approach_sign: float,
        dy:            float,
    ):
        """
        Impulso pie-bola usando colisión 1D (pie como masa infinita).

        Fórmula estándar:
            v_bola_after = v_pie + restitution * (v_pie - v_bola)

        Esto garantiza que:
          - Con pie estático: la bola simplemente rebota y pierde energía.
          - Con pie en movimiento (golpeo): la bola recibe la velocidad del pie
            multiplicada por (1 + restitution), mucho más energético.

        approach_sign: +1 bola venía por la derecha, -1 por la izquierda.
        """
        rest = RESTITUTION_PLAYER
        if self.noise:
            rest = float(np.clip(
                rest + self.rng.uniform(-NOISE_RESTITUTION, NOISE_RESTITUTION),
                0.45, 0.90
            ))

        # Velocidad tangencial del pie en X
        vx_foot = bar.foot_tip_vel_x()

        # Colisión 1D: v_after = v_foot + rest * (v_foot - v_ball)
        ball.vx = vx_foot + rest * (vx_foot - ball.vx)

        # Garantizar que la bola sale en la dirección correcta (no vuelve a entrar)
        if ball.vx * approach_sign < 0:
            ball.vx = approach_sign * abs(ball.vx)

        # Velocidad mínima de salida (la colisión siempre mueve la bola)
        if abs(ball.vx) < 0.25:
            ball.vx = approach_sign * 0.25

        # Transferencia de velocidad lineal de la barra en Y
        ball.vy += BAR_LINVEL_TRANSFER * (bar.linear_vel - ball.vy)

        # Deflexión Y por ángulo del pie: si el pie está inclinado, la bola sale diagonal
        angle_deflect = np.sin(bar.angle) * abs(ball.vx) * 0.4
        ball.vy += angle_deflect

        # Separar la bola de la barra
        ball.x = bar.bar_x + approach_sign * (CONTACT_ZONE_X + 1e-4)

        # Separación en Y
        overlap_y = (BALL_RADIUS + PLAYER_HALF_WIDTH) - abs(dy)
        if overlap_y > 0:
            ball.y += np.sign(dy) * (overlap_y + 1e-4)

        # Ruido de contacto (imprecisión física)
        if self.noise:
            ball.vx += self.rng.uniform(-0.06, 0.06)
            ball.vy += self.rng.uniform(-0.06, 0.06)

    # ------------------------------------------------------------------
    # Bloqueo pasivo (jugador fuera de kick zone o zona vacía de barra)
    # ------------------------------------------------------------------

    def _apply_passive_block(
        self,
        ball:          BallState,
        bar:           BarState,
        approach_sign: float,
    ):
        """
        Rebote simple cuando el jugador está rotado (fuera de kick zone)
        o la bola cruza la zona de la barra sin alinearse con ningún jugador.
        La barra actúa como obstáculo físico con menor restitución.
        """
        rest    = RESTITUTION_BLOCK
        ball.vx = approach_sign * abs(ball.vx) * rest
        ball.x  = bar.bar_x + approach_sign * (CONTACT_ZONE_X + 1e-4)

    # ------------------------------------------------------------------
    # Detección de gol
    # ------------------------------------------------------------------

    def _check_goal(self, ball: BallState, field: FieldConfig):
        in_goal_y = field.goal_y_min <= ball.y <= field.goal_y_max

        if ball.x - BALL_RADIUS <= field.x_min and in_goal_y:
            return 'B'   # bola entró en portería A → gol de B

        if ball.x + BALL_RADIUS >= field.x_max and in_goal_y:
            return 'A'   # bola entró en portería B → gol de A

        return None

    # ------------------------------------------------------------------
    # Predicción de trayectoria (agente heurístico)
    # ------------------------------------------------------------------

    @staticmethod
    def predict_ball_y(ball: BallState, target_x: float, field: FieldConfig) -> float:
        """
        Predice la posición Y de la bola cuando llegue a target_x,
        incluyendo rebotes lineales en paredes Y.
        """
        if abs(ball.vx) < 1e-6:
            return float(np.clip(ball.y, field.y_min, field.y_max))

        t = (target_x - ball.x) / ball.vx
        if t < 0:
            return float(np.clip(ball.y, field.y_min, field.y_max))

        raw_y  = ball.y + ball.vy * t
        height = field.width
        y_min  = field.y_min
        y_max  = field.y_max

        raw_shifted = raw_y - y_min
        period      = 2.0 * height
        frac        = raw_shifted % period

        if frac <= height:
            predicted = y_min + frac
        else:
            predicted = y_max - (frac - height)

        return float(np.clip(predicted, y_min, y_max))
