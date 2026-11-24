"""
physics.py
----------
Motor de física para futbolín tipo Madrid.

La bola rueda en el plano del campo (2D). Los jugadores se modelan en 3D como
figuras Madrid: una pierna desde el eje de la barra (altura ROD_HEIGHT) que
termina en un pie de TACO, un bloque macizo de suela plana. En el plano x-z la
figura es una unión de rectángulos redondeados (cabeza y torso, pierna y taco)
que gira con la barra; en Y se extiende ±PLAYER_HALF_WIDTH.
Esto reproduce lo que importa del juego real:

  - El taco solo toca la bola cuando está lo bastante bajo; con el jugador
    levantado (|θ| ≳ 70°) la bola pasa por debajo.
  - La posición del taco en X depende del ángulo (x = bar_x + L·sin θ).
  - Golpe con la cara plana (puntera o talón) y la suela pisando la bola.
  - El golpeo transmite la velocidad del punto de contacto (ω × r) con
    restitución y fricción de Coulomb; un pie quieto bloquea y amortigua.
  - El taco apoyado encima de la bola la "pisa" y la frena (control de bola).
  - Una bola parada se puede golpear.

Integración con subpasos (por defecto 6 por paso de control de 20 ms) para que
los golpes rápidos no atraviesen al jugador. El entorno avanza los actuadores
en cada subpaso a través de `bar_update`.

Masas reales: la bola (≈24 g) y la barra (inercia de varilla + figuras, más
el rotor del motor o la mano del jugador) intercambian impulso. Una barra ligera
retrocede al recibir la bola y transmite menos velocidad al golpear. La física
cambia la velocidad de la barra y avisa al entorno (`on_bar_change`) para que
el actuador lo note (un motor paso a paso puede perder pasos).

Bola con giro: rueda sin deslizar salvo tras un golpe o un rebote, cuando
desliza con rozamiento cinético hasta volver a rodar (pierde ~2/7 de la
velocidad tras un golpe seco). Rodadura con resistencia constante + viscosa
sobre un campo plano.

Otros fenómenos: rebotes en bandas y fondos con restitución, rampas en las
esquinas de los fondos (típicas de la mesa Madrid, devuelven la bola al juego)
y detección de gol en el cruce.

Con `randomize=True` los parámetros físicos se muestrean por episodio
(aleatorización de dominio para sim-to-real).
"""

import math
import numpy as np
from dataclasses import dataclass, replace
from world import (
    WorldState, BallState, FieldConfig,
    BALL_RADIUS, PLAYER_HALF_WIDTH, PLAYER_DEPTH, MAX_BALL_SPEED,
    FOOT_LENGTH, FOOT_HEIGHT, FOOT_HALF_LENGTH, FOOT_ROUNDING, ROD_HEIGHT,
    HEAD_LENGTH, BODY_HALF_DEPTH,
    LINEAR_RANGE, BALL_MASS, GRAVITY,
)


# ---------------------------------------------------------------------------
# Parámetros físicos
# ---------------------------------------------------------------------------

@dataclass
class PhysicsParams:
    # shortcut: valores estimados de una mesa estándar, calibrar con vídeo de la mesa real.
    rolling_decel:      float = 0.15   # m/s²  deceleración de rodadura (constante)
    viscous_drag:       float = 0.05   # 1/s   pérdida proporcional a la velocidad
    restitution_wall:   float = 0.65   # rebote en bandas y fondos
    restitution_player: float = 0.70   # rebote pie-bola (componente normal)
    friction_player:    float = 0.30   # Coulomb pie-bola (componente tangencial)
    pin_damping:        float = 0.80   # fracción de vel. relativa eliminada por subpaso al pisar
    corner_ramp_accel:  float = 0.6    # m/s²  rampa de esquina hacia el centro
    corner_ramp_depth:  float = 0.05   # m     profundidad de la rampa desde el fondo
    max_speed:          float = 8.0    # m/s   tope de seguridad de la bola
    ball_mass:          float = BALL_MASS
    slide_friction:     float = 0.25   # rozamiento cinético bola-campo (deslizando)
    wall_friction:      float = 0.20   # rozamiento bola-banda (frena el giro en el choque)
    finite_inertia:     bool  = True   # False: barras de masa infinita (modelo antiguo)
    spin:               bool  = True   # False: la bola siempre rueda (modelo antiguo)


DEFAULT_PARAMS = PhysicsParams()

# Rangos de aleatorización (multiplicativos) cuando randomize=True
_RANDOMIZE = {
    "rolling_decel":      (0.6, 1.5),
    "viscous_drag":       (0.5, 1.5),
    "restitution_wall":   (0.85, 1.15),
    "restitution_player": (0.85, 1.15),
    "friction_player":    (0.7, 1.3),
    "slide_friction":     (0.7, 1.3),
    "ball_mass":          (0.9, 1.1),
}

# Compatibilidad con nombres antiguos
FRICTION_ROLLING   = DEFAULT_PARAMS.viscous_drag
RESTITUTION_WALL   = DEFAULT_PARAMS.restitution_wall
RESTITUTION_PLAYER = DEFAULT_PARAMS.restitution_player

NOISE_RESTITUTION  = 0.05   # ruido por rebote
NOISE_WALL_ANGLE   = 0.03   # ruido tangencial en rebote de pared
NOISE_CONTACT      = 0.04   # m/s ruido de contacto con el pie

_CONTACT_R  = BALL_RADIUS + FOOT_ROUNDING                # distancia de contacto
_CONTACT_R2 = _CONTACT_R * _CONTACT_R
_BAR_REACH_X = math.hypot(FOOT_LENGTH, FOOT_HALF_LENGTH) + _CONTACT_R   # alcance en X
_BAR_REACH_Y = PLAYER_HALF_WIDTH + _CONTACT_R

# Figura = unión de rectángulos en el marco local (u: eje→pies, v: hacia delante)
_LEG_RECT  = (0.0, FOOT_LENGTH - FOOT_HEIGHT, -PLAYER_DEPTH, PLAYER_DEPTH)
_FOOT_RECT = (FOOT_LENGTH - FOOT_HEIGHT, FOOT_LENGTH, -FOOT_HALF_LENGTH, FOOT_HALF_LENGTH)
# Con la barra girada ~180° la cabeza baja hasta ROD_HEIGHT − HEAD_LENGTH y alcanza la bola
_HEAD_RECT = (-HEAD_LENGTH, 0.0, -BODY_HALF_DEPTH, BODY_HALF_DEPTH)


def _rect_closest(u: float, v: float, rect: tuple):
    """
    Punto (uc, vc) del rectángulo más cercano a (u, v) y vector (du, dv) desde
    él. Si el punto está dentro, devuelve la normal de salida por la cara más
    próxima y `depth` > 0 (cuánto hay que sacarlo); si está fuera, depth = 0.
    """
    u0, u1, v0, v1 = rect
    uc = u0 if u < u0 else (u1 if u > u1 else u)
    vc = v0 if v < v0 else (v1 if v > v1 else v)
    du, dv = u - uc, v - vc
    if du != 0.0 or dv != 0.0:
        return uc, vc, du, dv, 0.0
    faces = ((u - u0, -1.0, 0.0, u0, v), (u1 - u, 1.0, 0.0, u1, v),
             (v - v0, 0.0, -1.0, u, v0), (v1 - v, 0.0, 1.0, u, v1))
    depth, nu, nv, uc, vc = min(faces, key=lambda f: f[0])
    return uc, vc, nu, nv, depth


class PhysicsEngine:
    """
    Motor de física del futbolín.

    Parámetros
    ----------
    dt : float
        Paso de control [s]. 0.02 s → 50 Hz.
    noise : bool
        Añade variabilidad a rebotes y contactos.
    seed : int, opcional
        Semilla del RNG.
    substeps : int
        Subpasos de integración por paso de control.
    params : PhysicsParams, opcional
    randomize : bool
        Si True, `randomize_params()` muestrea parámetros nuevos (se llama en reset).
    """

    def __init__(self, dt: float = 0.02, noise: bool = True, seed: int = None,
                 substeps: int = 6, params: PhysicsParams = None,
                 randomize: bool = False):
        if substeps < 1:
            raise ValueError("substeps debe ser >= 1")
        self.dt        = dt
        self.noise     = noise
        self.substeps  = substeps
        self.rng       = np.random.default_rng(seed)
        self.base_params = params or DEFAULT_PARAMS
        self.params    = self.base_params
        self.randomize = randomize
        self._on_change = None

    # ------------------------------------------------------------------
    # Aleatorización de dominio
    # ------------------------------------------------------------------

    def randomize_params(self):
        if not self.randomize:
            self.params = self.base_params
            return
        values = {k: getattr(self.base_params, k) * float(self.rng.uniform(lo, hi))
                  for k, (lo, hi) in _RANDOMIZE.items()}
        values["restitution_wall"]   = min(values["restitution_wall"], 0.95)
        values["restitution_player"] = min(values["restitution_player"], 0.95)
        self.params = replace(self.base_params, **values)

    # ------------------------------------------------------------------
    # API principal
    # ------------------------------------------------------------------

    def step(self, state: WorldState, bar_update=None, on_bar_change=None) -> dict:
        """
        Avanza la simulación un paso de control (dt) en `substeps` subpasos.

        bar_update(h) : callable opcional que avanza las barras h segundos
                        (el entorno lo usa para mover los actuadores).
                        Sin él las barras quedan fijas con su velocidad actual.
        on_bar_change(bar_idx) : callable opcional; se llama cuando la física
                        cambia el estado de una barra (choque con la bola, taco
                        que pisa la bola, bola atrapada). El entorno copia ese
                        estado a su actuador.

        Devuelve dict con eventos:
          'goal'        : None | 'A' | 'B'
          'ball_out'    : bool
          'wall_bounce' : bool
          'touches'     : list de (bar_idx, player_idx) sin repetir
          'pinned'      : bool — algún taco pisa la bola
          'squeezed'    : bool — bola atrapada entre dos figuras
        """
        events = {"goal": None, "ball_out": False, "wall_bounce": False,
                  "touches": [], "pinned": False, "squeezed": False}
        self._on_change = on_bar_change
        touched = set()
        h = self.dt / self.substeps
        for _ in range(self.substeps):
            if bar_update is not None:
                bar_update(h)
            if self._substep(state, h, events, touched):
                break
        return events

    def _substep(self, state: WorldState, h: float, events: dict, touched: set) -> bool:
        """Un subpaso. Devuelve True si hubo gol."""
        p     = self.params
        ball  = state.ball
        field = state.field
        prev_x, prev_y = ball.x, ball.y

        # 1. Integrar posición
        ball.x += ball.vx * h
        ball.y += ball.vy * h

        # 2. Deslizamiento → rodadura, resistencia a la rodadura, rampas
        self._roll(ball, h)
        self._apply_corner_ramps(ball, field, h)

        # 3. Bandas laterales
        if self._resolve_wall_bounce(ball, field):
            events["wall_bounce"] = True

        # 4. Jugadores
        pushers = None            # (bar_idx, nx) de los contactos de este subpaso
        near = [i for i, bar in enumerate(state.bars)
                if -_BAR_REACH_X <= ball.x - bar.bar_x <= _BAR_REACH_X]
        if len(near) > 1:
            # Resolver primero la barra más cercana: el orden no depende del
            # índice de la barra, así A y B son simétricos.
            near.sort(key=lambda i: abs(ball.x - state.bars[i].bar_x))
        for bar_idx in near:
            bar = state.bars[bar_idx]
            dxb = ball.x - bar.bar_x
            for j, off in enumerate(bar.player_offsets):
                dy_raw = ball.y - (bar.linear_pos + off)
                if dy_raw > _BAR_REACH_Y or dy_raw < -_BAR_REACH_Y:
                    continue
                nx = self._resolve_player_contact(ball, bar, dxb, dy_raw, events, bar_idx)
                if nx is not None:
                    if pushers is None:
                        pushers = []
                    pushers.append((bar_idx, nx))
                    if (bar_idx, j) not in touched:
                        touched.add((bar_idx, j))
                        events["touches"].append((bar_idx, j))
                    dxb = ball.x - bar.bar_x
        if pushers is not None and len(pushers) > 1:
            self._resolve_squeeze(state, pushers, h, events)

        # 5. Gol / fondos
        goal = self._check_goal(ball, field, prev_x, prev_y)
        if goal is not None:
            events["goal"]     = goal
            events["ball_out"] = True
            state.time += h
            return True
        if self._resolve_end_wall_bounce(ball, field):
            events["wall_bounce"] = True

        # 6. Tope de velocidad
        speed = math.hypot(ball.vx, ball.vy)
        if speed > p.max_speed:
            k = p.max_speed / speed
            ball.vx *= k
            ball.vy *= k
            ball.wx *= k
            ball.wy *= k

        state.time += h
        return False

    # ------------------------------------------------------------------
    # Rodadura
    # ------------------------------------------------------------------

    def _roll(self, ball: BallState, h: float):
        p = self.params
        if p.spin:
            # Deslizamiento: el rozamiento frena v y acelera el giro hasta igualarlos
            sx, sy = ball.vx - ball.wx, ball.vy - ball.wy
            slip = math.hypot(sx, sy)
            if slip > 1e-6:
                a = p.slide_friction * GRAVITY * h
                if 3.5 * a >= slip:
                    # Llega a rodar en este subpaso: v = w = (5v + 2w) / 7
                    ball.vx -= sx * (2.0 / 7.0)
                    ball.vy -= sy * (2.0 / 7.0)
                    ball.wx, ball.wy = ball.vx, ball.vy
                else:
                    ux, uy = sx / slip, sy / slip
                    ball.vx -= a * ux
                    ball.vy -= a * uy
                    ball.wx += 2.5 * a * ux
                    ball.wy += 2.5 * a * uy
        # Resistencia a la rodadura (constante + viscosa)
        speed = math.hypot(ball.vx, ball.vy)
        if speed > 0.0:
            new_speed = max(speed * (1.0 - p.viscous_drag * h) - p.rolling_decel * h, 0.0)
            k = new_speed / speed
            ball.vx *= k
            ball.vy *= k
            ball.wx *= k
            ball.wy *= k
        if not p.spin:
            ball.wx, ball.wy = ball.vx, ball.vy

    # ------------------------------------------------------------------
    # Contacto bola–jugador
    # ------------------------------------------------------------------

    def _resolve_player_contact(self, ball: BallState, bar, dxb: float,
                                dy_raw: float, events: dict, bar_idx: int = -1):
        """
        Contacto entre la bola (esfera apoyada en el campo, centro a z=r) y una
        figura Madrid (pierna + taco, rectángulos redondeados en el plano x-z
        que giran con la barra, extruidos en Y). Devuelve la componente X de la
        normal de contacto, o None si no hay contacto.
        """
        p = self.params
        s_th = math.sin(bar.angle)
        c_th = math.cos(bar.angle)

        # Centro de la bola en el marco local de la figura.
        #   eje hacia los pies d = (sin θ, −cos θ); hacia delante f = (cos θ, sin θ)
        rz = BALL_RADIUS - ROD_HEIGHT
        u = dxb * s_th - rz * c_th
        v = dxb * c_th + rz * s_th

        best = None
        for rect in (_FOOT_RECT, _LEG_RECT, _HEAD_RECT):
            uc, vc, du, dv, depth = _rect_closest(u, v, rect)
            dist = -depth if depth > 0.0 else math.hypot(du, dv)
            if best is None or dist < best[0]:
                best = (dist, uc, vc, du, dv, depth)
        _, uc, vc, du, dv, depth = best

        # Cara del taco en Y
        if dy_raw > PLAYER_HALF_WIDTH:
            dy = dy_raw - PLAYER_HALF_WIDTH
        elif dy_raw < -PLAYER_HALF_WIDTH:
            dy = dy_raw + PLAYER_HALF_WIDTH
        else:
            dy = 0.0

        if depth > 0.0:
            # Centro de la bola dentro del bloque (solo con penetraciones grandes):
            # sacarla por la cara más cercana del plano x-z.
            nx = du * s_th + dv * c_th
            nz = -du * c_th + dv * s_th
            ny = 0.0
            penetration = _CONTACT_R + depth
        else:
            # Vector local → mundo: du·d + dv·f
            dx = du * s_th + dv * c_th
            dz = -du * c_th + dv * s_th
            d2 = dx * dx + dy * dy + dz * dz
            if d2 >= _CONTACT_R2:
                return None
            d = math.sqrt(d2)
            if d < 1e-9:
                nx, ny, nz = (1.0 if ball.vx < 0 else -1.0), 0.0, 0.0
            else:
                nx, ny, nz = dx / d, dy / d, dz / d
            penetration = _CONTACT_R - d

        # Desplazamiento del punto de contacto por radián de giro: dP/dθ = uc·f − vc·d
        gx = uc * c_th - vc * s_th
        gz = uc * s_th + vc * c_th
        # Velocidad del punto de contacto: ω·dP/dθ, más el desplazamiento lateral
        w = bar.angular_vel
        vfx = w * gx
        vfz = w * gz
        vfy = bar.linear_vel

        rvx = ball.vx - vfx
        rvy = ball.vy - vfy
        rvz = -vfz
        vn = rvx * nx + rvy * ny + rvz * nz

        if vn < 0.0:
            e = p.restitution_player
            if self.noise:
                e = min(max(e + self.rng.uniform(-NOISE_RESTITUTION, NOISE_RESTITUTION), 0.3), 0.95)
            # Inversas de masa a lo largo de una dirección: bola (solo se mueve en
            # horizontal, el campo absorbe lo vertical) + giro y lateral de la barra.
            inv_mb = 1.0 / p.ball_mass
            if p.finite_inertia:
                inv_J, inv_M = 1.0 / bar.inertia, 1.0 / bar.lin_mass
            else:
                inv_J = inv_M = 0.0
            l_n = gx * nx + gz * nz
            K_n = max(nx * nx + ny * ny, 0.3) * inv_mb + l_n * l_n * inv_J + ny * ny * inv_M
            jn = -(1.0 + e) * vn / K_n
            # Fricción de Coulomb en la dirección tangencial
            tx, ty, tz = rvx - vn * nx, rvy - vn * ny, rvz - vn * nz
            vt = math.sqrt(tx * tx + ty * ty + tz * tz)
            jt = 0.0
            if vt > 1e-9:
                tx, ty, tz = tx / vt, ty / vt, tz / vt
                l_t = gx * tx + gz * tz
                K_t = max(tx * tx + ty * ty, 0.3) * inv_mb + l_t * l_t * inv_J + ty * ty * inv_M
                jt = min(p.friction_player * jn, vt / K_t)
            else:
                tx = ty = tz = l_t = 0.0
            # Bola: solo cambian las componentes horizontales
            ball.vx += (jn * nx - jt * tx) * inv_mb
            ball.vy += (jn * ny - jt * ty) * inv_mb
            if self.noise:
                ball.vx += self.rng.uniform(-NOISE_CONTACT, NOISE_CONTACT)
                ball.vy += self.rng.uniform(-NOISE_CONTACT, NOISE_CONTACT)
            # Reacción sobre la barra (acción y reacción)
            if inv_J or inv_M:
                bar.angular_vel -= (jn * l_n - jt * l_t) * inv_J
                bar.linear_vel  -= (jn * ny - jt * ty) * inv_M
                if self._on_change is not None and bar_idx >= 0:
                    self._on_change(bar_idx)

        # Suela del taco apoyada encima de la bola (normal hacia abajo): la pisa.
        # La bola no puede hundirse en el campo, así que el taco no la atraviesa:
        # la barra se frena en el ángulo de contacto (el motor se queda clavado).
        if nz < -0.6:
            events["pinned"] = True
            k = p.pin_damping
            ball.vx -= k * (ball.vx - vfx)
            ball.vy -= k * (ball.vy - vfy)
            ball.wx, ball.wy = ball.vx, ball.vy
            g = gx * nx + gz * nz          # > 0: girar en +θ hunde más el taco
            if abs(g) > 1e-4:
                dtheta = max(-0.3, min(0.3, -penetration / g))
                bar.angle += dtheta
                bar.angular_vel = 0.0
                if self._on_change is not None and bar_idx >= 0:
                    self._on_change(bar_idx)
            return nx

        # Separación: la bola solo puede moverse en horizontal
        nxy = math.hypot(nx, ny)
        if nxy > 0.25:
            push = min(penetration / nxy + 1e-4, 0.03)
            ball.x += nx / nxy * push
            ball.y += ny / nxy * push
        return nx

    def _resolve_squeeze(self, state: WorldState, pushers: list, h: float, events: dict):
        """
        Bola atrapada entre dos figuras por lados opuestos (en X). No puede
        escapar: se queda quieta en X y las barras que la empujan se frenan
        (el motor se queda bloqueado), en vez de hacerla vibrar.
        """
        if not (any(nx > 0.3 for _, nx in pushers) and any(nx < -0.3 for _, nx in pushers)):
            return
        events["squeezed"] = True
        ball = state.ball
        ball.vx = 0.0
        ball.vy *= 0.5
        ball.wx, ball.wy = ball.vx, ball.vy
        for bar_idx, _ in pushers:
            bar = state.bars[bar_idx]
            if bar.angular_vel != 0.0:
                bar.angle -= bar.angular_vel * h      # deshacer el giro de este subpaso
                bar.angular_vel = 0.0
                if self._on_change is not None:
                    self._on_change(bar_idx)

    # ------------------------------------------------------------------
    # Rampas de esquina (mesa Madrid)
    # ------------------------------------------------------------------

    def _apply_corner_ramps(self, ball: BallState, field: FieldConfig, h: float):
        """
        Junto a los fondos y fuera de la boca de gol el campo está inclinado:
        la bola rueda hacia el centro en Y y hacia dentro del campo en X.
        """
        p = self.params
        if p.corner_ramp_accel <= 0.0 or abs(ball.y) <= field.goal_y_max:
            return
        depth = p.corner_ramp_depth
        if ball.x > field.x_max - depth:
            sx = -1.0
        elif ball.x < field.x_min + depth:
            sx = 1.0
        else:
            return
        a = p.corner_ramp_accel * h
        ball.vx += sx * a * 0.5
        ball.vy -= math.copysign(a, ball.y)

    # ------------------------------------------------------------------
    # Rebote en paredes
    # ------------------------------------------------------------------

    def _restitution_wall(self) -> float:
        rest = self.params.restitution_wall
        if self.noise:
            rest = min(max(rest + self.rng.uniform(-NOISE_RESTITUTION, NOISE_RESTITUTION), 0.3), 0.95)
        return rest

    def _wall_spin(self, w_n: float, v_n: float, e: float) -> float:
        """
        El giro no se invierte al rebotar, pero el rozamiento con la banda (en el
        ecuador de la bola) lo frena durante el choque: Δw = 2.5·μ·(1+e)·|v_n|.
        """
        if not self.params.spin:
            return -e * v_n
        dw = 2.5 * self.params.wall_friction * (1.0 + e) * abs(v_n)
        return w_n - math.copysign(min(abs(w_n), dw), w_n)

    def _resolve_wall_bounce(self, ball: BallState, field: FieldConfig) -> bool:
        if ball.y - BALL_RADIUS <= field.y_min:
            ball.y = field.y_min + BALL_RADIUS
            sign = 1.0
        elif ball.y + BALL_RADIUS >= field.y_max:
            ball.y = field.y_max - BALL_RADIUS
            sign = -1.0
        else:
            return False
        if ball.vy * sign < 0.0:   # solo si se acerca a la banda
            e = self._restitution_wall()
            ball.wy = self._wall_spin(ball.wy, ball.vy, e)
            ball.vy = sign * abs(ball.vy) * e
            if self.noise:
                ball.vx += self.rng.uniform(-NOISE_WALL_ANGLE, NOISE_WALL_ANGLE) * abs(ball.vy)
        return True

    def _resolve_end_wall_bounce(self, ball: BallState, field: FieldConfig) -> bool:
        if ball.x - BALL_RADIUS <= field.x_min:
            ball.x = field.x_min + BALL_RADIUS
            direction = 1.0
        elif ball.x + BALL_RADIUS >= field.x_max:
            ball.x = field.x_max - BALL_RADIUS
            direction = -1.0
        else:
            return False
        if ball.vx * direction < 0.0:
            e = self._restitution_wall()
            ball.wx = self._wall_spin(ball.wx, ball.vx, e)
            ball.vx = direction * abs(ball.vx) * e
            if self.noise:
                ball.vy += self.rng.uniform(-NOISE_WALL_ANGLE, NOISE_WALL_ANGLE) * abs(ball.vx)
        return True

    # ------------------------------------------------------------------
    # Detección de gol
    # ------------------------------------------------------------------

    def _check_goal(self, ball: BallState, field: FieldConfig,
                    prev_x: float, prev_y: float):
        for team, line_x, direction in (
            ('B', field.x_min + BALL_RADIUS, -1.0),
            ('A', field.x_max - BALL_RADIUS, 1.0),
        ):
            if direction * (ball.x - line_x) < 0:
                continue
            dx = ball.x - prev_x
            fraction = min(max((line_x - prev_x) / dx, 0.0), 1.0) if abs(dx) > 1e-9 else 1.0
            crossing_y = prev_y + fraction * (ball.y - prev_y)
            # La bola debe caber entre los postes en el instante del cruce.
            if field.goal_y_min + BALL_RADIUS <= crossing_y <= field.goal_y_max - BALL_RADIUS:
                return team
        return None

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    @staticmethod
    def predict_ball_y(ball: BallState, target_x: float, field: FieldConfig) -> float:
        """
        Predice la posición Y de la bola cuando llegue a target_x,
        incluyendo rebotes lineales en las bandas.
        """
        y_min = field.y_min + BALL_RADIUS
        y_max = field.y_max - BALL_RADIUS
        if abs(ball.vx) < 1e-6:
            return min(max(ball.y, y_min), y_max)
        t = (target_x - ball.x) / ball.vx
        if t < 0:
            return min(max(ball.y, y_min), y_max)

        height = y_max - y_min
        frac = (ball.y + ball.vy * t - y_min) % (2.0 * height)
        predicted = y_min + frac if frac <= height else y_max - (frac - height)
        return min(max(predicted, y_min), y_max)

    @staticmethod
    def shot_on_target(ball: BallState, field: FieldConfig, team: str,
                       min_speed: float = 0.8) -> bool:
        """True si la bola va directa (sin rebotar) a la portería rival de `team`."""
        direction = 1.0 if team == 'A' else -1.0
        if direction * ball.vx < min_speed * 0.5 or ball.speed < min_speed:
            return False
        line_x = field.x_max if team == 'A' else field.x_min
        t = (line_x - ball.x) / ball.vx
        y = ball.y + ball.vy * t
        return field.goal_y_min + BALL_RADIUS <= y <= field.goal_y_max - BALL_RADIUS

    @staticmethod
    def ball_reachable(state: WorldState, margin: float = 0.0) -> bool:
        """True si algún jugador puede alcanzar la bola moviendo su barra."""
        ball = state.ball
        for bar in state.bars:
            if abs(ball.x - bar.bar_x) > _BAR_REACH_X + margin:
                continue
            for off in bar.player_offsets:
                if abs(ball.y - off) <= bar.linear_range + _BAR_REACH_Y + margin:
                    return True
        return False
