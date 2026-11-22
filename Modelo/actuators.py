"""
actuators.py
------------
Actuadores de las barras.

Dos tipos:

StepperBarActuator — barra del robot, dos motores paso a paso (giro y lineal).
    Cada eje tiene dos estados:
      * estimado: el perfil trapezoidal que genera el firmware (pasos enviados).
        El firmware limita la aceleración según el par disponible a cada
        velocidad. Sin encoder es lo único que el robot conoce de su barra.
      * real: el rotor sigue al campo magnético como un muelle sinusoidal,
            T = −T(ω)·sin(N·(q − q_cmd)) − c·(ω − ω_cmd) − rozamiento,
        con la inercia real de la barra. Si la carga (un golpe de la bola, una
        bola pisada, un motor más débil de lo previsto) separa el rotor más de
        2 pasos completos, el motor "salta" y se pierden pasos: la posición
        real deja de coincidir con la estimada.

HumanBarActuator (alias BarActuator) — barra manejada por una persona, o
    actuador ideal: perfil trapezoidal cinemático con límites de mano/muñeca.
    Un golpe le cambia la velocidad y la mano la vuelve a llevar a su objetivo.

Control: posición objetivo normalizada en [-1, 1].
  - lineal:  cmd → objetivo ∈ [-LINEAR_RANGE, +LINEAR_RANGE]
  - angular: cmd → objetivo ∈ [-90°, +90°] (0 = vertical). El tope es de la orden: un
    golpe puede empujar la barra más allá.
"""

import math
import numpy as np
from world import (
    LINEAR_RANGE, LINEAR_VMAX, LINEAR_AMAX, ANGULAR_RANGE, angle_limits,
    ANGULAR_OMEGA_MAX, ANGULAR_ALPHA_MAX,
    bar_inertia, bar_mass,
)
from hardware import DEFAULT_SPEC, SimSpec, StepperSpec, RobotSpec, HumanSpec


# ---------------------------------------------------------------------------
# Ruido e incertidumbre
# ---------------------------------------------------------------------------

LINEAR_POS_NOISE  = 0.002   # m    (lectura para observaciones v1/v2)
ANGULAR_POS_NOISE = 0.02    # rad
SLIPPAGE_PROB     = 0.005   # pérdida de paso aleatoria (solo actuador ideal)
SLIPPAGE_LINEAR   = 0.003   # m
SLIPPAGE_ANGULAR  = 0.03    # rad
LIMIT_RANDOMIZATION = (0.85, 1.10)

def _stops(lo: float, hi: float) -> tuple:
    """
    Topes mecánicos del giro. Un recorrido recortado (el portero, -10° a +90°) es un tope
    real; en las demás barras los ±90° son solo lo que se ordena, y pueden girar libremente.
    """
    return (lo, hi) if (hi - lo) < 2.0 * ANGULAR_RANGE - 1e-6 else (-np.pi, np.pi)


def _cmd_to_angle(cmd: float, lo: float, hi: float) -> float:
    """Orden [-1, 1] -> ángulo: 0 es siempre vertical, +1 es el tope `hi` y -1 el tope `lo`."""
    cmd = min(max(cmd, -1.0), 1.0)
    return cmd * hi if cmd >= 0.0 else -cmd * lo


MOTOR_DT = 0.0011           # s, paso de integración del rotor
HOME_TOL = 0.02             # rad, el final de carrera se pulsa ~1° antes del tope
HOME_SEARCH = 1.2           # rad, recorrido máximo buscando el final de carrera


# ---------------------------------------------------------------------------
# Perfil trapezoidal (común)
# ---------------------------------------------------------------------------

def _profile_step(pos, vel, target, vmax, amax, h):
    """Un paso del perfil trapezoidal hacia `target`. Devuelve (pos, vel)."""
    error = target - pos
    if abs(error) < 1e-6 and abs(vel) < amax * h:
        return target, 0.0
    v_brake = math.sqrt(2.0 * amax * abs(error))
    v_prop = abs(error) / h
    vel_desired = math.copysign(min(vmax, v_brake, v_prop), error)
    dv = vel_desired - vel
    dv_max = amax * h
    if dv > dv_max:
        dv = dv_max
    elif dv < -dv_max:
        dv = -dv_max
    vel += dv
    return pos + vel * h, vel


class _TrapezoidalActuator:
    """Eje cinemático con perfil trapezoidal (mano humana o actuador ideal)."""

    def __init__(self, pos_min, pos_max, vmax, amax, pos_noise, dt,
                 add_noise=True, rng=None):
        self.pos_min, self.pos_max = pos_min, pos_max
        self.vmax0, self.amax0 = vmax, amax
        self.vmax, self.amax = vmax, amax
        self.pos_noise = pos_noise
        self.dt = dt
        self.add_noise = add_noise
        self.rng = rng or np.random.default_rng()
        self.pos = self.vel = self.cmd = 0.0

    # El estado estimado coincide con el real
    @property
    def est_pos(self): return self.pos

    @property
    def est_vel(self): return self.vel

    def reset(self, pos: float = 0.0):
        self.pos = min(max(float(pos), self.pos_min), self.pos_max)
        self.vel = 0.0
        self.cmd = self.pos

    def set_target(self, target: float):
        self.cmd = min(max(float(target), self.pos_min), self.pos_max)

    def set_true(self, pos: float, vel: float):
        self.pos, self.vel = pos, vel

    def update(self, h: float = None) -> tuple:
        h = self.dt if h is None else h
        self.pos, self.vel = _profile_step(self.pos, self.vel, self.cmd, self.vmax, self.amax, h)
        if self.pos <= self.pos_min:
            self.pos, self.vel = self.pos_min, max(self.vel, 0.0)
        elif self.pos >= self.pos_max:
            self.pos, self.vel = self.pos_max, min(self.vel, 0.0)
        return self.pos, self.vel

    def slip(self, amount: float):
        self.pos = min(max(self.pos + amount, self.pos_min), self.pos_max)

    def randomize_limits(self, enable: bool, scale: float = 1.0):
        lo, hi = LIMIT_RANDOMIZATION
        f1 = float(self.rng.uniform(lo, hi)) if enable else 1.0
        f2 = float(self.rng.uniform(lo, hi)) if enable else 1.0
        self.vmax, self.amax = self.vmax0 * f1 * scale, self.amax0 * f2 * scale


class _StepperAxis:
    """
    Eje accionado por un motor paso a paso.
    `k` convierte la coordenada de la barra (m o rad) en ángulo del eje del motor.
    """

    def __init__(self, pos_min, pos_max, k, load_inertia, friction, spec: RobotSpec,
                 rng=None, home: float = None, closed_loop: bool = False):
        self.closed_loop = closed_loop
        self.home = home        # posición del final de carrera (None = no hay)
        self.pos_min, self.pos_max = pos_min, pos_max
        self.k = k
        self.spec = spec
        self.stepper = spec.stepper
        self.rng = rng or np.random.default_rng()
        # Valores nominales (los que conoce el firmware)
        self.J_nom = load_inertia
        self.fric_nom = friction
        self.T0_nom = spec.stepper.holding_torque
        # Valores reales (aleatorizables)
        self.J, self.fric, self.T0 = self.J_nom, self.fric_nom, self.T0_nom
        self._update_constants()
        self.reset()

    def _update_constants(self):
        st = self.stepper
        self.N = st.teeth
        self.c = 2.0 * st.damping_ratio * math.sqrt(self.N * self.T0 * self.J)
        # Firmware: crucero y aceleración según el par nominal
        self.vmax = self.spec.firmware_vmax * st.max_speed / self.k
        self._a_scale = self.spec.firmware_safety / (self.J_nom * self.k)

    def randomize(self, enable: bool):
        if enable:
            r = self.rng
            self.J = self.J_nom * float(r.uniform(0.85, 1.15))
            self.fric = self.fric_nom * float(r.uniform(0.5, 2.0))
            self.T0 = self.T0_nom * float(r.uniform(0.8, 1.1))
        else:
            self.J, self.fric, self.T0 = self.J_nom, self.fric_nom, self.T0_nom
        self._update_constants()

    def reset(self, pos: float = 0.0):
        pos = min(max(float(pos), self.pos_min), self.pos_max)
        self.pos = self.est_pos = self.cmd = pos
        self.vel = self.est_vel = 0.0
        self._pressed = False
        self.lost_steps = 0
        self._slip_cycles = 0

    def set_target(self, target: float):
        self.cmd = min(max(float(target), self.pos_min), self.pos_max)

    def set_true(self, pos: float, vel: float):
        self.pos, self.vel = pos, vel

    def _torque(self, w_abs: float, T0: float) -> float:
        return self.stepper.torque(w_abs) * T0 / self.T0_nom

    def update(self, h: float) -> tuple:
        n = max(1, int(math.ceil(h / MOTOR_DT - 1e-9)))
        dt = h / n
        k, N, J, c, fric = self.k, self.N, self.J, self.c, self.fric
        target = self.cmd
        if self.home is not None and not self._pressed and abs(self.cmd - self.home) < 1e-6:
            # Buscar el final de carrera: seguir más allá hasta que se pulse
            target = self.home + math.copysign(HOME_SEARCH, self.home)
        for _ in range(n):
            # Firmware: perfil con aceleración limitada por el par a esa velocidad
            v = self.est_vel
            a_lim = self._a_scale * max(self._torque(abs(v) * k, self.T0_nom) - self.fric_nom, 0.0)
            a_lim = max(a_lim, 1e-3)
            self.est_pos, self.est_vel = _profile_step(self.est_pos, v, target, self.vmax, a_lim, dt)

            # Rotor real
            q = self.pos * k
            w = self.vel * k
            err = q - self.est_pos * k
            T = -self._torque(abs(w), self.T0) * math.sin(N * err) - c * (w - self.est_vel * k)
            if w > 1e-4:
                T -= fric
            elif w < -1e-4:
                T += fric
            elif abs(T) <= fric:
                T = 0.0
                w = 0.0
            else:
                T -= math.copysign(fric, T)
            w += T / J * dt
            q += w * dt
            pos, vel = q / k, w / k
            if pos <= self.pos_min:
                pos, vel = self.pos_min, max(vel, 0.0)
            elif pos >= self.pos_max:
                pos, vel = self.pos_max, min(vel, 0.0)
            self.pos, self.vel = pos, vel
        if self.closed_loop:
            self.end_of_step()          # el MCU cierra el lazo con el encoder en cada ciclo
        if self.home is not None:
            pressed = self.pos * math.copysign(1.0, self.home) >= abs(self.home) - HOME_TOL
            if pressed and not self._pressed:
                self._resync()
            self._pressed = pressed
        return self.pos, self.vel

    def _resync(self):
        """El firmware sabe dónde está (final de carrera): quita los ciclos eléctricos perdidos."""
        period = 2.0 * math.pi / self.N / self.k
        self.est_pos += round((self.pos - self.est_pos) / period) * period
        self._slip_cycles = 0

    def end_of_step(self):
        """Cuenta los pasos perdidos; con encoder el firmware corrige su estimación."""
        period = 2.0 * math.pi / self.N / self.k        # un ciclo eléctrico en unidades de barra
        cycles = round((self.pos - self.est_pos) / period)
        if cycles != self._slip_cycles:
            self.lost_steps += 4 * abs(cycles - self._slip_cycles)   # 1 ciclo = 4 pasos
            self._slip_cycles = cycles
        if cycles and self.closed_loop:
            self.est_pos += cycles * period
            self._slip_cycles = 0


# ---------------------------------------------------------------------------
# Actuadores de barra
# ---------------------------------------------------------------------------

class HumanBarActuator:
    """Barra manejada por una persona (o actuador ideal): perfil trapezoidal cinemático."""

    kind = "human"

    def __init__(self, dt: float = 0.02, add_noise: bool = True, seed: int = None,
                 randomize: bool = False, n_players: int = 1, human: HumanSpec = None,
                 ideal: bool = False, limits: tuple = None):
        self.angle_lo, self.angle_hi = limits or (-ANGULAR_RANGE, ANGULAR_RANGE)
        self.dt = dt
        self.add_noise = add_noise
        self.randomize = randomize
        self.rng = np.random.default_rng(seed)
        self.skill_scale = 1.0
        self.n_players = n_players
        if ideal or human is None:
            # Actuador ideal histórico (límites de world.py)
            lin = (LINEAR_VMAX, LINEAR_AMAX)
            rot = (ANGULAR_OMEGA_MAX, ANGULAR_ALPHA_MAX)
            factor = 1.0
            self.slippage = True
        else:
            lin = (human.lin_vmax, human.lin_amax)
            rot = (human.rot_vmax, human.rot_amax)
            factor = human.hand_inertia_factor
            self.slippage = False
        self.inertia = bar_inertia(n_players) * factor
        self.lin_mass = bar_mass(n_players) * factor
        self._linear = _TrapezoidalActuator(-LINEAR_RANGE, LINEAR_RANGE, *lin,
                                            LINEAR_POS_NOISE, dt, add_noise, self.rng)
        self._angular = _TrapezoidalActuator(*_stops(self.angle_lo, self.angle_hi), *rot,
                                             ANGULAR_POS_NOISE, dt, add_noise, self.rng)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self._linear.rng = self._angular.rng = self.rng
        self._linear.reset(0.0)
        self._angular.reset(0.0)
        self._linear.randomize_limits(self.randomize, self.skill_scale)
        self._angular.randomize_limits(self.randomize, self.skill_scale)

    def set_command(self, linear_cmd: float, angular_cmd: float):
        self._linear.set_target(linear_cmd * LINEAR_RANGE)
        self._angular.set_target(_cmd_to_angle(angular_cmd, self.angle_lo, self.angle_hi))

    def update(self, h: float = None) -> tuple:
        lin_pos, lin_vel = self._linear.update(h)
        ang_pos, ang_vel = self._angular.update(h)
        return lin_pos, ang_pos, lin_vel, ang_vel

    def estimate(self) -> tuple:
        """Lo que conoce el controlador: aquí coincide con el estado real."""
        return self._linear.pos, self._angular.pos, self._linear.vel, self._angular.vel

    def lost_steps(self) -> tuple:
        return 0, 0

    def end_of_step(self):
        if not (self.add_noise and self.slippage):
            return
        if self.rng.random() < SLIPPAGE_PROB:
            self._linear.slip(self.rng.uniform(-SLIPPAGE_LINEAR, SLIPPAGE_LINEAR))
        if self.rng.random() < SLIPPAGE_PROB:
            self._angular.slip(self.rng.uniform(-SLIPPAGE_ANGULAR, SLIPPAGE_ANGULAR))

    def encoder_noise(self) -> tuple:
        if not self.add_noise:
            return 0.0, 0.0
        return (self.rng.uniform(-LINEAR_POS_NOISE, LINEAR_POS_NOISE),
                self.rng.uniform(-ANGULAR_POS_NOISE, ANGULAR_POS_NOISE))

    @property
    def state(self) -> dict:
        return {"linear_pos": self._linear.pos, "angle": self._angular.pos,
                "linear_vel": self._linear.vel, "angular_vel": self._angular.vel,
                "linear_cmd": self._linear.cmd, "angular_cmd": self._angular.cmd}


# Nombre histórico: actuador ideal con los límites de world.py
class BarActuator(HumanBarActuator):
    def __init__(self, dt: float = 0.02, add_noise: bool = True, seed: int = None,
                 randomize: bool = False, n_players: int = 1, limits: tuple = None):
        super().__init__(dt, add_noise, seed, randomize, n_players, human=None, ideal=True,
                         limits=limits)


class StepperBarActuator:
    """Barra del robot: motor de giro directo (o con reducción) + motor lineal con correa."""

    kind = "stepper"

    def __init__(self, dt: float = 0.02, add_noise: bool = True, seed: int = None,
                 randomize: bool = False, n_players: int = 1, robot: RobotSpec = None,
                 home_sign: float = -1.0, limits: tuple = None):
        robot = robot or DEFAULT_SPEC.robot
        self.angle_lo, self.angle_hi = limits or (-ANGULAR_RANGE, ANGULAR_RANGE)
        self.dt = dt
        self.add_noise = add_noise
        self.randomize = randomize
        self.rng = np.random.default_rng(seed)
        self.robot = robot
        self.n_players = n_players
        st = robot.stepper
        g = robot.rot_gear
        J_rot = st.rotor_inertia * g * g + bar_inertia(n_players)           # en la barra
        m_lin = bar_mass(n_players) + robot.carriage_mass
        J_lin_shaft = st.rotor_inertia + m_lin * robot.pulley_radius ** 2   # en el motor
        # Ejes: el giro trabaja en rad de barra (k = reducción), el lineal en m (k = 1/r)
        home = self.angle_lo if home_sign < 0 else self.angle_hi    # final de carrera: pie atrás
        self._angular = _StepperAxis(*_stops(self.angle_lo, self.angle_hi), g, J_rot / (g * g),
                                     robot.rot_friction / g, robot, self.rng,
                                     home=home if robot.home_switch else None,
                                     closed_loop=robot.encoders)
        self._linear = _StepperAxis(-LINEAR_RANGE, LINEAR_RANGE, 1.0 / robot.pulley_radius,
                                    J_lin_shaft, robot.lin_friction * robot.pulley_radius,
                                    robot, self.rng)
        self._update_effective()

    def _update_effective(self):
        """Inercia y masa efectivas en la barra (para los choques con la bola)."""
        g = self._angular.k
        self.inertia = self._angular.J * g * g
        r = 1.0 / self._linear.k
        self.lin_mass = self._linear.J / (r * r)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self._linear.rng = self._angular.rng = self.rng
        for ax in (self._linear, self._angular):
            ax.randomize(self.randomize)
            ax.reset(0.0)
        self._update_effective()

    def set_command(self, linear_cmd: float, angular_cmd: float):
        self._linear.set_target(linear_cmd * LINEAR_RANGE)
        self._angular.set_target(_cmd_to_angle(angular_cmd, self.angle_lo, self.angle_hi))

    def update(self, h: float = None) -> tuple:
        h = self.dt if h is None else h
        lin_pos, lin_vel = self._linear.update(h)
        ang_pos, ang_vel = self._angular.update(h)
        return lin_pos, ang_pos, lin_vel, ang_vel

    def estimate(self) -> tuple:
        """Estado que conoce el firmware (pasos enviados), no el real."""
        return (self._linear.est_pos, self._angular.est_pos,
                self._linear.est_vel, self._angular.est_vel)

    def lost_steps(self) -> tuple:
        return self._linear.lost_steps, self._angular.lost_steps

    def end_of_step(self):
        self._linear.end_of_step()
        self._angular.end_of_step()

    def encoder_noise(self) -> tuple:
        if not self.add_noise:
            return 0.0, 0.0
        return (self.rng.uniform(-LINEAR_POS_NOISE, LINEAR_POS_NOISE),
                self.rng.uniform(-ANGULAR_POS_NOISE, ANGULAR_POS_NOISE))

    @property
    def state(self) -> dict:
        return {"linear_pos": self._linear.pos, "angle": self._angular.pos,
                "linear_vel": self._linear.vel, "angular_vel": self._angular.vel,
                "linear_cmd": self._linear.cmd, "angular_cmd": self._angular.cmd,
                "linear_est": self._linear.est_pos, "angle_est": self._angular.est_pos}


# ---------------------------------------------------------------------------
# Banco de actuadores
# ---------------------------------------------------------------------------

class ActuatorBank:
    """
    Una instancia por barra.

    kinds : lista con "stepper" | "human" | "ideal" por barra. Por defecto todas
            "ideal" (comportamiento histórico).
    """

    def __init__(self, n_bars: int = 8, dt: float = 0.02, add_noise: bool = True,
                 seed: int = None, randomize: bool = False, kinds=None,
                 n_players=None, spec: SimSpec = None, home_signs=None, limits=None):
        home_signs = home_signs or [-1.0] * n_bars
        limits = limits or [None] * n_bars
        spec = spec or DEFAULT_SPEC
        kinds = kinds or ["ideal"] * n_bars
        n_players = n_players or [1] * n_bars
        seeds = np.random.SeedSequence(seed).spawn(n_bars)
        self.actuators = []
        for kind, n, s, home_sign, lim in zip(kinds, n_players, seeds, home_signs, limits):
            child = int(s.generate_state(1)[0])
            if kind == "stepper":
                act = StepperBarActuator(dt, add_noise, child, randomize, n, spec.robot, home_sign, lim)
            elif kind == "human":
                act = HumanBarActuator(dt, add_noise, child, randomize, n, spec.human, limits=lim)
            else:
                act = BarActuator(dt, add_noise, child, randomize, n, lim)
            self.actuators.append(act)

    def reset(self, seed=None):
        seeds = (np.random.SeedSequence(seed).spawn(len(self)) if seed is not None
                 else [None] * len(self))
        for act, child in zip(self.actuators, seeds):
            act.reset(seed=None if child is None else int(child.generate_state(1)[0]))

    def set_commands(self, bar_idx: int, linear_cmd: float, angular_cmd: float):
        self.actuators[bar_idx].set_command(linear_cmd, angular_cmd)

    def set_all_commands(self, commands: np.ndarray):
        for i, act in enumerate(self.actuators):
            act.set_command(float(commands[i, 0]), float(commands[i, 1]))

    def update(self, h: float = None) -> list:
        return [act.update(h) for act in self.actuators]

    def apply_to(self, bars, h: float = None):
        """Avanza h segundos y escribe el estado real en las barras del mundo."""
        for act, bar in zip(self.actuators, bars):
            bar.linear_pos, bar.angle, bar.linear_vel, bar.angular_vel = act.update(h)

    def sync_from_bar(self, bar_idx: int, bar):
        """Tras un choque, la física cambió el estado real de la barra: copiarlo al actuador."""
        act = self.actuators[bar_idx]
        act._linear.set_true(bar.linear_pos, bar.linear_vel)
        act._angular.set_true(bar.angle, bar.angular_vel)

    def write_physical_properties(self, bars):
        for act, bar in zip(self.actuators, bars):
            bar.inertia, bar.lin_mass = act.inertia, act.lin_mass

    def end_of_step(self):
        for act in self.actuators:
            act.end_of_step()

    def encoder_noise(self) -> np.ndarray:
        return np.array([act.encoder_noise() for act in self.actuators])

    def __getitem__(self, idx: int):
        return self.actuators[idx]

    def __len__(self) -> int:
        return len(self.actuators)
