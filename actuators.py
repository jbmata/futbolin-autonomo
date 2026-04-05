"""
actuators.py
------------
Modelo de actuadores para el futbolín tipo Madrid.

Cada barra tiene DOS grados de libertad independientes:
  1. LinearActuator  — desplazamiento lateral (eje Y), ±7 cm
  2. AngularActuator — rotación alrededor del eje de la barra, ±π rad

Ambos usan perfil trapezoidal de velocidad (vmax + amax), más:
  - Ruido de posición (encoder real tiene resolución finita)
  - Pequeño retardo de respuesta (latencia del driver)
  - Errores estocásticos ocasionales (slippage, fricción interna)

Control: posición objetivo normalizada en [-1, 1].
  - Linear:  cmd ∈ [-1, 1] → target ∈ [-LINEAR_RANGE, +LINEAR_RANGE]
  - Angular: cmd ∈ [-1, 1] → target ∈ [-π, +π]
"""

import numpy as np
from dataclasses import dataclass
from world import (
    LINEAR_RANGE, LINEAR_VMAX, LINEAR_AMAX,
    ANGULAR_OMEGA_MAX, ANGULAR_ALPHA_MAX,
)


# ---------------------------------------------------------------------------
# Ruido e incertidumbre
# ---------------------------------------------------------------------------

LINEAR_POS_NOISE  = 0.002   # m    (±2 mm, resolución de encoder)
ANGULAR_POS_NOISE = 0.02    # rad  (±1.1°)
SLIPPAGE_PROB     = 0.005   # probabilidad de pérdida de paso por step


# ---------------------------------------------------------------------------
# Sub-actuador genérico (trapezoidal)
# ---------------------------------------------------------------------------

class _TrapezoidalActuator:
    """
    Actuador con perfil de velocidad trapezoidal.
    Controla posición mediante limitación de velocidad y aceleración.
    """

    def __init__(
        self,
        pos_min: float,
        pos_max: float,
        vmax: float,
        amax: float,
        pos_noise: float,
        dt: float,
        add_noise: bool = True,
        rng: np.random.Generator = None,
    ):
        self.pos_min   = pos_min
        self.pos_max   = pos_max
        self.vmax      = vmax
        self.amax      = amax
        self.pos_noise = pos_noise
        self.dt        = dt
        self.add_noise = add_noise
        self.rng       = rng or np.random.default_rng()

        # Estado
        self.pos: float = 0.0
        self.vel: float = 0.0
        self.cmd: float = 0.0   # target position

    def reset(self, pos: float = 0.0):
        self.pos = float(np.clip(pos, self.pos_min, self.pos_max))
        self.vel = 0.0
        self.cmd = self.pos

    def set_target(self, target: float):
        self.cmd = float(np.clip(target, self.pos_min, self.pos_max))

    def update(self) -> tuple:
        """Actualiza un paso. Devuelve (pos, vel)."""
        tol   = 1e-4
        error = self.cmd - self.pos

        if abs(error) < tol:
            vel_desired = 0.0
        else:
            # Velocidad proporcional al error, saturada en vmax
            kp          = self.vmax / 0.05
            vel_desired = float(np.clip(kp * error, -self.vmax, self.vmax))

            # Perfil de frenado: ¿queda energía cinética suficiente para frenar?
            braking_dist = self.vel ** 2 / (2.0 * self.amax + 1e-9)
            if abs(error) < braking_dist:
                brake_vel   = np.sign(error) * np.sqrt(2.0 * self.amax * abs(error))
                vel_desired = float(np.clip(brake_vel, -self.vmax, self.vmax))

        # Limitar aceleración
        dv_max   = self.amax * self.dt
        dv       = float(np.clip(vel_desired - self.vel, -dv_max, dv_max))
        self.vel += dv

        # Integrar posición
        self.pos += self.vel * self.dt

        # Topes mecánicos
        if self.pos <= self.pos_min:
            self.pos = self.pos_min
            self.vel = max(self.vel, 0.0)
        elif self.pos >= self.pos_max:
            self.pos = self.pos_max
            self.vel = min(self.vel, 0.0)

        # Ruido de posición (resolución de encoder)
        if self.add_noise:
            noise = self.rng.uniform(-self.pos_noise, self.pos_noise)
            # El ruido afecta solo a la posición observada, no al estado interno
            observed_pos = float(np.clip(
                self.pos + noise, self.pos_min, self.pos_max
            ))
            # Slippage: pérdida de paso aleatoria
            if self.rng.random() < SLIPPAGE_PROB:
                slip = self.rng.uniform(-0.003, 0.003)
                self.pos = float(np.clip(self.pos + slip, self.pos_min, self.pos_max))
            return observed_pos, self.vel

        return self.pos, self.vel


# ---------------------------------------------------------------------------
# Actuador de barra (lineal + angular)
# ---------------------------------------------------------------------------

class BarActuator:
    """
    Actuador completo de una barra del futbolín.
    Encapsula los sub-actuadores lineal y angular.

    Parámetros
    ----------
    dt : float
        Paso de tiempo [s].
    add_noise : bool
        Activar incertidumbre en la simulación.
    seed : int, opcional
        Semilla del RNG.
    """

    def __init__(self, dt: float = 0.02, add_noise: bool = True, seed: int = None):
        self.dt        = dt
        self.add_noise = add_noise
        rng            = np.random.default_rng(seed)

        self._linear = _TrapezoidalActuator(
            pos_min   = -LINEAR_RANGE,
            pos_max   =  LINEAR_RANGE,
            vmax      = LINEAR_VMAX,
            amax      = LINEAR_AMAX,
            pos_noise = LINEAR_POS_NOISE,
            dt        = dt,
            add_noise = add_noise,
            rng       = rng,
        )
        self._angular = _TrapezoidalActuator(
            pos_min   = -np.pi,
            pos_max   =  np.pi,
            vmax      = ANGULAR_OMEGA_MAX,
            amax      = ANGULAR_ALPHA_MAX,
            pos_noise = ANGULAR_POS_NOISE,
            dt        = dt,
            add_noise = add_noise,
            rng       = rng,
        )

    def reset(self):
        self._linear.reset(0.0)
        self._angular.reset(0.0)

    def set_command(self, linear_cmd: float, angular_cmd: float):
        """
        Establece los objetivos normalizados.
          linear_cmd  ∈ [-1, 1] → target lineal  ∈ [-LINEAR_RANGE, +LINEAR_RANGE]
          angular_cmd ∈ [-1, 1] → target angular ∈ [-π, +π]
        """
        self._linear.set_target(linear_cmd  * LINEAR_RANGE)
        self._angular.set_target(angular_cmd * np.pi)

    def update(self) -> tuple:
        """
        Avanza un paso. Devuelve (linear_pos, angle, linear_vel, angular_vel).
        """
        lin_pos, lin_vel   = self._linear.update()
        ang_pos, ang_vel   = self._angular.update()
        return lin_pos, ang_pos, lin_vel, ang_vel

    @property
    def state(self) -> dict:
        return {
            "linear_pos":  self._linear.pos,
            "angle":       self._angular.pos,
            "linear_vel":  self._linear.vel,
            "angular_vel": self._angular.vel,
            "linear_cmd":  self._linear.cmd,
            "angular_cmd": self._angular.cmd,
        }


# ---------------------------------------------------------------------------
# Banco de actuadores (una instancia por barra)
# ---------------------------------------------------------------------------

class ActuatorBank:
    """Gestiona N actuadores de barra de forma centralizada."""

    def __init__(self, n_bars: int = 8, dt: float = 0.02,
                 add_noise: bool = True, seed: int = None):
        seeds = np.random.SeedSequence(seed).spawn(n_bars)
        self.actuators = [
            BarActuator(dt=dt, add_noise=add_noise, seed=int(s.generate_state(1)[0]))
            for s in seeds
        ]

    def reset(self):
        for act in self.actuators:
            act.reset()

    def set_commands(self, bar_idx: int, linear_cmd: float, angular_cmd: float):
        self.actuators[bar_idx].set_command(linear_cmd, angular_cmd)

    def set_all_commands(self, commands: np.ndarray):
        """
        Asigna comandos para todas las barras.
        commands: array de shape (n_bars, 2) → [linear_cmd, angular_cmd] por barra
        """
        for i, act in enumerate(self.actuators):
            act.set_command(float(commands[i, 0]), float(commands[i, 1]))

    def update(self) -> list:
        """Actualiza todos los actuadores. Devuelve lista de (pos, angle, vel, ang_vel)."""
        return [act.update() for act in self.actuators]

    def __getitem__(self, idx: int) -> BarActuator:
        return self.actuators[idx]

    def __len__(self) -> int:
        return len(self.actuators)
