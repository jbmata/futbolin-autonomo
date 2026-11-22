"""
hardware.py
-----------
Especificaciones del sistema real que el simulador reproduce.

El equipo del robot mueve cada barra con dos motores paso a paso (NEMA17 +
TMC2209, como en Hardware/MCU-RT): uno gira la barra y otro la desplaza con
una correa GT2. El equipo rival lo maneja una persona. Una cámara cenital ve la
bola y las barras rivales.

Todos los valores son estimaciones (`shortcut:`) que hay que sustituir por
medidas reales: par del motor a la tensión usada, polea, masas de las figuras,
latencia y fps de la cámara. `describe()` imprime el resumen de capacidades que
se deducen de ellos.
"""

import math
from dataclasses import dataclass, field

from world import FOOT_LENGTH, bar_inertia, bar_mass


# ---------------------------------------------------------------------------
# Motor paso a paso
# ---------------------------------------------------------------------------

@dataclass
class StepperSpec:
    # shortcut: NEMA17 tipo 17HS4401 a 24 V con TMC2209 ~1.5 A RMS. Medir la curva real.
    holding_torque: float = 0.40     # N·m
    corner_speed:   float = 19.0     # rad/s (≈3 rev/s): a partir de aquí el par cae ~1/ω
    max_speed:      float = 94.0     # rad/s (≈15 rev/s): par nulo
    rotor_inertia:  float = 5.7e-6   # kg·m²
    teeth:          int   = 50       # dientes del rotor (motor de 1.8°)
    damping_ratio:  float = 0.25     # amortiguamiento eléctrico del driver

    def torque(self, speed: float) -> float:
        """Par disponible [N·m] a la velocidad `speed` [rad/s] del eje del motor."""
        w = abs(speed)
        if w >= self.max_speed:
            return 0.0
        t = self.holding_torque if w <= self.corner_speed else self.holding_torque * self.corner_speed / w
        return t * (1.0 - w / self.max_speed)


@dataclass
class RobotSpec:
    """Mecánica y firmware de las barras del robot."""
    stepper:          StepperSpec = field(default_factory=StepperSpec)
    # shortcut: polea GT2 de 30 dientes (60 mm por vuelta). Medir la real.
    pulley_radius:    float = 0.00955   # m
    rot_gear:         float = 1.0       # reducción motor:barra en el giro (1 = directo)
    carriage_mass:    float = 0.45      # kg que se desplazan con la barra (motor de giro + carro)
    rot_friction:     float = 0.010     # N·m  rozamiento de giro (cojinetes)
    lin_friction:     float = 4.0       # N    rozamiento lineal (casquillos)
    firmware_safety:  float = 0.5       # fracción del par que el firmware usa para planificar
    firmware_vmax:    float = 0.5       # fracción de max_speed como velocidad de crucero
    command_latency:  float = 0.010     # s  PC → placa → driver
    encoders:         bool  = False     # sin encoder la posición es la estimada (lazo abierto)


@dataclass
class HumanSpec:
    """Límites de una persona manejando las barras."""
    # shortcut: valores típicos de un jugador aficionado.
    lin_vmax:  float = 1.5      # m/s
    lin_amax:  float = 25.0     # m/s²
    rot_vmax:  float = 45.0     # rad/s
    rot_amax:  float = 1500.0   # rad/s²
    hand_inertia_factor: float = 3.0   # la mano y el brazo añaden inercia a la barra


@dataclass
class CameraSpec:
    """
    Por defecto, una Logitech Brio en 720p a 90 fps (a 1280 px sobre 1.16 m,
    ~0.9 mm por píxel). Obturador rolling y MJPEG por USB.
    """
    # shortcut: la latencia extremo a extremo de la Brio no está publicada; medirla
    # (p. ej. filmando un LED que se enciende desde el PC) y sustituir este valor.
    fps:            float = 90.0
    latency:        float = 0.070   # s desde la exposición hasta tener la detección
    exposure:       float = 0.004   # s, con buena iluminación (emborrona la bola rápida)
    ball_noise:     float = 0.0015  # m (desviación típica por eje, bola quieta)
    miss_prob:      float = 0.01    # detecciones fallidas aleatorias
    bar_lin_noise:  float = 0.002   # m, posición de las barras rivales
    bar_ang_noise:  float = 0.08    # rad, ángulo de las barras rivales (difícil desde arriba)
    tracker_alpha:  float = 0.6     # filtro α-β de la bola
    tracker_beta:   float = 0.25


@dataclass
class SimSpec:
    robot:  RobotSpec  = field(default_factory=RobotSpec)
    human:  HumanSpec  = field(default_factory=HumanSpec)
    camera: CameraSpec = field(default_factory=CameraSpec)


DEFAULT_SPEC = SimSpec()


# ---------------------------------------------------------------------------
# Capacidades deducidas
# ---------------------------------------------------------------------------

def rot_load_inertia(n_players: int, robot: RobotSpec) -> float:
    """Inercia en el eje del motor de giro [kg·m²]."""
    g = robot.rot_gear
    return robot.stepper.rotor_inertia + bar_inertia(n_players) / (g * g)


def lin_load_inertia(n_players: int, robot: RobotSpec) -> float:
    """Inercia equivalente en el eje del motor lineal [kg·m²]."""
    m = bar_mass(n_players) + robot.carriage_mass
    return robot.stepper.rotor_inertia + m * robot.pulley_radius ** 2


def describe(spec: SimSpec = DEFAULT_SPEC):
    """Imprime qué puede hacer cada barra del robot con esta especificación."""
    r, st = spec.robot, spec.robot.stepper
    print("Barra  figuras  a_giro0[rad/s²]  w_giro[rad/s]  v_punta[m/s]  a_lin0[m/s²]  v_lin[m/s]")
    for n, name in ((1, "GK"), (2, "DEF"), (5, "MID"), (3, "ATK")):
        j = rot_load_inertia(n, r)
        a_rot = r.firmware_safety * (st.torque(0) - r.rot_friction / r.rot_gear) / j * 1 / r.rot_gear
        w_rot = r.firmware_vmax * st.max_speed / r.rot_gear
        jl = lin_load_inertia(n, r)
        a_lin = r.firmware_safety * (st.torque(0) - r.lin_friction * r.pulley_radius) / jl * r.pulley_radius
        v_lin = r.firmware_vmax * st.max_speed * r.pulley_radius
        print(f"{name:5s}  {n:7d}  {a_rot:15.0f}  {w_rot:13.1f}  {w_rot * FOOT_LENGTH:12.2f}  "
              f"{a_lin:12.1f}  {v_lin:10.2f}")


if __name__ == "__main__":
    describe()
