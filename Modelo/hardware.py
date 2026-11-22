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
    """
    NEMA17 17HS4401 con TMC2209. Ficha: 1,7 A, 1,5 Ω, 40 N·cm (datasheetcafe, 17HS4401);
    2,8 mH y 54 g·cm² (variante 17HS4401S). El par a cada velocidad se calcula con el
    modelo eléctrico del motor (límite de corriente y de tensión), no de una curva medida.
    """
    # shortcut: corriente del driver y factor dinámico estimados; medir el par real a 12 V.
    rated_torque:   float = 0.40     # N·m con las dos fases a rated_current
    rated_current:  float = 1.7      # A por fase
    resistance:     float = 1.5      # Ω por fase
    driver_res:     float = 0.35     # Ω del driver (dos transistores por fase)
    inductance:     float = 2.8e-3   # H por fase
    supply:         float = 12.0     # V (EPS de la placa base)
    current_rms:    float = 1.2      # A RMS configurados en el TMC2209 (pico 1,7 A)
    dynamic_factor: float = 0.75     # pérdidas en marcha frente al modelo eléctrico ideal
    max_speed:      float = 94.0     # rad/s (≈15 rev/s), límite de pasos del firmware
    rotor_inertia:  float = 5.4e-6   # kg·m²
    teeth:          int   = 50       # dientes del rotor (motor de 1.8°)
    damping_ratio:  float = 0.25     # amortiguamiento eléctrico del driver

    def __post_init__(self):
        # Constante de par (= constante de f.e.m.) por fase y par a motor parado
        self.kt = self.rated_torque / (self.rated_current * math.sqrt(2.0))
        self.holding_torque = self.kt * self.current_rms * math.sqrt(2.0)

    def torque(self, speed: float) -> float:
        """Par disponible [N·m] a la velocidad `speed` [rad/s] del eje del motor."""
        w = abs(speed)
        if w >= self.max_speed:
            return 0.0
        r = self.resistance + self.driver_res
        x = w * self.teeth * self.inductance
        z2 = r * r + x * x
        # Par máximo de un motor síncrono de imanes limitado por la tensión
        limit = self.kt * self.supply / math.sqrt(z2) - self.kt * self.kt * w * r / z2
        return max(min(self.holding_torque, self.dynamic_factor * limit), 0.0)


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
    encoders:         bool  = True      # giro en lazo cerrado (encoder): corrige los pasos perdidos.
                                        # El lateral va en lazo abierto
    home_switch:      bool  = False     # alternativa sin encoder: final de carrera a -90° en el giro: cada vez que el
                                        # jugador se levanta, el motor recupera la referencia


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
    occlusion:      bool  = True    # la bola no se ve bajo una figura
    tracker_alpha:  float = 0.6     # filtro α-β de la bola
    tracker_beta:   float = 0.25


@dataclass
class SimSpec:
    robot:  RobotSpec  = field(default_factory=RobotSpec)
    human:  HumanSpec  = field(default_factory=HumanSpec)
    camera: CameraSpec = field(default_factory=CameraSpec)


DEFAULT_SPEC = SimSpec()


def ideal_sensors_spec() -> SimSpec:
    """
    Mismo robot, pero viendo el estado real: cámara sin latencia, ruido ni oclusión
    y encoders. Solo para diagnóstico: separa lo que se pierde por la percepción de lo
    que se pierde por la imitación.
    """
    return SimSpec(robot=RobotSpec(encoders=True),
                   camera=CameraSpec(fps=300.0, latency=0.0, exposure=0.0, ball_noise=0.0,
                                     miss_prob=0.0, bar_lin_noise=0.0, bar_ang_noise=0.0,
                                     occlusion=False, tracker_alpha=1.0, tracker_beta=1.0))


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
