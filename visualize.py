"""
visualize.py
------------
Visualización en tiempo real del agente RL jugando al futbolín.

Muestra:
  - Campo verde con porterías
  - Bola (círculo blanco) con estela de trayectoria
  - Jugadores de cada equipo (azul=A, rojo=B)
  - Barras como líneas verticales
  - Marcador y estadísticas en tiempo real
  - Gráfica de reward acumulado

Uso:
    python visualize.py                          # modelo entrenado por defecto
    python visualize.py --model models/best/best_model
    python visualize.py --heuristic              # comparar con agente heurístico
    python visualize.py --episodes 5             # jugar N episodios
    python visualize.py --speed 2.0              # velocidad x2
    python visualize.py --save replay.gif        # guardar como GIF animado
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from collections import deque

from env import FoosballEnv
from world import get_team_bar_indices, LINEAR_RANGE, BALL_RADIUS, FOOT_LENGTH


# ---------------------------------------------------------------------------
# Configuración visual
# ---------------------------------------------------------------------------

COLOR_FIELD       = "#2d7a2d"
COLOR_FIELD_LINES = "#3a9e3a"
COLOR_BALL        = "white"
COLOR_TEAM_A      = "#4488ff"   # azul
COLOR_TEAM_B      = "#ff4444"   # rojo
COLOR_BAR_A       = "#2255cc"
COLOR_BAR_B       = "#cc2222"
COLOR_TRAIL       = "#ffff88"
COLOR_GOAL_A      = "#88aaff"
COLOR_GOAL_B      = "#ff8888"
COLOR_SCORE_BG    = "#1a1a1a"

PLAYER_RADIUS     = 0.018   # radio del círculo que representa cada jugador [m]
TRAIL_LENGTH      = 40      # pasos de estela de la bola


# ---------------------------------------------------------------------------
# Clase principal de visualización
# ---------------------------------------------------------------------------

class FoosballVisualizer:

    def __init__(
        self,
        model_path: str = None,
        use_heuristic: bool = False,
        n_episodes: int = 3,
        speed: float = 1.0,
        save_path: str = None,
    ):
        self.n_episodes   = n_episodes
        self.speed        = speed
        self.save_path    = save_path
        self.use_heuristic = use_heuristic

        # Entorno
        self.env = FoosballEnv(dt=0.02, max_steps=750, noise=True)
        self.field = self.env.state.field

        # Modelo RL o heurístico
        self.model = None
        if not use_heuristic and model_path:
            try:
                from stable_baselines3 import PPO
                self.model = PPO.load(model_path)
                print(f"Modelo cargado: {model_path}")
            except Exception as e:
                print(f"No se pudo cargar el modelo: {e}")
                print("Usando agente heurístico.")
                self.use_heuristic = True

        if self.use_heuristic or self.model is None:
            from example import HeuristicAgent
            self._heuristic = HeuristicAgent(self.env)
            print("Usando agente heurístico.")
        else:
            self._heuristic = None

        # Estado de la animación
        self._ep_count    = 0
        self._step_count  = 0
        self._obs         = None
        self._done        = True
        self._score       = {"A": 0, "B": 0}
        self._ep_rewards  = []
        self._ep_reward   = 0.0
        self._trail       = deque(maxlen=TRAIL_LENGTH)
        self._last_event  = ""
        self._event_timer = 0
        self._all_ep_rewards = []
        # {(bar_idx, player_idx): frames_restantes} — flash en contacto real
        self._contact_flash: dict = {}

        # Construir figura
        self._build_figure()

    # ------------------------------------------------------------------
    # Construcción de la figura
    # ------------------------------------------------------------------

    def _build_figure(self):
        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(14, 7), facecolor=COLOR_SCORE_BG)
        self.fig.canvas.manager.set_window_title("Futbolin Autonomo — RL Agent")

        gs = GridSpec(2, 2, figure=self.fig,
                      width_ratios=[3, 1], height_ratios=[3, 1],
                      hspace=0.3, wspace=0.3)

        self.ax_field  = self.fig.add_subplot(gs[0, 0])
        self.ax_score  = self.fig.add_subplot(gs[0, 1])
        self.ax_reward = self.fig.add_subplot(gs[1, :])

        self._setup_field_ax()
        self._setup_score_ax()
        self._setup_reward_ax()

    def _setup_field_ax(self):
        ax = self.ax_field
        f  = self.field

        ax.set_facecolor(COLOR_FIELD)
        ax.set_xlim(f.x_min - 0.06, f.x_max + 0.06)
        ax.set_ylim(f.y_min - 0.04, f.y_max + 0.04)
        ax.set_aspect("equal")
        ax.axis("off")

        # Campo exterior
        rect = patches.FancyBboxPatch(
            (f.x_min, f.y_min), f.length, f.width,
            boxstyle="round,pad=0.005",
            linewidth=2, edgecolor="white", facecolor=COLOR_FIELD, zorder=1
        )
        ax.add_patch(rect)

        # Línea central
        ax.axvline(0, color=COLOR_FIELD_LINES, lw=1.5, alpha=0.6, zorder=2)

        # Círculo central
        circle = plt.Circle((0, 0), 0.06, color=COLOR_FIELD_LINES,
                             fill=False, lw=1.5, alpha=0.6, zorder=2)
        ax.add_patch(circle)

        # Porterías
        gw = f.goal_width / 2
        gd = 0.04   # profundidad visual

        for side, color in [(-1, COLOR_GOAL_A), (1, COLOR_GOAL_B)]:
            gx = f.x_min if side == -1 else f.x_max
            goal = patches.Rectangle(
                (gx + side * gd if side == 1 else gx - gd, -gw),
                gd, gw * 2,
                linewidth=1.5, edgecolor="white",
                facecolor=color, alpha=0.35, zorder=2
            )
            ax.add_patch(goal)
            ax.plot([gx, gx], [-gw, gw], color="white", lw=2.5, zorder=3)

        # Barras (líneas estáticas)
        self._bar_lines = {}
        for i, bar in enumerate(self.env.state.bars):
            color = COLOR_BAR_A if bar.team == 'A' else COLOR_BAR_B
            line, = ax.plot(
                [bar.bar_x, bar.bar_x], [f.y_min, f.y_max],
                color=color, lw=1.5, alpha=0.25, zorder=3
            )
            self._bar_lines[i] = line

        # Estela de la bola
        self._trail_line, = ax.plot([], [], color=COLOR_TRAIL,
                                    lw=1.5, alpha=0.5, zorder=4)

        # Jugadores (círculos)
        self._player_circles = {}
        for i, bar in enumerate(self.env.state.bars):
            color = COLOR_TEAM_A if bar.team == 'A' else COLOR_TEAM_B
            circles = []
            for _ in bar.player_offsets:
                c = plt.Circle((bar.bar_x, 0), PLAYER_RADIUS,
                               color=color, zorder=5, ec="white", lw=0.8)
                ax.add_patch(c)
                circles.append(c)
            self._player_circles[i] = circles

        # Bola
        self._ball_circle = plt.Circle((0, 0), BALL_RADIUS + 0.005,
                                        color=COLOR_BALL, zorder=6,
                                        ec="#cccccc", lw=1)
        ax.add_patch(self._ball_circle)

        # Texto de evento (GOL, etc.)
        self._event_text = ax.text(
            0, 0, "", fontsize=28, fontweight="bold",
            color="yellow", ha="center", va="center",
            zorder=10, alpha=0.0
        )

        # Título
        agent_label = "Heuristico" if self.use_heuristic else "Agente RL (PPO)"
        ax.set_title(
            f"  Equipo A ({agent_label})  vs  Equipo B (Heuristico)  ",
            fontsize=11, color="white", pad=6,
            bbox=dict(boxstyle="round", facecolor=COLOR_SCORE_BG, alpha=0.7)
        )

    def _setup_score_ax(self):
        ax = self.ax_score
        ax.set_facecolor(COLOR_SCORE_BG)
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.text(0.5, 0.92, "MARCADOR", fontsize=10, color="#aaaaaa",
                ha="center", va="top", transform=ax.transAxes)

        self._score_text_A = ax.text(
            0.25, 0.68, "0", fontsize=52, fontweight="bold",
            color=COLOR_TEAM_A, ha="center", va="center", transform=ax.transAxes
        )
        ax.text(0.5, 0.68, "–", fontsize=40, color="white",
                ha="center", va="center", transform=ax.transAxes)
        self._score_text_B = ax.text(
            0.75, 0.68, "0", fontsize=52, fontweight="bold",
            color=COLOR_TEAM_B, ha="center", va="center", transform=ax.transAxes
        )

        ax.text(0.25, 0.50, "A (RL)", fontsize=9, color=COLOR_TEAM_A,
                ha="center", va="center", transform=ax.transAxes)
        ax.text(0.75, 0.50, "B (Heur.)", fontsize=9, color=COLOR_TEAM_B,
                ha="center", va="center", transform=ax.transAxes)

        ax.plot([0, 1], [0.44, 0.44], color="#444444", lw=0.8, transform=ax.transAxes)

        self._info_lines = []
        labels = ["Episodio", "Paso", "Reward ep."]
        for i, lbl in enumerate(labels):
            y = 0.35 - i * 0.10
            ax.text(0.05, y, lbl + ":", fontsize=8, color="#888888",
                    va="center", transform=ax.transAxes)
            t = ax.text(0.95, y, "—", fontsize=8, color="white",
                        ha="right", va="center", transform=ax.transAxes)
            self._info_lines.append(t)

        # Indicador de velocidad bola
        ax.text(0.05, 0.05, "Vel. bola:", fontsize=8,
                color="#888888", va="center", transform=ax.transAxes)
        self._vel_text = ax.text(0.95, 0.05, "—", fontsize=8, color="white",
                                  ha="right", va="center", transform=ax.transAxes)

    def _setup_reward_ax(self):
        ax = self.ax_reward
        ax.set_facecolor(COLOR_SCORE_BG)
        ax.set_xlabel("Paso (episodio actual)", fontsize=8, color="#888888")
        ax.set_ylabel("Reward acumulado", fontsize=8, color="#888888")
        ax.tick_params(colors="#666666", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
        ax.grid(True, alpha=0.15, color="#555555")

        self._reward_line, = ax.plot([], [], color="#88ff88", lw=1.5)
        self._reward_steps = []
        self._reward_vals  = []

    # ------------------------------------------------------------------
    # Lógica de simulación
    # ------------------------------------------------------------------

    def _get_action(self):
        if self.model is not None:
            action, _ = self.model.predict(self._obs, deterministic=True)
        else:
            action = self._heuristic.predict()
        return action

    def _reset_episode(self):
        if self._ep_count >= self.n_episodes:
            return False
        self._obs, _      = self.env.reset()
        self._done        = False
        self._step_count  = 0
        self._ep_reward   = 0.0
        self._trail.clear()
        self._reward_steps = []
        self._reward_vals  = []
        self._ep_count    += 1
        return True

    # ------------------------------------------------------------------
    # Update de animación
    # ------------------------------------------------------------------

    def _update(self, frame):
        if self._done:
            ok = self._reset_episode()
            if not ok:
                self._anim.event_source.stop()
                return self._get_artists()

        # N pasos por frame (controla la velocidad)
        steps_per_frame = max(1, int(self.speed))
        for _ in range(steps_per_frame):
            if self._done:
                break
            action = self._get_action()
            self._obs, r, terminated, truncated, info = self.env.step(action)
            self._done        = terminated or truncated
            self._ep_reward  += r
            self._step_count += 1

            # Actualizar estela
            ball = self.env.state.ball
            self._trail.append((ball.x, ball.y))

            # Detectar eventos
            evt = info["events"]
            if evt.get("goal") == "A":
                self._score["A"] += 1
                self._last_event  = "GOL A!"
                self._event_timer = 30
            elif evt.get("goal") == "B":
                self._score["B"] += 1
                self._last_event  = "GOL B!"
                self._event_timer = 30

            # Registrar qué jugadores tocaron la bola este paso
            # (bar_idx, player_idx) → flash durante N frames
            for bar_idx, player_idx in evt.get("touches", []):
                self._contact_flash[(bar_idx, player_idx)] = 8  # duración del flash

        # Actualizar gráficos
        self._draw_field()
        self._draw_score(info)
        self._draw_reward()

        return self._get_artists()

    def _draw_field(self):
        ball = self.env.state.ball

        # Bola
        self._ball_circle.center = (ball.x, ball.y)

        # Estela
        if len(self._trail) > 1:
            tx = [p[0] for p in self._trail]
            ty = [p[1] for p in self._trail]
            self._trail_line.set_data(tx, ty)

        # Jugadores
        for i, bar in enumerate(self.env.state.bars):
            base_color = COLOR_TEAM_A if bar.team == 'A' else COLOR_TEAM_B
            for j, (circle, player_y) in enumerate(
                zip(self._player_circles[i], bar.get_player_abs_y())
            ):
                circle.center = (bar.bar_x, player_y)

                key = (i, j)
                flash = self._contact_flash.get(key, 0)

                if flash > 0:
                    # Contacto real con la bola: agrandar y poner amarillo
                    circle.set_radius(PLAYER_RADIUS * 1.7)
                    circle.set_facecolor("yellow")
                    self._contact_flash[key] = flash - 1
                else:
                    # Tamaño y color normales
                    circle.set_radius(PLAYER_RADIUS)
                    circle.set_facecolor(base_color)

        # Texto de evento
        if self._event_timer > 0:
            self._event_text.set_text(self._last_event)
            alpha = min(1.0, self._event_timer / 15.0)
            self._event_text.set_alpha(alpha)
            self._event_timer -= 1
        else:
            self._event_text.set_alpha(0.0)

    def _draw_score(self, info):
        self._score_text_A.set_text(str(self._score["A"]))
        self._score_text_B.set_text(str(self._score["B"]))

        self._info_lines[0].set_text(f"{self._ep_count}/{self.n_episodes}")
        self._info_lines[1].set_text(str(self._step_count))
        self._info_lines[2].set_text(f"{self._ep_reward:.1f}")

        ball = self.env.state.ball
        speed_ms = np.hypot(ball.vx, ball.vy)
        self._vel_text.set_text(f"{speed_ms:.2f} m/s")

    def _draw_reward(self):
        self._reward_steps.append(self._step_count)
        self._reward_vals.append(self._ep_reward)
        self._reward_line.set_data(self._reward_steps, self._reward_vals)
        self.ax_reward.relim()
        self.ax_reward.autoscale_view()

    def _get_artists(self):
        artists = [
            self._ball_circle, self._trail_line,
            self._event_text, self._reward_line,
            self._score_text_A, self._score_text_B,
            self._vel_text,
        ] + self._info_lines
        for circles in self._player_circles.values():
            artists.extend(circles)
        return artists

    # ------------------------------------------------------------------
    # Ejecutar
    # ------------------------------------------------------------------

    def run(self):
        interval_ms = max(10, int(20 / self.speed))   # 20ms base = 50fps sim

        self._anim = animation.FuncAnimation(
            self.fig,
            self._update,
            interval=interval_ms,
            blit=False,
            cache_frame_data=False,
        )

        if self.save_path:
            print(f"Guardando animacion en {self.save_path} ...")
            writer = animation.PillowWriter(fps=30)
            # Capturar solo los primeros 300 frames para el GIF
            anim_save = animation.FuncAnimation(
                self.fig, self._update,
                frames=300, interval=interval_ms, blit=False
            )
            anim_save.save(self.save_path, writer=writer, dpi=100)
            print(f"Guardado: {self.save_path}")
        else:
            plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Visualizador del agente RL de futbolin")
    ap.add_argument("--model",      type=str,   default="models/foosball_ppo_final",
                    help="Ruta al modelo PPO guardado")
    ap.add_argument("--heuristic",  action="store_true",
                    help="Usar agente heuristico en lugar del RL")
    ap.add_argument("--episodes",   type=int,   default=5,
                    help="Numero de episodios a visualizar")
    ap.add_argument("--speed",      type=float, default=1.0,
                    help="Velocidad de la simulacion (1.0=normal, 2.0=doble)")
    ap.add_argument("--save",       type=str,   default=None,
                    help="Guardar como GIF (ej: --save replay.gif)")
    args = ap.parse_args()

    viz = FoosballVisualizer(
        model_path    = args.model,
        use_heuristic = args.heuristic,
        n_episodes    = args.episodes,
        speed         = args.speed,
        save_path     = args.save,
    )
    viz.run()


if __name__ == "__main__":
    main()
