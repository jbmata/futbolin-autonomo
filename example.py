"""
example.py
----------
Ejemplo de uso del simulador con el agente heurístico.
Sirve como test de sanidad y como baseline antes de entrenar RL.

Uso:
    python example.py              # benchmark 10 episodios
    python example.py --render     # render ASCII en consola
    python example.py --plot       # gráfico con matplotlib
    python example.py --episodes 20
"""

import argparse
import numpy as np

from env import FoosballEnv
from physics import PhysicsEngine
from world import get_team_bar_indices, LINEAR_RANGE


# ---------------------------------------------------------------------------
# Agente heurístico propio (para el equipo A)
# ---------------------------------------------------------------------------

class HeuristicAgent:
    """
    Agente heurístico que controla las 4 barras del equipo A.
    Cada barra intenta alinearse con la predicción de posición de la bola.
    La barra más cercana activa el golpeo (angular = -1).
    """

    def __init__(self, env: FoosballEnv):
        self.env      = env
        self.bar_idxs = get_team_bar_indices('A')

    def predict(self) -> np.ndarray:
        state = self.env.state
        ball  = state.ball
        field = state.field

        action = np.zeros(8, dtype=np.float32)

        # Encontrar la barra del equipo A más cercana a la bola en X
        closest_idx = min(
            self.bar_idxs,
            key=lambda i: abs(state.bars[i].bar_x - ball.x)
        )

        for k, bar_idx in enumerate(self.bar_idxs):
            bar = state.bars[bar_idx]

            # Predecir Y cuando la bola llegue a esta barra
            pred_y     = PhysicsEngine.predict_ball_y(ball, bar.bar_x, field)
            linear_cmd = float(np.clip(pred_y / LINEAR_RANGE, -1.0, 1.0))

            # Golpear si esta es la barra más cercana y la bola se acerca
            approaching = (ball.vx * (bar.bar_x - ball.x)) > 0
            kick        = (bar_idx == closest_idx) and approaching
            angular_cmd = -1.0 if kick else 0.0

            action[2 * k]     = linear_cmd
            action[2 * k + 1] = angular_cmd

        return action


# ---------------------------------------------------------------------------
# Bucle de episodio
# ---------------------------------------------------------------------------

def run_episode(env: FoosballEnv, agent: HeuristicAgent,
                render: bool = False, verbose: bool = True) -> dict:
    obs, _ = env.reset()
    total_r = 0.0
    touches = 0
    done    = False

    while not done:
        action = agent.predict()
        obs, reward, terminated, truncated, info = env.step(action)
        done     = terminated or truncated
        total_r += reward
        touches += len(info["events"].get("touches", []))
        if render:
            env.render()

    if verbose:
        reason = "gol/bola fuera" if terminated else "max steps"
        print(f"  [{reason:15s}] {env._step_count:4d} pasos | "
              f"reward={total_r:7.2f} | toques={touches:3d} | "
              f"Score A:{info['score']['A']} B:{info['score']['B']}")
    return {
        "total_reward": total_r,
        "n_steps":      env._step_count,
        "touches":      touches,
        "score":        info["score"],
    }


def benchmark(n_episodes: int = 10):
    env   = FoosballEnv(dt=0.02, max_steps=750)
    agent = HeuristicAgent(env)

    print(f"\n{'='*60}")
    print(f"  Benchmark — Agente Heurístico — {n_episodes} episodios")
    print(f"{'='*60}")

    results = [run_episode(env, agent) for _ in range(n_episodes)]

    rewards = [r["total_reward"] for r in results]
    touches = [r["touches"]      for r in results]
    goals_A = sum(r["score"]["A"] for r in results)
    goals_B = sum(r["score"]["B"] for r in results)

    print(f"\n--- Resumen ---")
    print(f"  Reward medio:   {np.mean(rewards):7.2f} +/- {np.std(rewards):.2f}")
    print(f"  Toques/episodio:{np.mean(touches):6.1f}")
    print(f"  Goles marcados: {goals_A}  |  Goles recibidos: {goals_B}")


# ---------------------------------------------------------------------------
# Visualización matplotlib
# ---------------------------------------------------------------------------

def run_with_plot(n_steps: int = 500):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        print("matplotlib no disponible. Instalar con: pip install matplotlib")
        return

    env   = FoosballEnv(dt=0.02, max_steps=n_steps)
    agent = HeuristicAgent(env)
    obs, _ = env.reset(seed=42)

    ball_xs, ball_ys, rewards = [], [], []
    done = False
    while not done:
        action = agent.predict()
        obs, r, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ball_xs.append(env.state.ball.x)
        ball_ys.append(env.state.ball.y)
        rewards.append(r)

    f = env.state.field
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Simulador Futbolín — Agente Heurístico", fontsize=13)

    ax = axes[0]
    ax.set_title("Trayectoria de la bola")
    rect = patches.Rectangle((f.x_min, f.y_min), f.length, f.width,
                               linewidth=2, edgecolor='green',
                               facecolor='lightgreen', alpha=0.25)
    ax.add_patch(rect)
    # Porterías
    for gx, color in [(f.x_min, 'blue'), (f.x_max, 'red')]:
        ax.plot([gx, gx], [f.goal_y_min, f.goal_y_max],
                linewidth=4, color=color, alpha=0.7)
    sc = ax.scatter(ball_xs, ball_ys, c=range(len(ball_xs)),
                    cmap='viridis', s=4, alpha=0.5)
    plt.colorbar(sc, ax=ax, label='Paso')
    # Barras
    for bar in env.state.bars:
        color = '#4444ff' if bar.team == 'A' else '#ff4444'
        ax.axvline(bar.bar_x, color=color, alpha=0.25, linewidth=1)
    ax.set_xlim(f.x_min - 0.05, f.x_max + 0.05)
    ax.set_ylim(f.y_min - 0.05, f.y_max + 0.05)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_aspect('equal')

    ax = axes[1]
    ax.set_title("Recompensa acumulada")
    ax.plot(np.cumsum(rewards), color='purple')
    ax.set_xlabel("Paso")
    ax.set_ylabel("Reward acumulado")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("simulacion_foosball.png", dpi=150, bbox_inches='tight')
    print("Grafico guardado en: simulacion_foosball.png")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Simulador de Futbolin")
    ap.add_argument("--render",   action="store_true")
    ap.add_argument("--plot",     action="store_true")
    ap.add_argument("--episodes", type=int, default=10)
    args = ap.parse_args()

    if args.plot:
        run_with_plot()
    elif args.render:
        env   = FoosballEnv(dt=0.02, max_steps=120)
        agent = HeuristicAgent(env)
        run_episode(env, agent, render=True, verbose=True)
    else:
        benchmark(args.episodes)

    # Info del entorno
    env = FoosballEnv()
    print(f"\nEspacio de observacion: {env.obs_dim} dims")
    print(f"Espacio de accion:      {env.action_dim} dims")
    print("Accion: [lin_GK, ang_GK, lin_ATK, ang_ATK, lin_MID, ang_MID, lin_DEF, ang_DEF]")


if __name__ == "__main__":
    main()
