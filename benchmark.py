"""
benchmark.py
------------
Compara agentes con las mismas semillas y reporta metricas que SI discriminan
cuando la mayoria de partidas acaban 0-0.

Uso:
    python benchmark.py                                   # solo baseline heuristico
    python benchmark.py models/ppo_v2/foosball_ppo_final  # + uno o varios modelos PPO
    python benchmark.py --episodes 200 models/best/best_model

Metricas: goles marcados/recibidos por episodio, diferencia de goles,
% de partidas con algun gol y toques propios por episodio.
"""

import argparse
import numpy as np

from env import FoosballEnv
from example import HeuristicAgent


def play(policy, n_episodes: int, seed: int, opponent="heuristic") -> dict:
    """Juega n episodios con las mismas semillas. policy(env, obs) -> accion."""
    env = FoosballEnv(dt=0.02, max_steps=750, opponent=opponent)
    scored, conceded, touches = [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, ep_touch = False, 0
        while not done:
            obs, _, terminated, truncated, info = env.step(policy(env, obs))
            done = terminated or truncated
            ep_touch += sum(b in env._team_bar_idxs
                            for b, _ in info["events"].get("touches", []))
        scored.append(info["score"]["A"])
        conceded.append(info["score"]["B"])
        touches.append(ep_touch)
    scored, conceded = np.array(scored), np.array(conceded)
    return {
        "scored": scored.mean(),
        "conceded": conceded.mean(),
        "goal_diff": (scored - conceded).mean(),
        "any_goal": float(np.mean((scored + conceded) > 0)),
        "touches": float(np.mean(touches)),
    }


def heuristic_policy():
    agent = {}

    def policy(env, obs):
        if agent.get("env") is not env:
            agent["env"], agent["a"] = env, HeuristicAgent(env)
        return agent["a"].predict()
    return policy


def ppo_policy(path: str):
    from stable_baselines3 import PPO
    model = PPO.load(path)
    return lambda env, obs: model.predict(obs, deterministic=True)[0]


def main():
    ap = argparse.ArgumentParser(description="Benchmark de agentes del futbolin")
    ap.add_argument("models", nargs="*", help="Rutas a modelos PPO (.zip sin extension)")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=10_042)
    args = ap.parse_args()

    rows = [("heuristico A vs heuristico B", play(heuristic_policy(), args.episodes, args.seed))]
    for path in args.models:
        rows.append((path, play(ppo_policy(path), args.episodes, args.seed)))

    print(f"\n{args.episodes} episodios, semillas {args.seed}..{args.seed + args.episodes - 1}")
    print(f"{'agente':44s} {'marca':>6s} {'recibe':>7s} {'dif':>6s} {'%gol':>6s} {'toques':>7s}")
    for name, r in rows:
        print(f"{name[-44:]:44s} {r['scored']:6.2f} {r['conceded']:7.2f} "
              f"{r['goal_diff']:+6.2f} {r['any_goal']:6.0%} {r['touches']:7.1f}")


if __name__ == "__main__":
    main()
