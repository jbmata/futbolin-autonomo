"""
benchmark.py
------------
Compara agentes contra el mismo rival y con las mismas semillas.

Formato de partido real: el juego sigue tras cada gol (saca el portero que lo
ha recibido) durante `--seconds` segundos. Se informa de goles por minuto y de
la diferencia de goles por minuto con su intervalo de confianza del 95 %: si
los intervalos de dos agentes se solapan mucho, no hay una diferencia clara.

Uso:
    python benchmark.py                                     # solo el heurístico
    python benchmark.py models/ppo_v4/best/best_model       # + uno o varios modelos
    python benchmark.py --matches 100 --seconds 60 --skill 0.7 modelo1 modelo2
"""

import argparse
import math
import os
import numpy as np

from env import FoosballEnv
from example import HeuristicAgent


def play(policy, n_matches: int, seed: int, seconds: float = 60.0, opponent="heuristic",
         obs_version: int = None, opponent_skill=0.7, dt: float = 0.02, spec=None) -> dict:
    """Juega n partidos con las mismas semillas. policy(env, obs) -> acción."""
    steps = int(round(seconds / dt))
    env = FoosballEnv(dt=dt, max_steps=steps, opponent=opponent,
                      obs_version=obs_version, opponent_skill=opponent_skill, spec=spec)
    scored, conceded, touches, shots, lost = [], [], [], [], []
    for m in range(n_matches):
        obs, _ = env.reset(seed=seed + m)
        done, ep_touch, ep_shots = False, 0, 0
        while not done:
            obs, _, terminated, truncated, info = env.step(policy(env, obs))
            done = terminated or truncated
            ev = info["events"]
            ep_touch += sum(b in env._team_bar_idxs for b, _ in ev.get("touches", []))
            ep_shots += ev.get("shot_on_target") == env.controlled_team
        scored.append(info["score"]["A"])
        conceded.append(info["score"]["B"])
        touches.append(ep_touch)
        shots.append(ep_shots)
        lost.append(info.get("lost_steps", 0))
    minutes = steps * dt / 60.0
    scored, conceded = np.array(scored) / minutes, np.array(conceded) / minutes
    diff = scored - conceded
    half = 1.96 * diff.std(ddof=1) / math.sqrt(len(diff)) if len(diff) > 1 else float("nan")
    return {
        "scored": scored.mean(), "conceded": conceded.mean(),
        "diff": diff.mean(), "diff_ci": half,
        "win": float(np.mean(diff > 0)), "loss": float(np.mean(diff < 0)),
        "shots": float(np.mean(shots)) / minutes,
        "conversion": float(np.sum(scored) / max(np.sum(shots) / minutes, 1e-9)),
        "touches": float(np.mean(touches)) / minutes,
        "lost_steps": float(np.mean(lost)) / minutes,
    }


def heuristic_policy(params=None, reaction_delay=None, perceived=False):
    """Heurístico del robot. Con `params` y retardo 0 es el profesor optimizado (ve el estado real).
    Con `perceived=True` decide con la bola que daría la cámara: el techo de lo que puede imitar el alumno."""
    agent = {}

    def policy(env, obs):
        if agent.get("env") is not env:
            agent["env"], agent["a"] = env, HeuristicAgent(env, seed=0, params=params,
                                                           reaction_delay=reaction_delay,
                                                           perceived=perceived)
        return agent["a"].predict()
    return policy


def ppo_policy(path: str):
    """Devuelve (policy, obs_version) para un modelo PPO guardado."""
    from stable_baselines3 import PPO
    from train import obs_version_of
    model = PPO.load(path, device="cpu")
    return (lambda env, obs: model.predict(obs, deterministic=True)[0]), obs_version_of(model)


def main():
    ap = argparse.ArgumentParser(description="Benchmark de agentes del futbolin")
    ap.add_argument("models", nargs="*", help="Rutas a modelos PPO (.zip sin extension)")
    ap.add_argument("--matches", "--episodes", type=int, default=60, dest="matches")
    ap.add_argument("--seconds", type=float, default=60.0, help="Duración de cada partido")
    ap.add_argument("--seed", type=int, default=10_042)
    ap.add_argument("--skill", type=float, default=0.7, help="Nivel del heurístico rival (0..1)")
    ap.add_argument("--no-heuristic", action="store_true", help="No incluir al heurístico")
    ap.add_argument("--ideal-sensors", action="store_true",
                    help="Diagnóstico: evaluar los modelos con cámara perfecta y encoders")
    ap.add_argument("--teacher", default="models/teacher/heuristic_optimized.json",
                    help="Parámetros del profesor optimizado (se omite si el fichero no existe)")
    args = ap.parse_args()

    rows = []
    if not args.no_heuristic:
        rows.append(("heuristico (por defecto)", play(heuristic_policy(), args.matches, args.seed,
                                                    args.seconds, opponent_skill=args.skill)))
        if os.path.isfile(args.teacher):
            from env import RoleParams
            teacher = heuristic_policy(RoleParams.load(args.teacher), reaction_delay=0.0)
            rows.append(("profesor optimizado (estado real)",
                         play(teacher, args.matches, args.seed, args.seconds,
                              opponent_skill=args.skill)))
            if not args.ideal_sensors:
                eyes = heuristic_policy(RoleParams.load(args.teacher), reaction_delay=0.0, perceived=True)
                rows.append(("profesor con ojos del alumno (camara)",
                             play(eyes, args.matches, args.seed, args.seconds,
                                  opponent_skill=args.skill)))
    from hardware import ideal_sensors_spec
    spec = ideal_sensors_spec() if args.ideal_sensors else None
    for path in args.models:
        policy, version = ppo_policy(path)
        rows.append((path, play(policy, args.matches, args.seed, args.seconds,
                                obs_version=version, opponent_skill=args.skill, spec=spec)))

    print(f"\n{args.matches} partidos de {args.seconds:.0f} s, semillas {args.seed}.."
          f"{args.seed + args.matches - 1}, rival heuristico nivel {args.skill}. Valores por minuto.")
    print(f"{'agente':40s} {'marca':>6s} {'recibe':>7s} {'diferencia (IC 95%)':>21s} "
          f"{'gana':>5s} {'pierde':>6s} {'tiros':>6s} {'conv':>5s} {'pasos perd':>10s}")
    for name, r in rows:
        print(f"{name[-40:]:40s} {r['scored']:6.2f} {r['conceded']:7.2f} "
              f"{r['diff']:+9.2f} ± {r['diff_ci']:4.2f}      {r['win']:5.0%} {r['loss']:6.0%} "
              f"{r['shots']:6.2f} {r['conversion']:5.0%} {r['lost_steps']:10.0f}")


if __name__ == "__main__":
    main()
