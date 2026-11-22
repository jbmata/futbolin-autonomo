"""
dagger.py
---------
Imitación con DAgger (Ross et al., 2011) del heurístico optimizado.

Profesor: el heurístico con los parámetros de `optimize_teacher.py`; decide con
el estado real y sin retardo. Alumno: la red del robot, que solo ve cámara y
firmware (186 entradas). Con una imitación simple, el alumno solo ve los estados
que visita el profesor y, al equivocarse un poco, se sale de ellos. Con DAgger:

  1. el profesor juega y se entrena al alumno con sus acciones;
  2. el alumno juega (mezclado con el profesor con probabilidad β, que baja en
     cada ronda), y el profesor etiqueta lo que HARÍA en esos mismos estados;
  3. se añaden esos datos al conjunto y se reentrena;
  4. se repite desde el paso 2.

Las partidas son de 30 s, siguen tras cada gol y se reparten entre procesos.

Uso:
    python dagger.py --teacher models/teacher/heuristic_optimized.json --out models/dagger/student
"""

import argparse
import multiprocessing as mp
import os
import time
from functools import partial

import numpy as np

GAMMA = 0.99
RIVALS = (("absent", 1.0), ("heuristic", 0.3), ("heuristic", 0.6), ("heuristic", 0.9))
RIVAL_WEIGHTS = (0.1, 0.3, 0.3, 0.3)


def _rollout(task):
    """Juega partidos y devuelve (obs, acciones del profesor, retornos, resumen)."""
    (seeds, team, rival, beta, student_path, teacher_path, noise_std, steps) = task
    import torch
    torch.set_num_threads(1)
    from env import FoosballEnv, RoleParams
    from example import HeuristicAgent
    from train import final_reward_config

    opp, skill = rival
    env = FoosballEnv(controlled_team=team, opponent=opp, opponent_skill=skill, noise=True,
                      max_steps=steps, reward_config=final_reward_config(1.0))
    teacher = HeuristicAgent(env, seed=int(seeds[0]), reaction_delay=0.0,
                             params=RoleParams.load(teacher_path))
    student = None
    if beta < 1.0 and student_path:
        from stable_baselines3 import PPO
        student = PPO.load(student_path, device="cpu")
    rng = np.random.default_rng(int(seeds[0]))
    obs_l, act_l, ret_l = [], [], []
    scored = conceded = 0
    minutes = 0.0
    for seed in seeds:
        obs, _ = env.reset(seed=int(seed))
        o, a, r = [], [], []
        done = False
        while not done:
            label = teacher.predict()
            o.append(obs)
            a.append(label)
            if student is not None and rng.random() >= beta:
                act = student.predict(obs, deterministic=True)[0]
            else:
                act = label
            act = np.clip(act + rng.normal(0.0, noise_std, size=act.shape), -1.0, 1.0)
            obs, rew, term, trunc, info = env.step(act)
            r.append(rew)
            done = term or trunc
        g, ret = 0.0, np.zeros(len(r), dtype=np.float32)
        for t in reversed(range(len(r))):
            g = r[t] + GAMMA * g
            ret[t] = g
        obs_l.append(np.asarray(o, dtype=np.float32))
        act_l.append(np.asarray(a, dtype=np.float32))
        ret_l.append(ret)
        scored += info["score"]["A" if team == "A" else "B"]
        conceded += info["score"]["B" if team == "A" else "A"]
        minutes += env.max_steps * env.dt / 60.0
    return (np.concatenate(obs_l), np.concatenate(act_l), np.concatenate(ret_l),
            (scored, conceded, minutes))


def _tasks(n_matches, beta, student_path, teacher_path, noise_std, steps, seed, n_workers):
    """Reparte los partidos en trabajos pequeños variando equipo y rival."""
    rng = np.random.default_rng(seed)
    per_job = max(n_matches // (n_workers * 2), 1)
    tasks = []
    for j in range(0, n_matches, per_job):
        team = "A" if rng.random() < 0.7 else "B"
        rival = RIVALS[int(rng.choice(len(RIVALS), p=RIVAL_WEIGHTS))]
        seeds = rng.integers(1, 10**9, size=min(per_job, n_matches - j))
        tasks.append((seeds, team, rival, beta, student_path, teacher_path, noise_std, steps))
    return tasks


def build_dagger(out: str, teacher_path: str, rounds=(1.0, 0.5, 0.25, 0.0, 0.0),
                 matches: int = 48, epochs_first: int = 40, epochs_next: int = 15,
                 workers: int = 6, steps: int = 1500, noise_std: float = 0.1,
                 seed: int = 0, log_std: float = -1.2, net_size: int = 256):
    from stable_baselines3.common.env_util import make_vec_env
    from imitation import pretrain
    from train import FoosballGymEnv, final_reward_config, make_model

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    env = make_vec_env(partial(FoosballGymEnv, reward_config=final_reward_config(1.0)), n_envs=1)
    model = make_model(env, seed=seed, log_std_init=log_std, verbose=0, net_size=net_size)
    ctx = mp.get_context("spawn")
    data_o, data_a, data_r = [], [], []
    t0 = time.time()
    with ctx.Pool(workers) as pool:
        for it, beta in enumerate(rounds):
            student_path = out + "_tmp" if it > 0 else None
            tasks = _tasks(matches, beta, student_path, teacher_path, noise_std, steps,
                           seed * 1000 + it, workers)
            res = pool.map(_rollout, tasks, chunksize=1)
            for o, a, r, _ in res:
                data_o.append(o); data_a.append(a); data_r.append(r)
            sc = sum(x[3][0] for x in res); co = sum(x[3][1] for x in res)
            mins = sum(x[3][2] for x in res)
            O, A, R = (np.concatenate(d) for d in (data_o, data_a, data_r))
            print(f"ronda {it + 1}/{len(rounds)}  beta={beta:.2f}  datos {len(O):,}  "
                  f"partidos de esta ronda: marca {sc / mins:.2f} recibe {co / mins:.2f} por minuto  "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)
            pretrain(model, O, A, R, epochs=epochs_first if it == 0 else epochs_next,
                     seed=seed + it, log_std=log_std, verbose=False)
            model.save(out + "_tmp")
    model.save(out)
    env.close()
    print(f"Alumno guardado en {out}.zip ({(time.time() - t0) / 60:.1f} min)")
    return model


if __name__ == "__main__":
    mp.freeze_support()
    ap = argparse.ArgumentParser(description="Imitacion DAgger del heuristico optimizado")
    ap.add_argument("--teacher", default="models/teacher/heuristic_optimized.json")
    ap.add_argument("--out", default="models/dagger/student")
    ap.add_argument("--matches", type=int, default=48, help="Partidos por ronda")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--net-size", type=int, default=256)
    args = ap.parse_args()
    betas = [1.0, 0.5, 0.25] + [0.0] * max(args.rounds - 3, 0)
    build_dagger(args.out, args.teacher, rounds=tuple(betas[:args.rounds]), matches=args.matches,
                 workers=args.workers, seed=args.seed, net_size=args.net_size)
