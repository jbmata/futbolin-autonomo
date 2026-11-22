"""
optimize_teacher.py
-------------------
Optimiza automáticamente los parámetros del heurístico del robot (CMA-ES).

El heurístico del equipo A juega partidos de 30 s en el simulador realista
(motores paso a paso, cámara no: el profesor ve el estado real) contra el
heurístico rival con niveles variados. CMA-ES prueba combinaciones de los 14
parámetros de `env.HeuristicParams` y se queda con las que más goles por minuto
ganan. Cada generación usa las mismas semillas para todos los candidatos
(números aleatorios comunes), lo que reduce mucho el ruido de la comparación.

Al final valida el resultado con partidos nuevos y los compara con los
parámetros por defecto, con intervalo de confianza.

Uso:
    python optimize_teacher.py --minutes 30
    python optimize_teacher.py --minutes 5 --popsize 8 --matches 6      # prueba rápida
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import time

import numpy as np

SKILLS = (0.4, 0.7, 1.0)
SHOT_WEIGHT = 0.10          # peso de los tiros a puerta (señal más densa que los goles)

_ENV = None


def _init_worker(seconds: float):
    global _ENV
    from env import FoosballEnv
    _ENV = FoosballEnv(max_steps=int(round(seconds / 0.02)), opponent_skill=0.7)


def _play(args):
    """Juega partidos con unos parámetros. Devuelve array (n, 4): marca, recibe, tiros, minutos."""
    x, seeds, skills = args
    from env import HeuristicParams
    from example import HeuristicAgent
    env = _ENV
    params = HeuristicParams.from_array(x)
    agent = HeuristicAgent(env, seed=0, reaction_delay=0.0, params=params)
    minutes = env.max_steps * env.dt / 60.0
    out = []
    for seed, skill in zip(seeds, skills):
        env.opponent_skill = skill
        env.reset(seed=int(seed))
        done, shots = False, 0
        while not done:
            _, _, te, tr, info = env.step(agent.predict())
            done = te or tr
            shots += info["events"].get("shot_on_target") == env.controlled_team
        out.append((info["score"]["A"], info["score"]["B"], shots, minutes))
    return np.array(out, dtype=float)


def _fitness(res: np.ndarray) -> float:
    minutes = res[:, 3]
    diff = (res[:, 0] - res[:, 1]) / minutes
    shots = res[:, 2] / minutes
    return float(np.mean(diff) + SHOT_WEIGHT * np.mean(shots))


def _summary(res: np.ndarray) -> dict:
    m = res[:, 3]
    diff = (res[:, 0] - res[:, 1]) / m
    ci = 1.96 * diff.std(ddof=1) / math.sqrt(len(diff)) if len(diff) > 1 else float("nan")
    return {"scored": float(np.mean(res[:, 0] / m)), "conceded": float(np.mean(res[:, 1] / m)),
            "diff": float(diff.mean()), "ci": ci, "shots": float(np.mean(res[:, 2] / m))}


def validate(pool, named: dict, matches: int, seed: int) -> dict:
    """Evalúa varios conjuntos de parámetros con las mismas semillas, repartiendo partidos entre procesos."""
    seeds = seed + np.arange(matches)
    skills = [SKILLS[i % len(SKILLS)] for i in range(matches)]
    chunks = 6
    jobs, keys = [], []
    for name, x in named.items():
        for c in range(chunks):
            sl = slice(c, matches, chunks)
            jobs.append((x, seeds[sl], [skills[i] for i in range(matches)][sl]))
            keys.append(name)
    results = pool.map(_play, jobs, chunksize=1)
    out = {}
    for name in named:
        out[name] = _summary(np.concatenate([r for k, r in zip(keys, results) if k == name]))
    return out


def main():
    import cma
    from env import HeuristicParams, PARAM_BOUNDS

    ap = argparse.ArgumentParser(description="Optimiza el heuristico del robot con CMA-ES")
    ap.add_argument("--minutes", type=float, default=30.0, help="Tiempo máximo de optimización")
    ap.add_argument("--popsize", type=int, default=14)
    ap.add_argument("--matches", type=int, default=12, help="Partidos por candidato y generación")
    ap.add_argument("--seconds", type=float, default=30.0, help="Duración de cada partido")
    ap.add_argument("--sigma", type=float, default=0.25)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--validate-matches", type=int, default=72)
    ap.add_argument("--out", default="models/teacher/heuristic_optimized.json")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    names = list(PARAM_BOUNDS)
    lo = np.array([PARAM_BOUNDS[n][0] for n in names])
    hi = np.array([PARAM_BOUNDS[n][1] for n in names])
    to_x = lambda u: lo + (hi - lo) * np.clip(u, 0.0, 1.0)
    default_u = (HeuristicParams().to_array() - lo) / (hi - lo)

    es = cma.CMAEvolutionStrategy(default_u, args.sigma, {
        "bounds": [0, 1], "popsize": args.popsize, "seed": args.seed + 1, "verbose": -9})
    ctx = mp.get_context("spawn")
    t0 = time.time()
    best = {"fit": -1e9, "x": None}
    print(f"CMA-ES: {len(names)} parámetros, población {args.popsize}, {args.matches} partidos de "
          f"{args.seconds:.0f} s por candidato, {args.workers} procesos, máximo {args.minutes:.0f} min", flush=True)
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(args.seconds,)) as pool:
        gen = 0
        while time.time() - t0 < args.minutes * 60:
            gen += 1
            rng = np.random.default_rng(args.seed * 1000 + gen)
            seeds = rng.integers(1, 10**9, size=args.matches)
            skills = [SKILLS[i % len(SKILLS)] for i in range(args.matches)]
            cands = es.ask()
            res = pool.map(_play, [(to_x(u), seeds, skills) for u in cands], chunksize=1)
            fits = np.array([_fitness(r) for r in res])
            es.tell(cands, [-f for f in fits])
            i = int(np.argmax(fits))
            if fits[i] > best["fit"]:
                best = {"fit": float(fits[i]), "x": to_x(cands[i])}
            mean_x = to_x(es.result.xfavorite)
            HeuristicParams.from_array(mean_x).save(args.out)
            print(f"gen {gen:3d}  mejor {fits.max():+.3f}  media {fits.mean():+.3f}  "
                  f"sigma {es.sigma:.3f}  {(time.time() - t0) / 60:5.1f} min", flush=True)

        print("\nValidando con partidos nuevos...", flush=True)
        final = to_x(es.result.xfavorite)
        named = {"por_defecto": HeuristicParams().to_array(), "optimizado": final}
        if best["x"] is not None:
            named["mejor_candidato"] = best["x"]
        val = validate(pool, named, args.validate_matches, seed=900_000 + args.seed)
    print(f"\n{args.validate_matches} partidos de {args.seconds:.0f} s con semillas nuevas, rivales de nivel "
          f"{', '.join(map(str, SKILLS))}. Valores por minuto.")
    print(f"{'parámetros':18s} {'marca':>6s} {'recibe':>7s} {'diferencia (IC 95%)':>21s} {'tiros':>6s}")
    for name, r in val.items():
        print(f"{name:18s} {r['scored']:6.2f} {r['conceded']:7.2f} {r['diff']:+9.2f} ± {r['ci']:4.2f}   {r['shots']:6.2f}")
    winner = max(("optimizado", "mejor_candidato"), key=lambda n: val[n]["diff"] if n in val else -1e9)
    chosen = named[winner]
    HeuristicParams.from_array(chosen).save(args.out)
    with open(args.out.replace(".json", "_validacion.json"), "w", encoding="utf-8") as f:
        json.dump({"elegido": winner, "validacion": val}, f, indent=2)
    print(f"\nGuardado '{winner}' en {args.out}")
    print(json.dumps({n: round(float(v), 4) for n, v in zip(names, chosen)}, indent=2))


if __name__ == "__main__":
    mp.freeze_support()
    main()
