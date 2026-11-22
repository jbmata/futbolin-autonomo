"""
optimize_teacher.py
-------------------
Optimiza automáticamente los parámetros del heurístico del robot (CMA-ES).

El heurístico del equipo A juega partidos de 30 s en el simulador realista
(motores paso a paso; el profesor ve el estado real, sin cámara) contra el
heurístico rival con niveles variados. CMA-ES prueba combinaciones de parámetros
y se queda con las que más goles por minuto ganan. Cada generación usa las
mismas semillas para todos los candidatos (números aleatorios comunes), lo que
reduce mucho el ruido de la comparación.

Hay un juego de 14 parámetros POR ROL (portero, defensa, medios, delanteros).
Se pueden optimizar todos a la vez (56 parámetros) o solo algunos roles, dejando
el resto fijo en los de `--base`; así cada rol se optimiza en un proceso o en una
máquina distinta y luego se juntan con `--merge`.

Uso:
    python optimize_teacher.py --roles GK --minutes 60 --out models/teacher/GK.json
    python optimize_teacher.py --roles GK,DEF,MID,ATK --minutes 120
    python optimize_teacher.py --merge GK=models/teacher/GK.json DEF=models/teacher/DEF.json \\
        MID=models/teacher/MID.json ATK=models/teacher/ATK.json --out models/teacher/roles.json
    python optimize_teacher.py --compare base=models/teacher/heuristic_optimized.json \\
        roles=models/teacher/roles.json --validate-matches 240
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
    params_dict, seeds, skills = args
    from env import RoleParams
    from example import HeuristicAgent
    env = _ENV
    agent = HeuristicAgent(env, seed=0, reaction_delay=0.0, params=RoleParams.from_dict(params_dict))
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
    """
    Evalúa varios conjuntos de parámetros (nombre -> dict de RoleParams) con las mismas
    semillas, repartiendo los partidos entre procesos.
    """
    seeds = seed + np.arange(matches)
    skills = [SKILLS[i % len(SKILLS)] for i in range(matches)]
    chunks = 6
    jobs, keys = [], []
    for name, pdict in named.items():
        for c in range(chunks):
            sl = slice(c, matches, chunks)
            jobs.append((pdict, seeds[sl], [skills[i] for i in range(matches)][sl]))
            keys.append(name)
    results = pool.map(_play, jobs, chunksize=1)
    return {name: _summary(np.concatenate([r for k, r in zip(keys, results) if k == name]))
            for name in named}


def _print_table(val: dict, matches: int, seconds: float):
    print(f"\n{matches} partidos de {seconds:.0f} s con semillas nuevas, rivales de nivel "
          f"{', '.join(map(str, SKILLS))}. Valores por minuto.")
    print(f"{'parametros':18s} {'marca':>6s} {'recibe':>7s} {'diferencia (IC 95%)':>21s} {'tiros':>6s}")
    for name, r in val.items():
        print(f"{name:18s} {r['scored']:6.2f} {r['conceded']:7.2f} {r['diff']:+9.2f} ± {r['ci']:4.2f}   {r['shots']:6.2f}",
              flush=True)


def main():
    import cma
    from env import PARAM_BOUNDS, ROLES, RoleParams, HeuristicParams

    ap = argparse.ArgumentParser(description="Optimiza el heuristico del robot con CMA-ES")
    ap.add_argument("--roles", default=",".join(ROLES), help="Roles a optimizar (GK,DEF,MID,ATK)")
    ap.add_argument("--base", default="models/teacher/heuristic_optimized.json",
                    help="Parámetros de partida (y valores fijos de los roles no optimizados)")
    ap.add_argument("--minutes", type=float, default=30.0, help="Tiempo máximo de optimización")
    ap.add_argument("--popsize", type=int, default=0, help="0 = automático según el número de parámetros")
    ap.add_argument("--matches", type=int, default=12, help="Partidos por candidato y generación")
    ap.add_argument("--seconds", type=float, default=30.0, help="Duración de cada partido")
    ap.add_argument("--sigma", type=float, default=0.25)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--validate-matches", type=int, default=72)
    ap.add_argument("--out", default="models/teacher/roles.json")
    ap.add_argument("--merge", nargs="+", metavar="ROL=FICHERO",
                    help="Juntar los parámetros de cada rol de varios ficheros en --out y salir")
    ap.add_argument("--compare", nargs="+", metavar="NOMBRE=FICHERO",
                    help="Solo validar y comparar ficheros de parámetros")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.merge:
        base = RoleParams.load(args.base)
        for item in args.merge:
            role, path = item.split("=", 1)
            base.roles[role] = RoleParams.load(path).roles[role]
        base.save(args.out)
        print(f"Parámetros por rol guardados en {args.out}")
        return

    ctx = mp.get_context("spawn")
    if args.compare:
        named = {}
        for item in args.compare:
            name, path = item.split("=", 1)
            named[name] = RoleParams.load(path).to_dict()
        with ctx.Pool(args.workers, initializer=_init_worker, initargs=(args.seconds,)) as pool:
            val = validate(pool, named, args.validate_matches, seed=3_000_000 + args.seed)
        _print_table(val, args.validate_matches, args.seconds)
        return

    roles = [r.strip() for r in args.roles.split(",") if r.strip()]
    for r in roles:
        if r not in ROLES:
            raise SystemExit(f"Rol desconocido: {r}")
    names = list(PARAM_BOUNDS)
    n = len(names)
    lo = np.tile(np.array([PARAM_BOUNDS[k][0] for k in names]), len(roles))
    hi = np.tile(np.array([PARAM_BOUNDS[k][1] for k in names]), len(roles))
    base = RoleParams.load(args.base)
    default_u = (np.concatenate([base.roles[r].to_array() for r in roles]) - lo) / (hi - lo)

    def build(u) -> dict:
        """Vector normalizado -> dict de RoleParams (roles no optimizados, fijos en la base)."""
        x = lo + (hi - lo) * np.clip(u, 0.0, 1.0)
        rp = RoleParams({r: HeuristicParams.from_array(base.roles[r].to_array()) for r in ROLES})
        for i, r in enumerate(roles):
            rp.roles[r] = HeuristicParams.from_array(x[i * n:(i + 1) * n])
        return rp.to_dict()

    popsize = args.popsize or int(4 + 3 * math.log(len(default_u))) + 6
    es = cma.CMAEvolutionStrategy(default_u, args.sigma, {
        "bounds": [0, 1], "popsize": popsize, "seed": args.seed + 1, "verbose": -9})
    t0 = time.time()
    best = {"fit": -1e9, "u": None}
    print(f"CMA-ES roles={roles}: {len(default_u)} parametros, poblacion {popsize}, {args.matches} partidos de "
          f"{args.seconds:.0f} s por candidato, {args.workers} procesos, maximo {args.minutes:.0f} min", flush=True)
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(args.seconds,)) as pool:
        gen = 0
        while time.time() - t0 < args.minutes * 60:
            gen += 1
            rng = np.random.default_rng(args.seed * 1000 + gen)
            seeds = rng.integers(1, 10**9, size=args.matches)
            skills = [SKILLS[i % len(SKILLS)] for i in range(args.matches)]
            cands = es.ask()
            res = pool.map(_play, [(build(u), seeds, skills) for u in cands], chunksize=1)
            fits = np.array([_fitness(r) for r in res])
            es.tell(cands, [-f for f in fits])
            i = int(np.argmax(fits))
            if fits[i] > best["fit"]:
                best = {"fit": float(fits[i]), "u": np.array(cands[i])}
            RoleParams.from_dict(build(es.result.xfavorite)).save(args.out)
            print(f"gen {gen:3d}  mejor {fits.max():+.3f}  media {fits.mean():+.3f}  "
                  f"sigma {es.sigma:.3f}  {(time.time() - t0) / 60:5.1f} min", flush=True)

        print("\nValidando con partidos nuevos...", flush=True)
        named = {"base": base.to_dict(), "optimizado": build(es.result.xfavorite)}
        if best["u"] is not None:
            named["mejor_candidato"] = build(best["u"])
        val = validate(pool, named, args.validate_matches, seed=900_000 + args.seed)
    _print_table(val, args.validate_matches, args.seconds)
    chosen = "optimizado" if val["optimizado"]["diff"] >= val["base"]["diff"] else "base"
    RoleParams.from_dict(named[chosen]).save(args.out)
    with open(args.out.replace(".json", "_validacion.json"), "w", encoding="utf-8") as f:
        json.dump({"roles": roles, "elegido": chosen, "validacion": val}, f, indent=2)
    print(f"\nGuardado '{chosen}' en {args.out}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
