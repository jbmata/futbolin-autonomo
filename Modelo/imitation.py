"""
imitation.py
------------
Arranque de la política por imitación del heurístico (behavior cloning).

Profesor-alumno: el heurístico decide con el estado real y sin retardo; el
alumno solo ve lo que verá el robot (cámara con latencia y ruido, estimación
del firmware). Así aprende a actuar con información imperfecta.

Con recompensas escasas y un golpe que exige una secuencia (armar el pie hacia
atrás y luego golpear), PPO desde cero casi nunca descubre cómo marcar. Aquí se
graban partidas del heurístico y se entrena la red de PPO para copiarlo (actor)
y para predecir los retornos (crítico). Después PPO afina y supera al maestro.

Para cubrir más estados que los que visita el heurístico, durante la recogida
se ejecuta la acción del maestro con ruido gaussiano, pero se guarda la acción
sin ruido (DART, Laskey et al. 2017).

Uso:
    python imitation.py --out models/bc_init --episodes 300
    python train.py --init-model models/bc_init --curriculum ...
"""

import argparse
import numpy as np
import torch

from env import FoosballEnv
from example import HeuristicAgent
from reward import RewardConfig

GAMMA = 0.99


def collect(n_episodes: int, seed: int = 0, noise_std: float = 0.25,
            reward_config: RewardConfig = None, teams=("A", "B"),
            rivals=(("absent", 1.0, False), ("heuristic", (0.2, 0.6), True),
                    ("heuristic", (0.4, 1.0), True))):
    """
    Graba partidas del heurístico.
    Devuelve (obs, actions, returns) como arrays; los retornos son descontados
    por episodio con la recompensa del entorno.
    """
    rng = np.random.default_rng(seed)
    obs_l, act_l, ret_l = [], [], []
    per = max(n_episodes // (len(rivals) * len(teams)), 1)
    ep_seed = seed * 100_000
    for team in teams:
        for opp, skill, noise in rivals:
            env = FoosballEnv(controlled_team=team, opponent=opp, opponent_skill=skill,
                              noise=noise, reward_config=reward_config)
            agent = HeuristicAgent(env, seed=int(rng.integers(1 << 30)), reaction_delay=0.0)
            for _ in range(per):
                ep_seed += 1
                obs, _ = env.reset(seed=ep_seed)
                o, a, r = [], [], []
                done = False
                while not done:
                    teacher = agent.predict()
                    o.append(obs)
                    a.append(teacher)
                    run = np.clip(teacher + rng.normal(0.0, noise_std, size=teacher.shape), -1, 1)
                    obs, rew, term, trunc, _ = env.step(run)
                    r.append(rew)
                    done = term or trunc
                g, ret = 0.0, np.zeros(len(r), dtype=np.float32)
                g = 0.0
                for t in reversed(range(len(r))):
                    g = r[t] + GAMMA * g
                    ret[t] = g
                obs_l.append(np.asarray(o, dtype=np.float32))
                act_l.append(np.asarray(a, dtype=np.float32))
                ret_l.append(ret)
    return np.concatenate(obs_l), np.concatenate(act_l), np.concatenate(ret_l)


def pretrain(model, obs, actions, returns, epochs: int = 40, batch_size: int = 1024,
             lr: float = 1e-3, log_std: float = -1.2, seed: int = 0, verbose: bool = True):
    """Entrena actor (MSE a la acción del maestro) y crítico (retornos normalizados)."""
    torch.manual_seed(seed)
    policy = model.policy
    device = policy.device
    ret_scale = float(np.std(returns)) + 1e-6      # VecNormalize divide por la desviación de los retornos
    o = torch.as_tensor(obs, device=device)
    a = torch.as_tensor(actions, device=device)
    v = torch.as_tensor(returns / ret_scale, device=device)

    params = list(policy.mlp_extractor.parameters()) + list(policy.action_net.parameters()) \
        + list(policy.value_net.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    n = len(o)
    perm_rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        idx = perm_rng.permutation(n)
        tot_a = tot_v = 0.0
        for s in range(0, n, batch_size):
            b = torch.as_tensor(idx[s:s + batch_size], device=device)
            latent_pi, latent_vf = policy.mlp_extractor(o[b])
            mean = policy.action_net(latent_pi)
            loss_a = torch.mean((mean - a[b]) ** 2)
            loss_v = torch.mean((policy.value_net(latent_vf).squeeze(-1) - v[b]) ** 2)
            opt.zero_grad()
            (loss_a + 0.5 * loss_v).backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot_a += float(loss_a.detach()) * len(b)
            tot_v += float(loss_v.detach()) * len(b)
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"  epoca {epoch + 1:3d}: mse_accion={tot_a / n:.4f}  mse_valor={tot_v / n:.4f}")
    with torch.no_grad():
        policy.log_std.fill_(log_std)
    return model


def build_bc_model(save_path: str, episodes: int = 300, seed: int = 0,
                   epochs: int = 40, log_std: float = -1.2, verbose: bool = True):
    """Recoge datos, preentrena una red PPO y la guarda en `save_path`."""
    from functools import partial
    from stable_baselines3.common.env_util import make_vec_env
    from train import FoosballGymEnv, final_reward_config, make_model

    cfg = final_reward_config(1.0)
    if verbose:
        print(f"Recogiendo {episodes} partidas del heuristico...")
    obs, act, ret = collect(episodes, seed=seed, reward_config=cfg)
    if verbose:
        print(f"  {len(obs):,} muestras; accion media |a|={np.abs(act).mean():.3f}")
    env = make_vec_env(partial(FoosballGymEnv, reward_config=cfg), n_envs=1)
    model = make_model(env, seed=seed, log_std_init=log_std, verbose=0)
    pretrain(model, obs, act, ret, epochs=epochs, log_std=log_std, seed=seed, verbose=verbose)
    model.save(save_path)
    env.close()
    if verbose:
        print(f"Modelo BC guardado en {save_path}.zip")
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Preentrenamiento por imitacion del heuristico")
    ap.add_argument("--out", default="models/bc_init")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    build_bc_model(args.out, args.episodes, args.seed, args.epochs)
