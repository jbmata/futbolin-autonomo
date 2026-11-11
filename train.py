"""
train.py
--------
Entrenamiento de un agente RL con PPO (Proximal Policy Optimization)
sobre el simulador de futbolín tipo Madrid.

Requiere:
    pip install stable-baselines3[extra] gymnasium

Uso:
    python train.py                          # entrena con PPO
    python train.py --timesteps 500000       # más pasos
    python train.py --eval-only models/best/best_model  # evaluar modelo guardado
    python train.py --curriculum             # entrenamiento curricular (fases)

El agente controla el equipo A (4 barras, 8 acciones).
El equipo B usa el oponente heurístico.
"""

import argparse
import os
import time
from collections import deque
from functools import partial
import numpy as np

# ---------------------------------------------------------------------------
# Verificar dependencias
# ---------------------------------------------------------------------------

try:
    import gymnasium as gym
    GYM_OK = True
except ImportError:
    GYM_OK = False

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import (
        EvalCallback, CheckpointCallback, BaseCallback
    )
    SB3_OK = True
except ImportError:
    SB3_OK = False

from env import FoosballEnv
from reward import RewardConfig
from world import get_team_bar_indices


# ---------------------------------------------------------------------------
# Wrapper Gymnasium
# ---------------------------------------------------------------------------

class FoosballGymEnv(gym.Env if GYM_OK else object):
    """
    Wrapper Gymnasium alrededor de FoosballEnv.
    Necesario para que SB3 pueda usar el entorno correctamente.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, **kwargs):
        if GYM_OK:
            super().__init__()
        self._env = FoosballEnv(**kwargs)

        self.observation_space = self._env.observation_space
        self.action_space      = self._env.action_space

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self._env.reset(seed=seed, options=options)

    def step(self, action):
        return self._env.step(action)

    def render(self):
        self._env.render()

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Callback personalizado: log de métricas de juego
# ---------------------------------------------------------------------------

class FoosballMetricsCallback(BaseCallback):
    """
    Registra métricas específicas del futbolín en TensorBoard:
      - Goles marcados / recibidos por episodio
      - Toques de bola propios
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._ep_goals_scored = deque(maxlen=10)
        self._ep_goals_conceded = deque(maxlen=10)
        self._ep_touches = deque(maxlen=10)
        self._n_episodes = 0
        self._team_bar_idxs = set(get_team_bar_indices('A'))

    def _on_training_start(self):
        self._ep_touches_now = np.zeros(self.training_env.num_envs, dtype=int)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        dones = self.locals.get("dones", [False])

        for i, (info, done) in enumerate(zip(infos, dones)):
            events = info.get("events", {})
            self._ep_touches_now[i] += sum(
                bar_idx in self._team_bar_idxs
                for bar_idx, _ in events.get("touches", [])
            )

            if done:
                self._ep_goals_scored.append(info["score"]["A"])
                self._ep_goals_conceded.append(info["score"]["B"])
                self._ep_touches.append(self._ep_touches_now[i])
                self._ep_touches_now[i] = 0
                self._n_episodes += 1

                # Log cada 10 episodios
                if self._n_episodes % 10 == 0:
                    self.logger.record(
                        "game/goals_scored",
                        np.mean(self._ep_goals_scored)
                    )
                    self.logger.record(
                        "game/goals_conceded",
                        np.mean(self._ep_goals_conceded)
                    )
                    self.logger.record(
                        "game/touches_per_ep",
                        np.mean(self._ep_touches)
                    )

        return True


class FoosballEvalCallback(EvalCallback):
    """Elige modelos por resultados y goles; la recompensa desempata."""

    def __init__(self, eval_env, best_model_save_path, eval_seed=10_042, **kwargs):
        super().__init__(eval_env, best_model_save_path=None, **kwargs)
        if self.eval_env.num_envs != self.n_eval_episodes:
            raise ValueError("La evaluacion necesita un entorno por episodio para fijar las semillas.")
        self.game_save_path = best_model_save_path
        self.eval_seed = eval_seed
        self.best_game_score = None
        self.game_scores = []
        self.evaluations_goals = []

    def _init_callback(self):
        super()._init_callback()
        os.makedirs(self.game_save_path, exist_ok=True)

    def _log_success_callback(self, locals_, globals_):
        super()._log_success_callback(locals_, globals_)
        if locals_["done"]:
            score = locals_["info"]["score"]
            self.game_scores.append((score["A"], score["B"]))

    def _on_step(self):
        evaluating = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        if evaluating:
            self.game_scores = []
            self.eval_env.seed(self.eval_seed)
        continue_training = super()._on_step()
        if evaluating:
            goals = np.asarray(self.game_scores)
            scored, conceded = goals[:, 0], goals[:, 1]
            wins, draws = scored > conceded, scored == conceded
            points = float(np.mean(wins + 0.5 * draws))
            goal_difference = float(np.mean(scored - conceded))
            game_score = (points, goal_difference, float(np.mean(scored)), self.last_mean_reward)
            for name, value in (
                ("win_rate", np.mean(wins)), ("draw_rate", np.mean(draws)),
                ("loss_rate", np.mean(scored < conceded)), ("points", points),
                ("goals_scored", np.mean(scored)), ("goals_conceded", np.mean(conceded)),
                ("goal_difference", goal_difference),
            ):
                self.logger.record(f"eval/{name}", float(value))
            self.logger.dump(self.num_timesteps)
            if self.log_path is not None:
                self.evaluations_goals.append(self.game_scores)
                np.savez(
                    self.log_path + "_goals", timesteps=self.evaluations_timesteps,
                    goals=self.evaluations_goals, seed=self.eval_seed,
                )
            if self.best_game_score is None or game_score > self.best_game_score:
                self.model.save(os.path.join(self.game_save_path, "best_model"))
                self.best_game_score = game_score
                if self.verbose:
                    print(f"Nuevo mejor resultado: puntos={points:.3f}, "
                          f"diferencia de goles={goal_difference:.3f}")
        return continue_training


# ---------------------------------------------------------------------------
# Currículo de entrenamiento
# ---------------------------------------------------------------------------

def make_env_curriculum(phase: int, **base_kwargs):
    """
    Crea un entorno con dificultad progresiva.

    Fase 0: Recompensa densa, oponente estático, episodios cortos.
    Fase 1: Añadir oponente heurístico, episodios más largos.
    Fase 2: Configuración completa, goles y penalizaciones de control.
    """
    if phase == 0:
        cfg = RewardConfig(
            proximity_scale=0.03,
            ball_direction_scale=0.03,
            touch_reward=0.2,
            forward_kick_reward=0.3,
            ball_in_atk_zone=0.005,
            goal_reward=5.0,
            goal_penalty=-1.0,         # penalización baja al inicio
            action_penalty_scale=0.0,  # sin penalización de movimiento
        )
        return FoosballGymEnv(
            max_steps=300,
            opponent=None,             # oponente estático
            noise=False,               # sin ruido
            reward_config=cfg,
            **base_kwargs,
        )

    elif phase == 1:
        cfg = RewardConfig(
            proximity_scale=0.003,
            ball_direction_scale=0.01,
            touch_reward=0.05,
            forward_kick_reward=0.1,
            goal_reward=10.0,
            goal_penalty=-5.0,
            action_penalty_scale=0.002,
        )
        return FoosballGymEnv(
            max_steps=500,
            opponent="heuristic",
            noise=True,
            reward_config=cfg,
            **base_kwargs,
        )

    else:  # phase 2+
        cfg = RewardConfig()
        return FoosballGymEnv(
            max_steps=750,
            opponent="heuristic",
            noise=True,
            reward_config=cfg,
            **base_kwargs,
        )


# ---------------------------------------------------------------------------
# Entrenamiento PPO
# ---------------------------------------------------------------------------

def train_ppo(
    total_timesteps: int = 300_000,
    n_envs:          int = 4,
    curriculum:      bool = False,
    save_dir:        str = "models",
    log_dir:         str = "logs",
    seed:            int = 42,
):
    phases = [0, 1, 2] if curriculum else [None]
    if total_timesteps < len(phases) or n_envs < 1 or seed < 0:
        raise ValueError("Se necesita al menos un paso por fase, un entorno y una semilla no negativa.")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"\nEntrenamiento PPO | pasos={total_timesteps:,} | entornos={n_envs} | semilla={seed}")
    steps_per_phase, remainder = divmod(total_timesteps, len(phases))
    model = None

    def learning_rate(remaining):
        # SB3 calcula el progreso respecto al final de cada llamada a learn().
        completed_steps = (1.0 - remaining) * phase_target
        return 3e-5 + 2.7e-4 * max(0.0, 1.0 - completed_steps / total_timesteps)

    for i, phase in enumerate(phases):
        final_phase = i == len(phases) - 1
        steps = steps_per_phase + (remainder if final_phase else 0)
        phase_target = (model.num_timesteps if model is not None else 0) + steps
        label = f"phase{phase}" if curriculum else "direct"
        env_fn = (partial(make_env_curriculum, phase=phase) if curriculum
                  else partial(FoosballGymEnv, dt=0.02, max_steps=750))
        train_env = make_vec_env(
            env_fn, n_envs=n_envs, seed=seed + i * n_envs,
            monitor_dir=os.path.join(log_dir, label),
        )
        eval_env = None
        try:
            n_eval_episodes = 15 if final_phase else 10
            # Un episodio por entorno mantiene las mismas semillas entre modelos.
            eval_env = make_vec_env(env_fn, n_envs=n_eval_episodes, seed=seed + 10_000)
            print(f"\n--- {label} ---")
            if model is None:
                model = PPO(
                    policy="MlpPolicy",
                    env=train_env,
                    learning_rate=learning_rate,
                    n_steps=1024 if curriculum else 2048,
                    batch_size=256,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    ent_coef=0.005,
                    target_kl=0.02,
                    use_sde=True,
                    sde_sample_freq=4,
                    policy_kwargs=dict(
                        net_arch=dict(pi=[256, 256], vf=[256, 256]),
                        log_std_init=-2.0,
                        squash_output=True,
                    ),
                    seed=seed,
                    device="cpu",
                    verbose=1,
                    tensorboard_log=log_dir,
                )
            else:
                model.set_env(train_env)

            eval_cb = FoosballEvalCallback(
                eval_env,
                best_model_save_path=os.path.join(save_dir, "best" if final_phase else label),
                eval_seed=seed + 10_000,
                log_path=os.path.join(log_dir, "eval" if final_phase else label),
                eval_freq=max((10_000 if final_phase else 5000) // n_envs, 1),
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                verbose=1,
            )
            callbacks = [eval_cb, FoosballMetricsCallback()]
            if final_phase:
                callbacks.append(CheckpointCallback(
                    save_freq=max(20_000 // n_envs, 1),
                    save_path=save_dir,
                    name_prefix="foosball_ppo",
                ))
            model.learn(
                total_timesteps=steps,
                callback=callbacks,
                tb_log_name=f"ppo_{label}",
                reset_num_timesteps=i == 0,
                progress_bar=True,
            )
            if final_phase:
                final_path = os.path.join(save_dir, "foosball_ppo_final")
                model.save(final_path)
                print(f"\nModelo final guardado en: {final_path}.zip")
        finally:
            train_env.close()
            if eval_env is not None:
                eval_env.close()

    return model


# ---------------------------------------------------------------------------
# Evaluación de un modelo guardado
# ---------------------------------------------------------------------------

def evaluate_model(model_path: str, n_episodes: int = 20, render: bool = False,
                   seed: int = 10_042):
    """Carga y evalúa un modelo PPO guardado."""
    if n_episodes < 1 or seed < 0:
        raise ValueError("Se necesita al menos un episodio y una semilla no negativa.")
    print(f"\nCargando modelo: {model_path}")
    model = PPO.load(model_path)

    env   = FoosballEnv(dt=0.02, max_steps=750)
    stats = {"rewards": [], "goals_A": [], "goals_B": [], "touches": []}

    for ep in range(n_episodes):
        obs, _   = env.reset(seed=seed + ep)
        done     = False
        ep_r     = 0.0
        ep_touch = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            done      = terminated or truncated
            ep_r     += r
            ep_touch += sum(
                bar_idx in env._team_bar_idxs
                for bar_idx, _ in info["events"].get("touches", [])
            )
            if render and ep == 0:
                env.render()

        stats["rewards"].append(ep_r)
        stats["goals_A"].append(info["score"]["A"])
        stats["goals_B"].append(info["score"]["B"])
        stats["touches"].append(ep_touch)
        print(f"  Ep {ep+1:2d}: reward={ep_r:7.2f} | "
              f"Score A:{info['score']['A']} B:{info['score']['B']} | "
              f"toques={ep_touch}")

    scored, conceded = np.array(stats["goals_A"]), np.array(stats["goals_B"])
    stats["win_rate"] = float(np.mean(scored > conceded))
    stats["draw_rate"] = float(np.mean(scored == conceded))
    stats["loss_rate"] = float(np.mean(scored < conceded))
    stats["points"] = stats["win_rate"] + 0.5 * stats["draw_rate"]
    stats["goal_difference"] = float(np.mean(scored - conceded))
    print(f"\n--- Resumen ({n_episodes} episodios, semilla {seed}) ---")
    print(f"  Reward medio:    {np.mean(stats['rewards']):.2f} +/- {np.std(stats['rewards']):.2f}")
    print(f"  Goles marcados:  {np.mean(stats['goals_A']):.2f}")
    print(f"  Goles recibidos: {np.mean(stats['goals_B']):.2f}")
    print(f"  Toques/episodio: {np.mean(stats['touches']):.1f}")
    print(f"  Victorias:       {stats['win_rate']:.1%}")
    print(f"  Empates:         {stats['draw_rate']:.1%}")
    print(f"  Derrotas:        {stats['loss_rate']:.1%}")
    print(f"  Puntos/partida:  {stats['points']:.3f}")
    print(f"  Diferencia gol:  {stats['goal_difference']:.3f}")
    return stats


# ---------------------------------------------------------------------------
# Curva de aprendizaje desde logs
# ---------------------------------------------------------------------------

def plot_training_curve(log_dir: str = "logs"):
    """Genera la curva de aprendizaje a partir de los logs del Monitor."""
    try:
        import matplotlib.pyplot as plt
        from stable_baselines3.common.results_plotter import load_results, ts2xy
    except ImportError:
        print("matplotlib o SB3 no disponibles.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Curva de aprendizaje PPO — Futbolin")

    for subdir in os.listdir(log_dir):
        path = os.path.join(log_dir, subdir)
        if os.path.isdir(path):
            try:
                x, y = ts2xy(load_results(path), "timesteps")
                # Suavizado con media movil
                window = min(50, max(1, len(y) // 10))
                y_smooth = np.convolve(y, np.ones(window) / window, mode="valid")
                x_smooth = x[window - 1:]
                ax.plot(x_smooth, y_smooth, label=subdir, alpha=0.8)
            except Exception:
                pass

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Reward medio por episodio")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = "training_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Curva guardada en: {path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not SB3_OK:
        print("\nERROR: stable-baselines3 no esta instalado.")
        print("Instalar con:")
        print("  pip install stable-baselines3[extra] gymnasium torch")
        return

    ap = argparse.ArgumentParser(description="Entrenamiento RL Futbolin")
    ap.add_argument("--timesteps",  type=int,   default=300_000,
                    help="Total de pasos de entrenamiento")
    ap.add_argument("--n-envs",     type=int,   default=4,
                    help="Entornos paralelos")
    ap.add_argument("--curriculum", action="store_true",
                    help="Usar entrenamiento curricular (fases de dificultad)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Semilla de entrenamiento; evaluacion usa seed + 10000")
    ap.add_argument("--save-dir", default="models",
                    help="Directorio de modelos")
    ap.add_argument("--log-dir", default="logs",
                    help="Directorio de metricas y TensorBoard")
    ap.add_argument("--eval-only",  type=str,   default=None,
                    help="Ruta al modelo para solo evaluar (sin entrenar)")
    ap.add_argument("--plot-curve", action="store_true",
                    help="Generar curva de aprendizaje desde logs")
    ap.add_argument("--render-eval", action="store_true",
                    help="Render ASCII durante evaluacion")
    args = ap.parse_args()

    if args.eval_only:
        evaluate_model(args.eval_only, n_episodes=20, render=args.render_eval,
                       seed=args.seed + 10_000)

    elif args.plot_curve:
        plot_training_curve(args.log_dir)

    else:
        t0    = time.time()
        train_ppo(
            total_timesteps = args.timesteps,
            n_envs          = args.n_envs,
            curriculum      = args.curriculum,
            save_dir        = args.save_dir,
            log_dir         = args.log_dir,
            seed            = args.seed,
        )
        elapsed = time.time() - t0
        print(f"\nEntrenamiento completado en {elapsed/60:.1f} minutos.")

        # Evaluacion rapida del modelo entrenado
        print("\nEvaluando modelo entrenado...")
        evaluate_model(os.path.join(args.save_dir, "foosball_ppo_final"),
                       n_episodes=10, seed=args.seed + 10_000)

        print("\nPara ver el progreso en TensorBoard:")
        print(f'  tensorboard --logdir "{args.log_dir}"')


if __name__ == "__main__":
    main()
