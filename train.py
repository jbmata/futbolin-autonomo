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
    python train.py --eval-only models/best  # solo evaluar modelo guardado
    python train.py --curriculum             # entrenamiento curricular (fases)

El agente controla el equipo A (4 barras, 8 acciones).
El equipo B usa el oponente heurístico.
"""

import argparse
import os
import time
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
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import (
        EvalCallback, CheckpointCallback, BaseCallback
    )
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    SB3_OK = True
except ImportError:
    SB3_OK = False

from env import FoosballEnv
from reward import RewardConfig


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
        return self._env.reset(seed=seed)

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
      - Duración media del episodio
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._ep_goals_scored  = []
        self._ep_goals_conceded = []
        self._ep_touches       = []
        self._ep_rewards       = []
        self._ep_reward        = 0.0
        self._ep_touches_now   = 0
        self._goals_scored_now = 0
        self._goals_conceded_now = 0

    def _on_step(self) -> bool:
        # Leer info del entorno (primer env en caso de VecEnv)
        infos = self.locals.get("infos", [{}])
        dones = self.locals.get("dones", [False])

        for info, done in zip(infos, dones):
            events = info.get("events", {})
            goal   = events.get("goal")
            if goal == 'A':
                self._goals_scored_now   += 1
            elif goal == 'B':
                self._goals_conceded_now += 1

            self._ep_touches_now += len(events.get("touches", []))
            self._ep_reward      += info.get("episode_reward", 0.0)

            if done:
                self._ep_goals_scored.append(self._goals_scored_now)
                self._ep_goals_conceded.append(self._goals_conceded_now)
                self._ep_touches.append(self._ep_touches_now)
                self._ep_rewards.append(info.get("episode_reward", 0.0))

                self._goals_scored_now   = 0
                self._goals_conceded_now = 0
                self._ep_touches_now     = 0
                self._ep_reward          = 0.0

                # Log cada 10 episodios
                if len(self._ep_rewards) % 10 == 0:
                    self.logger.record(
                        "game/goals_scored",
                        np.mean(self._ep_goals_scored[-10:])
                    )
                    self.logger.record(
                        "game/goals_conceded",
                        np.mean(self._ep_goals_conceded[-10:])
                    )
                    self.logger.record(
                        "game/touches_per_ep",
                        np.mean(self._ep_touches[-10:])
                    )

        return True


# ---------------------------------------------------------------------------
# Currículo de entrenamiento
# ---------------------------------------------------------------------------

def make_env_curriculum(phase: int, **base_kwargs):
    """
    Crea un entorno con dificultad progresiva.

    Fase 0: Recompensa densa, oponente estático, episodios cortos.
    Fase 1: Añadir oponente heurístico, episodios más largos.
    Fase 2: Configuración completa, menos recompensa densa.
    """
    if phase == 0:
        cfg = RewardConfig(
            proximity_scale=0.3,       # recompensa densa alta
            ball_direction_scale=0.4,
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
            proximity_scale=0.2,
            ball_direction_scale=0.3,
            goal_reward=10.0,
            goal_penalty=-5.0,
            action_penalty_scale=0.01,
        )
        return FoosballGymEnv(
            max_steps=500,
            opponent="heuristic",
            noise=True,
            reward_config=cfg,
            **base_kwargs,
        )

    else:  # phase 2+
        cfg = RewardConfig(
            proximity_scale=0.15,
            ball_direction_scale=0.3,
            goal_reward=10.0,
            goal_penalty=-10.0,
            action_penalty_scale=0.02,
            jerk_penalty_scale=0.01,
        )
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
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Entrenamiento PPO — Futbolin tipo Madrid")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Entornos paralelos: {n_envs}")
    print(f"  Curriculo: {curriculum}")
    print(f"{'='*60}\n")

    if curriculum:
        # Fase 0: entrenamiento básico con recompensa densa
        print("--- Fase 0: oponente estatico, sin ruido ---")
        env_fn = lambda: Monitor(make_env_curriculum(phase=0))
        train_env  = DummyVecEnv([env_fn] * n_envs)
        eval_env_0 = Monitor(make_env_curriculum(phase=0))

        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log=log_dir,
            policy_kwargs=dict(net_arch=[256, 256]),
        )

        eval_cb_0 = EvalCallback(
            eval_env_0,
            best_model_save_path=os.path.join(save_dir, "phase0"),
            log_path=os.path.join(log_dir, "phase0"),
            eval_freq=max(5000 // n_envs, 1),
            n_eval_episodes=10,
            deterministic=True,
            verbose=1,
        )
        model.learn(
            total_timesteps=total_timesteps // 3,
            callback=[eval_cb_0, FoosballMetricsCallback()],
            tb_log_name="ppo_phase0",
            progress_bar=True,
        )
        train_env.close()

        # Fase 1: añadir oponente heurístico y ruido
        print("\n--- Fase 1: oponente heuristico, ruido activado ---")
        env_fn_1  = lambda: Monitor(make_env_curriculum(phase=1))
        train_env = DummyVecEnv([env_fn_1] * n_envs)
        eval_env_1 = Monitor(make_env_curriculum(phase=1))
        model.set_env(train_env)

        eval_cb_1 = EvalCallback(
            eval_env_1,
            best_model_save_path=os.path.join(save_dir, "phase1"),
            log_path=os.path.join(log_dir, "phase1"),
            eval_freq=max(5000 // n_envs, 1),
            n_eval_episodes=10,
            deterministic=True,
            verbose=1,
        )
        model.learn(
            total_timesteps=total_timesteps // 3,
            callback=[eval_cb_1, FoosballMetricsCallback()],
            tb_log_name="ppo_phase1",
            reset_num_timesteps=False,
            progress_bar=True,
        )
        train_env.close()

        # Fase 2: configuración completa
        print("\n--- Fase 2: configuracion completa ---")
        env_fn_2  = lambda: Monitor(make_env_curriculum(phase=2))
        train_env = DummyVecEnv([env_fn_2] * n_envs)
        eval_env_2 = Monitor(make_env_curriculum(phase=2))
        model.set_env(train_env)

    else:
        # Entrenamiento directo sin currículo
        env_fn    = lambda: Monitor(FoosballGymEnv(dt=0.02, max_steps=750))
        train_env = DummyVecEnv([env_fn] * n_envs)
        eval_env_2 = Monitor(FoosballGymEnv(dt=0.02, max_steps=750))

        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            verbose=1,
            tensorboard_log=log_dir,
            policy_kwargs=dict(net_arch=[256, 256]),
        )

    # Callbacks finales
    checkpoint_cb = CheckpointCallback(
        save_freq=max(20_000 // n_envs, 1),
        save_path=save_dir,
        name_prefix="foosball_ppo",
    )
    eval_cb_final = EvalCallback(
        eval_env_2,
        best_model_save_path=os.path.join(save_dir, "best"),
        log_path=os.path.join(log_dir, "eval"),
        eval_freq=max(10_000 // n_envs, 1),
        n_eval_episodes=15,
        deterministic=True,
        verbose=1,
    )

    tb_name = "ppo_curriculum_phase2" if curriculum else "ppo_foosball"
    steps   = total_timesteps // 3 if curriculum else total_timesteps
    reset   = not curriculum

    model.learn(
        total_timesteps=steps,
        callback=[eval_cb_final, checkpoint_cb, FoosballMetricsCallback()],
        tb_log_name=tb_name,
        reset_num_timesteps=reset,
        progress_bar=True,
    )

    final_path = os.path.join(save_dir, "foosball_ppo_final")
    model.save(final_path)
    print(f"\nModelo final guardado en: {final_path}.zip")
    train_env.close()

    return model


# ---------------------------------------------------------------------------
# Evaluación de un modelo guardado
# ---------------------------------------------------------------------------

def evaluate_model(model_path: str, n_episodes: int = 20, render: bool = False):
    """Carga y evalúa un modelo PPO guardado."""
    print(f"\nCargando modelo: {model_path}")
    model = PPO.load(model_path)

    env   = FoosballEnv(dt=0.02, max_steps=750)
    stats = {"rewards": [], "goals_A": [], "goals_B": [], "touches": []}

    for ep in range(n_episodes):
        obs, _   = env.reset()
        done     = False
        ep_r     = 0.0
        ep_touch = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            done      = terminated or truncated
            ep_r     += r
            ep_touch += len(info["events"].get("touches", []))
            if render and ep == 0:
                env.render()

        stats["rewards"].append(ep_r)
        stats["goals_A"].append(info["score"]["A"])
        stats["goals_B"].append(info["score"]["B"])
        stats["touches"].append(ep_touch)
        print(f"  Ep {ep+1:2d}: reward={ep_r:7.2f} | "
              f"Score A:{info['score']['A']} B:{info['score']['B']} | "
              f"toques={ep_touch}")

    print(f"\n--- Resumen ({n_episodes} episodios) ---")
    print(f"  Reward medio:    {np.mean(stats['rewards']):.2f} +/- {np.std(stats['rewards']):.2f}")
    print(f"  Goles marcados:  {np.mean(stats['goals_A']):.2f}")
    print(f"  Goles recibidos: {np.mean(stats['goals_B']):.2f}")
    print(f"  Toques/episodio: {np.mean(stats['touches']):.1f}")
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
    ap.add_argument("--eval-only",  type=str,   default=None,
                    help="Ruta al modelo para solo evaluar (sin entrenar)")
    ap.add_argument("--plot-curve", action="store_true",
                    help="Generar curva de aprendizaje desde logs")
    ap.add_argument("--render-eval", action="store_true",
                    help="Render ASCII durante evaluacion")
    args = ap.parse_args()

    if args.eval_only:
        evaluate_model(args.eval_only, n_episodes=20, render=args.render_eval)

    elif args.plot_curve:
        plot_training_curve()

    else:
        t0    = time.time()
        model = train_ppo(
            total_timesteps = args.timesteps,
            n_envs          = args.n_envs,
            curriculum      = args.curriculum,
        )
        elapsed = time.time() - t0
        print(f"\nEntrenamiento completado en {elapsed/60:.1f} minutos.")

        # Evaluacion rapida del modelo entrenado
        print("\nEvaluando modelo entrenado...")
        evaluate_model("models/foosball_ppo_final", n_episodes=10)

        print("\nPara ver el progreso en TensorBoard:")
        print("  tensorboard --logdir logs")


if __name__ == "__main__":
    main()
