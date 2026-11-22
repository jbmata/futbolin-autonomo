"""
train.py
--------
Entrenamiento de un agente RL con PPO sobre el simulador de futbolín Madrid.

Requiere:
    pip install stable-baselines3[extra] gymnasium

Uso:
    python train.py --timesteps 3000000 --curriculum            # recomendado
    python train.py --timesteps 3000000 --curriculum --self-play
    python train.py --eval-only models/best/best_model
    python train.py --timesteps 1000000 --init-model models/best/best_model  # ajuste fino

El agente controla el equipo A (4 barras, 8 acciones) y ve el campo en su
propio marco. El rival es el heurístico (con nivel creciente en el currículo)
y, con --self-play, también copias congeladas de la propia política.
"""

import argparse
import os
import time
from collections import deque
from functools import partial
import numpy as np

try:
    import gymnasium as gym
    GYM_OK = True
except ImportError:
    GYM_OK = False

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import (
        EvalCallback, CheckpointCallback, BaseCallback
    )
    SB3_OK = True
except ImportError:
    SB3_OK = False

from env import FoosballEnv, HeuristicOpponent, PolicyOpponent
from reward import RewardConfig
from world import get_team_bar_indices, OBS_DIM_V1, OBS_DIM_V2


def obs_version_of(model) -> int:
    """Versión de observación con la que se entrenó un modelo (por su tamaño)."""
    return {(OBS_DIM_V1,): 1, (OBS_DIM_V2,): 2}.get(model.observation_space.shape, 3)

GAMMA = 0.99


# ---------------------------------------------------------------------------
# Oponente mixto (self-play)
# ---------------------------------------------------------------------------

class MixedOpponent:
    """
    Elige un rival por episodio: el heurístico con probabilidad
    `heuristic_prob`, o una de las políticas congeladas del grupo.
    """

    def __init__(self, heuristic_skill=1.0, heuristic_prob: float = 0.5,
                 max_pool: int = 5):
        lo = heuristic_skill[0] if isinstance(heuristic_skill, (tuple, list)) else heuristic_skill
        self.heuristic = HeuristicOpponent(skill=lo)
        self.heuristic_prob = heuristic_prob
        self.max_pool = max_pool
        self.pool = []
        self.current = self.heuristic
        self.rng = np.random.default_rng()
        self._env = self._team = self._builder = None

    def bind(self, env, team):
        """Una sola cámara simulada para todas las políticas congeladas del grupo."""
        self._env, self._team = env, team

    def add_policy(self, path: str):
        opp = PolicyOpponent(path)
        if opp.obs_version == 3 and self._env is not None:
            if self._builder is None:
                self._builder = self._env.add_observer(self._team)
            opp._builder = self._builder
        self.pool.append(opp)
        self.pool = self.pool[-self.max_pool:]

    def set_skill(self, skill):
        self.heuristic.set_skill(skill)

    def reset(self, rng=None):
        if rng is not None:
            self.rng = rng
        if not self.pool or self.rng.random() < self.heuristic_prob:
            self.current = self.heuristic
        else:
            self.current = self.pool[int(self.rng.integers(len(self.pool)))]
        self.current.reset(rng=self.rng)

    def get_commands(self, state, team='B'):
        return self.current.get_commands(state, team)


# ---------------------------------------------------------------------------
# Wrapper Gymnasium
# ---------------------------------------------------------------------------

class FoosballGymEnv(gym.Env if GYM_OK else object):
    """Wrapper Gymnasium alrededor de FoosballEnv (necesario para SB3)."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, self_play: bool = False, **kwargs):
        if GYM_OK:
            super().__init__()
        if self_play:
            kwargs["opponent"] = MixedOpponent(
                heuristic_skill=kwargs.get("opponent_skill", 1.0))
        self._env = FoosballEnv(**kwargs)
        self.observation_space = self._env.observation_space
        self.action_space      = self._env.action_space

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self._env.reset(seed=seed, options=options)

    def step(self, action):
        return self._env.step(action)

    def add_opponent_policy(self, path: str):
        """Añade una política congelada al grupo de rivales (self-play)."""
        if isinstance(self._env.opponent, MixedOpponent):
            self._env.opponent.add_policy(path)

    def render(self):
        self._env.render()

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class FoosballMetricsCallback(BaseCallback):
    """Goles marcados / recibidos, toques propios y tiros a puerta por episodio."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._ep_goals_scored = deque(maxlen=100)
        self._ep_goals_conceded = deque(maxlen=100)
        self._ep_touches = deque(maxlen=100)
        self._ep_shots = deque(maxlen=100)
        self._n_episodes = 0
        self._team_bar_idxs = set(get_team_bar_indices('A'))

    def _on_training_start(self):
        n = self.training_env.num_envs
        self._ep_touches_now = np.zeros(n, dtype=int)
        self._ep_shots_now = np.zeros(n, dtype=int)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        dones = self.locals.get("dones", [False])
        for i, (info, done) in enumerate(zip(infos, dones)):
            events = info.get("events", {})
            self._ep_touches_now[i] += sum(
                bar_idx in self._team_bar_idxs for bar_idx, _ in events.get("touches", []))
            self._ep_shots_now[i] += events.get("shot_on_target") == 'A'
            if done:
                self._ep_goals_scored.append(info["score"]["A"])
                self._ep_goals_conceded.append(info["score"]["B"])
                self._ep_touches.append(self._ep_touches_now[i])
                self._ep_shots.append(self._ep_shots_now[i])
                self._ep_touches_now[i] = 0
                self._ep_shots_now[i] = 0
                self._n_episodes += 1
                if self._n_episodes % 10 == 0:
                    self.logger.record("game/goals_scored", np.mean(self._ep_goals_scored))
                    self.logger.record("game/goals_conceded", np.mean(self._ep_goals_conceded))
                    self.logger.record("game/touches_per_ep", np.mean(self._ep_touches))
                    self.logger.record("game/shots_on_target", np.mean(self._ep_shots))
        return True


class SelfPlayCallback(BaseCallback):
    """Guarda una copia de la política cada `snapshot_freq` pasos y la añade como rival."""

    def __init__(self, save_dir: str, snapshot_freq: int, verbose: int = 0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.snapshot_freq = snapshot_freq
        self._last = 0

    def _on_training_start(self):
        self._last = self.num_timesteps
        self._snapshot()

    def _snapshot(self):
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, f"snap_{self.num_timesteps}")
        self.model.save(path)
        self.training_env.env_method("add_opponent_policy", path)
        if self.verbose:
            print(f"Self-play: nuevo rival {path}")

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last >= self.snapshot_freq:
            self._last = self.num_timesteps
            self._snapshot()
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
            # La diferencia de goles va primero; la recompensa solo desempata.
            game_score = (goal_difference, float(np.mean(scored)), points, self.last_mean_reward)
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
# Currículo
# ---------------------------------------------------------------------------

def final_reward_config(final_shaping: float = 1.0) -> RewardConfig:
    """
    Recompensa de la fase final: goles, tiros a puerta y shaping de potencial.
    El shaping de potencial no cambia la política óptima, así que se mantiene
    por defecto (final_shaping=1). Con 0 queda solo goles + tiros.
    """
    if not 0.0 <= final_shaping <= 1.0:
        raise ValueError("final_shaping debe estar entre 0 y 1.")
    base = RewardConfig()
    return RewardConfig(
        potential_x=base.potential_x * final_shaping,
        potential_v=base.potential_v * final_shaping,
        gamma=GAMMA,
    )


def phase_settings(phase: int, final_shaping: float = 1.0) -> dict:
    """Parámetros del entorno para cada fase del currículo."""
    if phase == 0:
        # Rival ausente (jugadores levantados), sin ruido: aprender a llegar a la
        # bola, golpearla hacia delante y marcar.
        return dict(max_steps=600, opponent="absent", noise=False, reward_config=RewardConfig(
            potential_prox=0.5, touch_reward=0.01, goal_penalty=-5.0, gamma=GAMMA))
    # El nivel del rival se sortea en cada partida dentro del rango de la fase:
    # siempre hay partidas en las que se puede marcar y otras en las que hay que defender.
    if phase == 1:
        return dict(max_steps=1500, opponent="heuristic", opponent_skill=(0.2, 0.6), noise=True,
                    reward_config=RewardConfig(potential_prox=0.2, gamma=GAMMA))
    if phase == 2:
        return dict(max_steps=1500, opponent="heuristic", opponent_skill=(0.4, 1.0), noise=True,
                    reward_config=final_reward_config(final_shaping))
    # Fase 3: self-play (rival mixto heurístico + copias de la política)
    return dict(max_steps=1500, opponent_skill=(0.5, 1.0), noise=True, self_play=True,
                reward_config=final_reward_config(final_shaping))


def make_env_curriculum(phase: int, final_shaping: float = 1.0, **base_kwargs):
    """Crea un entorno de la fase indicada (0, 1, 2 o 3)."""
    kwargs = phase_settings(phase, final_shaping)
    kwargs.update(base_kwargs)
    return FoosballGymEnv(**kwargs)


EVAL_SKILL = 0.7


def make_eval_env_fn(final_shaping: float = 1.0):
    """
    Evaluación siempre contra el mismo rival (heurístico de nivel EVAL_SKILL, con
    ruido), comparable entre fases. Con el nivel máximo casi no hay goles y la
    selección del mejor modelo sería ruido.
    """
    return partial(make_env_curriculum, phase=2, final_shaping=final_shaping,
                   opponent_skill=EVAL_SKILL, max_steps=1500)


def make_model(env, learning_rate=3e-4, n_steps: int = 1024, seed: int = 0,
               log_std_init: float = -0.5, tensorboard_log=None, verbose: int = 1,
               net_size: int = 256):
    """PPO con actor-crítico asimétrico si la observación es v3."""
    from observation import ACTOR_DIM, OBS_DIM_V3
    from policies import AsymmetricActorCriticPolicy
    asymmetric = env.observation_space.shape == (OBS_DIM_V3,)
    policy_kwargs = dict(net_arch=dict(pi=[net_size, net_size], vf=[256, 256]),
                         log_std_init=log_std_init)
    if asymmetric:
        policy_kwargs["actor_dim"] = ACTOR_DIM
    return PPO(
        policy=AsymmetricActorCriticPolicy if asymmetric else "MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=512,
        n_epochs=10,
        gamma=GAMMA,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.03,
        policy_kwargs=policy_kwargs,
        seed=seed,
        device="cpu",
        verbose=verbose,
        tensorboard_log=tensorboard_log,
    )


# ---------------------------------------------------------------------------
# Entrenamiento PPO
# ---------------------------------------------------------------------------

def _make_vec(env_fn, n_envs, seed, monitor_dir=None, subproc=False):
    vec_cls = SubprocVecEnv if subproc and n_envs > 1 else DummyVecEnv
    kwargs = dict(start_method="spawn") if vec_cls is SubprocVecEnv else None
    return make_vec_env(env_fn, n_envs=n_envs, seed=seed, monitor_dir=monitor_dir,
                        vec_env_cls=vec_cls, vec_env_kwargs=kwargs)


def train_ppo(
    total_timesteps: int = 3_000_000,
    n_envs:          int = 8,
    curriculum:      bool = False,
    save_dir:        str = "models",
    log_dir:         str = "logs",
    seed:            int = 42,
    final_shaping:   float = 1.0,
    self_play:       bool = False,
    init_model:      str = None,
    subproc:         bool = None,
    phase_fractions: tuple = None,
    skip_phase0:     bool = None,
    net_size:        int = 256,
    log_std:         float = None,
):
    """
    skip_phase0: sin la fase de rival ausente. Por defecto True si se parte de un
    modelo inicial (imitación), porque ya sabe jugar contra rivales.
    """
    if skip_phase0 is None:
        skip_phase0 = bool(init_model)
    phases = ([1, 2] if skip_phase0 else [0, 1, 2]) if curriculum else [2]
    if self_play:
        phases.append(3)
    if total_timesteps < len(phases) or n_envs < 1 or seed < 0:
        raise ValueError("Se necesita al menos un paso por fase, un entorno y una semilla no negativa.")
    if subproc is None:
        subproc = n_envs >= 4
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Reparto de pasos por fase (las primeras son más cortas)
    if phase_fractions is None:
        default = {0: 0.10, 1: 0.20, 2: 0.70, 3: 0.0}
        if self_play:
            default.update({2: 0.35, 3: 0.35})
        if skip_phase0 and curriculum:
            default.update({1: 0.15, 2: 0.85} if not self_play else {1: 0.15, 2: 0.40, 3: 0.45})
        fractions = np.array([default[p] for p in phases])
    else:
        fractions = np.asarray(phase_fractions, dtype=float)
        if len(fractions) != len(phases):
            raise ValueError("phase_fractions debe tener una fracción por fase.")
    fractions = fractions / fractions.sum()
    steps_list = np.maximum((fractions * total_timesteps).astype(int), 1)
    steps_list[-1] = max(total_timesteps - int(steps_list[:-1].sum()), 1)

    print(f"\nEntrenamiento PPO | pasos={total_timesteps:,} | entornos={n_envs} | semilla={seed} "
          f"| fases={phases} | {'SubprocVecEnv' if subproc else 'DummyVecEnv'}")
    model = None
    n_steps = 1024 if n_envs >= 4 else 512

    # Con arranque por imitación se afina con una tasa de aprendizaje más baja
    lr_peak = 1e-4 if init_model else 3e-4

    def learning_rate(remaining):
        completed_steps = (1.0 - remaining) * phase_target
        return 3e-5 + (lr_peak - 3e-5) * max(0.0, 1.0 - completed_steps / total_timesteps)

    for i, phase in enumerate(phases):
        final_phase = i == len(phases) - 1
        steps = int(steps_list[i])
        phase_target = (model.num_timesteps if model is not None else 0) + steps
        label = f"phase{phase}" if (curriculum or self_play) else "direct"
        env_fn = partial(make_env_curriculum, phase=phase, final_shaping=final_shaping)
        train_env = VecNormalize(
            _make_vec(env_fn, n_envs, seed + i * n_envs,
                      monitor_dir=os.path.join(log_dir, label), subproc=subproc),
            norm_obs=False, norm_reward=True, gamma=GAMMA,
        )
        eval_env = None
        try:
            n_eval_episodes = 16 if final_phase else 10
            eval_env = VecNormalize(
                _make_vec(make_eval_env_fn(final_shaping), n_eval_episodes, seed + 10_000),
                training=False, norm_obs=False, norm_reward=False,
            )
            print(f"\n--- {label} ({steps:,} pasos) ---")
            if model is None:
                model = make_model(train_env, learning_rate=learning_rate, n_steps=n_steps,
                                   seed=seed, log_std_init=-1.2 if init_model else -0.5,
                                   tensorboard_log=log_dir, net_size=net_size)
                if init_model:
                    # Solo se copian los pesos; los hiperparámetros son los de este entrenamiento.
                    model.set_parameters(init_model, exact_match=False)
                    print(f"Pesos iniciales desde {init_model}")
                if log_std is not None:
                    # Ruido de exploración: exp(log_std) en unidades de acción
                    model.policy.log_std.data.fill_(log_std)
            else:
                model.set_env(train_env)

            eval_cb = FoosballEvalCallback(
                eval_env,
                best_model_save_path=os.path.join(save_dir, "best" if final_phase else label),
                eval_seed=seed + 10_000,
                log_path=os.path.join(log_dir, "eval" if final_phase else label),
                # Al menos dos evaluaciones por fase aunque la fase sea corta
                eval_freq=max(min((50_000 if final_phase else 25_000), steps // 2) // n_envs, 1),
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                verbose=1,
            )
            callbacks = [eval_cb, FoosballMetricsCallback()]
            if phase == 3:
                callbacks.append(SelfPlayCallback(
                    os.path.join(save_dir, "selfplay"),
                    snapshot_freq=max(steps // 6, 1), verbose=1))
            if final_phase:
                callbacks.append(CheckpointCallback(
                    save_freq=max(250_000 // n_envs, 1),
                    save_path=save_dir,
                    name_prefix="foosball_ppo",
                ))
            model.learn(
                total_timesteps=steps,
                callback=callbacks,
                tb_log_name=f"ppo_{label}",
                reset_num_timesteps=(i == 0),
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
                   seed: int = 10_042, opponent_skill: float = 1.0, max_steps: int = 1500):
    """Carga y evalúa un modelo PPO guardado contra el heurístico."""
    if n_episodes < 1 or seed < 0:
        raise ValueError("Se necesita al menos un episodio y una semilla no negativa.")
    print(f"\nCargando modelo: {model_path}")
    model = PPO.load(model_path, device="cpu")
    obs_version = obs_version_of(model)

    env   = FoosballEnv(dt=0.02, max_steps=max_steps, obs_version=obs_version,
                        opponent_skill=opponent_skill)
    stats = {"rewards": [], "goals_A": [], "goals_B": [], "touches": [], "shots": []}

    for ep in range(n_episodes):
        obs, _   = env.reset(seed=seed + ep)
        done     = False
        ep_r     = 0.0
        ep_touch = 0
        ep_shots = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            done      = terminated or truncated
            ep_r     += r
            ep_touch += sum(bar_idx in env._team_bar_idxs
                            for bar_idx, _ in info["events"].get("touches", []))
            ep_shots += info["events"].get("shot_on_target") == 'A'
            if render and ep == 0:
                env.render()

        stats["rewards"].append(ep_r)
        stats["goals_A"].append(info["score"]["A"])
        stats["goals_B"].append(info["score"]["B"])
        stats["touches"].append(ep_touch)
        stats["shots"].append(ep_shots)
        print(f"  Ep {ep+1:2d}: reward={ep_r:7.2f} | "
              f"Score A:{info['score']['A']} B:{info['score']['B']} | "
              f"toques={ep_touch} | tiros={ep_shots}")

    scored, conceded = np.array(stats["goals_A"]), np.array(stats["goals_B"])
    stats["win_rate"] = float(np.mean(scored > conceded))
    stats["draw_rate"] = float(np.mean(scored == conceded))
    stats["loss_rate"] = float(np.mean(scored < conceded))
    stats["points"] = stats["win_rate"] + 0.5 * stats["draw_rate"]
    stats["goal_difference"] = float(np.mean(scored - conceded))
    print(f"\n--- Resumen ({n_episodes} episodios, semilla {seed}, rival skill {opponent_skill}) ---")
    print(f"  Reward medio:    {np.mean(stats['rewards']):.2f} +/- {np.std(stats['rewards']):.2f}")
    print(f"  Goles marcados:  {np.mean(stats['goals_A']):.2f}")
    print(f"  Goles recibidos: {np.mean(stats['goals_B']):.2f}")
    print(f"  Toques/episodio: {np.mean(stats['touches']):.1f}")
    print(f"  Tiros a puerta:  {np.mean(stats['shots']):.2f}")
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
                window = min(50, max(1, len(y) // 10))
                y_smooth = np.convolve(y, np.ones(window) / window, mode="valid")
                ax.plot(x[window - 1:], y_smooth, label=subdir, alpha=0.8)
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
    ap.add_argument("--timesteps",  type=int,   default=3_000_000,
                    help="Total de pasos de entrenamiento")
    ap.add_argument("--n-envs",     type=int,   default=8,
                    help="Entornos paralelos")
    ap.add_argument("--curriculum", action="store_true",
                    help="Currículo: sin rival -> heurístico flojo -> heurístico completo")
    ap.add_argument("--self-play", action="store_true",
                    help="Añade una fase final contra copias de la propia política")
    ap.add_argument("--log-std", type=float, default=None,
                    help="Logaritmo de la desviación inicial de exploración (p. ej. -2.5 = 0,08)")
    ap.add_argument("--net-size", type=int, default=256,
                    help="Neuronas por capa del actor (debe coincidir con la del --init-model)")
    ap.add_argument("--with-phase0", action="store_true",
                    help="Con --init-model, mantener también la fase 0 (rival ausente)")
    ap.add_argument("--init-model", default=None,
                    help="Pesos iniciales (p. ej. models/bc_init de imitation.py)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Semilla de entrenamiento; evaluacion usa seed + 10000")
    ap.add_argument("--final-shaping", type=float, default=1.0,
                    help="0..1: peso del shaping de potencial en la fase final")
    ap.add_argument("--save-dir", default="models", help="Directorio de modelos")
    ap.add_argument("--log-dir", default="logs", help="Directorio de metricas y TensorBoard")
    ap.add_argument("--no-subproc", action="store_true",
                    help="Usar DummyVecEnv (un solo proceso)")
    ap.add_argument("--eval-only",  type=str,   default=None,
                    help="Ruta al modelo para solo evaluar (sin entrenar)")
    ap.add_argument("--eval-episodes", type=int, default=50)
    ap.add_argument("--skill", type=float, default=1.0,
                    help="Nivel del heurístico rival en --eval-only (0..1)")
    ap.add_argument("--plot-curve", action="store_true",
                    help="Generar curva de aprendizaje desde logs")
    ap.add_argument("--render-eval", action="store_true",
                    help="Render ASCII durante evaluacion")
    args = ap.parse_args()

    if args.eval_only:
        evaluate_model(args.eval_only, n_episodes=args.eval_episodes, render=args.render_eval,
                       seed=args.seed + 10_000, opponent_skill=args.skill)
    elif args.plot_curve:
        plot_training_curve(args.log_dir)
    else:
        t0 = time.time()
        train_ppo(
            total_timesteps = args.timesteps,
            n_envs          = args.n_envs,
            curriculum      = args.curriculum,
            save_dir        = args.save_dir,
            log_dir         = args.log_dir,
            seed            = args.seed,
            final_shaping   = args.final_shaping,
            self_play       = args.self_play,
            init_model      = args.init_model,
            subproc         = False if args.no_subproc else None,
            skip_phase0     = False if args.with_phase0 else None,
            net_size        = args.net_size,
            log_std         = args.log_std,
        )
        print(f"\nEntrenamiento completado en {(time.time() - t0)/60:.1f} minutos.")
        print("\nEvaluando modelo entrenado...")
        best = os.path.join(args.save_dir, "best", "best_model")
        if not os.path.exists(best + ".zip"):
            best = os.path.join(args.save_dir, "foosball_ppo_final")
        evaluate_model(best, n_episodes=args.eval_episodes, seed=args.seed + 20_000)
        print("\nPara ver el progreso en TensorBoard:")
        print(f'  tensorboard --logdir "{args.log_dir}"')


if __name__ == "__main__":
    main()
