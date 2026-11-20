import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.results_plotter import load_results

from env import FoosballEnv
from physics import FRICTION_ROLLING, RESTITUTION_WALL, PhysicsEngine
from reward import RewardConfig, compute_reward
from train import (FoosballEvalCallback, FoosballGymEnv, FoosballMetricsCallback,
                   evaluate_model, make_env_curriculum, train_ppo)
from world import BALL_RADIUS, create_standard_world


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def test_end_walls_bounce_and_goal_openings_stay_open(self):
        for direction in (-1, 1):
            for y in (0.0, 0.08, 0.09, -0.09, 0.25, -0.25):
                with self.subTest(direction=direction, y=y):
                    state = create_standard_world()
                    state.bars.clear()
                    line_x = direction * (state.field.x_max - BALL_RADIUS)
                    state.ball.reset(x=line_x - direction * 0.01, y=y,
                                     vx=direction * 2.0, vy=0.0)
                    events = PhysicsEngine(noise=False).step(state)
                    is_goal = abs(y) <= state.field.goal_y_max - BALL_RADIUS
                    self.assertEqual(events["ball_out"], is_goal)
                    self.assertEqual(events["goal"], ('A' if direction > 0 else 'B') if is_goal else None)
                    self.assertEqual(events["wall_bounce"], not is_goal)
                    if not is_goal:
                        self.assertAlmostEqual(state.ball.x, direction * (state.field.x_max - BALL_RADIUS))
                        self.assertAlmostEqual(state.ball.vx, -direction * 2.0 * (1 - FRICTION_ROLLING * 0.02) * RESTITUTION_WALL)

    def test_diagonal_shots_use_position_at_goal_line(self):
        for direction in (-1, 1):
            for y, vy, is_goal in ((0.07, 1.0, True), (0.09, -1.0, False)):
                with self.subTest(direction=direction, y=y):
                    state = create_standard_world()
                    state.bars.clear()
                    line_x = direction * (state.field.x_max - BALL_RADIUS)
                    state.ball.reset(x=line_x - direction * 0.01, y=y,
                                     vx=direction * 2.0, vy=vy)
                    events = PhysicsEngine(noise=False).step(state)
                    self.assertEqual(events["ball_out"], is_goal)
                    self.assertEqual(events["wall_bounce"], not is_goal)

    def test_curriculum_removes_auxiliary_rewards_in_final_phase(self):
        configs = [make_env_curriculum(phase)._env.reward_config for phase in range(3)]
        for name in ("proximity_scale", "ball_direction_scale", "touch_reward", "forward_kick_reward"):
            with self.subTest(component=name):
                self.assertGreater(getattr(configs[0], name), getattr(configs[1], name))
                self.assertGreater(getattr(configs[1], name), getattr(configs[2], name))
                self.assertEqual(getattr(configs[2], name), 0.0)
        self.assertEqual(configs[2], RewardConfig())
        self.assertTrue(all(cfg.ball_progress_scale == 0 for cfg in configs))
        self.assertEqual(configs[2].ball_in_atk_zone, 0.0)

    def test_final_shaping_scales_phase1_auxiliaries(self):
        phase1 = make_env_curriculum(1)._env.reward_config
        full = make_env_curriculum(2, final_shaping=1.0)._env.reward_config
        half = make_env_curriculum(2, final_shaping=0.5)._env.reward_config
        for name in ("proximity_scale", "ball_direction_scale", "touch_reward", "forward_kick_reward"):
            with self.subTest(component=name):
                self.assertAlmostEqual(getattr(full, name), getattr(phase1, name))
                self.assertAlmostEqual(getattr(half, name), getattr(phase1, name) / 2)
        self.assertEqual(make_env_curriculum(2, final_shaping=0.0)._env.reward_config, RewardConfig())
        with self.assertRaises(ValueError):
            make_env_curriculum(2, final_shaping=1.5)

    def test_reward_progress_uses_field_length(self):
        state = create_standard_world()
        config = RewardConfig(ball_progress_scale=0.08)
        for length in (state.field.length, 2.0):
            state.field.length = length
            state.ball.x = length / 4.0
            for team, expected in (('A', 0.04), ('B', -0.04)):
                with self.subTest(length=length, team=team):
                    _, breakdown = compute_reward(state, team, {}, np.zeros((4, 2)), config=config)
                    self.assertAlmostEqual(breakdown["ball_progress"], expected)

    def test_final_reward_favors_scoring_over_camping(self):
        state = create_standard_world()
        state.ball.reset(x=0.2, y=0.0, vx=0.0, vy=0.0)
        idle_reward, _ = compute_reward(state, 'A', {}, np.zeros((4, 2)))
        self.assertEqual(idle_reward, 0.0)
        for team in ('A', 'B'):
            totals = []
            for goal in (team, 'B' if team == 'A' else 'A'):
                total = 0.0
                for step in range(750):
                    action = np.full((4, 2), 1.0 if step % 2 else -1.0)
                    events = {"touches": [(0, 0), (5, 0)]}
                    if step == 749:
                        events["goal"] = goal
                    reward, _ = compute_reward(state, team, events, action, -action)
                    total += reward
                totals.append(total)
            self.assertGreater(totals[0], 750 * idle_reward)
            self.assertLess(totals[1], 750 * idle_reward)

    def test_best_model_uses_results_before_reward_and_records_goals(self):
        env = make_vec_env(FoosballGymEnv, n_envs=2)
        try:
            with tempfile.TemporaryDirectory() as directory:
                callback = FoosballEvalCallback(
                    env, best_model_save_path=directory, log_path=directory,
                    n_eval_episodes=2, eval_freq=1, verbose=0, eval_seed=123,
                )
                model = SimpleNamespace(
                    get_env=lambda: env, num_timesteps=0, save=Mock(),
                    get_vec_normalize_env=lambda: None,
                    logger=configure(folder=None, format_strings=[]),
                )
                callback.init_callback(model)

                def evaluation(scores, reward):
                    def run(*args, **kwargs):
                        for a, b in scores:
                            kwargs["callback"]({"done": True, "info": {"score": {"A": a, "B": b}}}, {})
                        return [reward, reward], [10, 10]
                    with patch("stable_baselines3.common.callbacks.evaluate_policy", side_effect=run):
                        callback.on_step()

                with patch.object(env, "seed", wraps=env.seed) as reseed:
                    evaluation([(0, 0), (0, 0)], 100)
                    evaluation([(1, 0), (0, 0)], 1)
                    evaluation([(0, 0), (0, 0)], 200)
                    self.assertEqual(model.save.call_count, 2)
                    evaluation([(1, 0), (0, 0)], 2)
                    self.assertEqual(model.save.call_count, 3)
                    self.assertEqual(callback.best_game_score, (0.5, 0.5, 0.75, 2.0))
                    self.assertEqual(reseed.call_args_list, [unittest.mock.call(123)] * 4)
                with np.load(Path(directory) / "evaluations_goals.npz") as data:
                    self.assertEqual(data["goals"].shape, (4, 2, 2))
                    np.testing.assert_array_equal(data["goals"][1], [[1, 0], [0, 0]])
                    self.assertEqual(data["seed"], 123)
        finally:
            env.close()

    def test_noisy_trajectories_and_relaunch_are_reproducible(self):
        first, second = FoosballEnv(), FoosballEnv()
        np.testing.assert_array_equal(first.reset(seed=7)[0], second.reset(seed=7)[0])
        rng = np.random.default_rng(8)
        for step in range(100):
            if step == 12:
                for env in (first, second):
                    env.state.ball.vx = env.state.ball.vy = 0.0
            action = rng.uniform(-1, 1, size=8)
            one, two = first.step(action), second.step(action)
            np.testing.assert_array_equal(one[0], two[0])
            self.assertEqual(one[1:], two[1:])
            if one[2] or one[3]:
                np.testing.assert_array_equal(first.reset()[0], second.reset()[0])
        np.testing.assert_array_equal(first.reset()[0], second.reset()[0])

    def test_reset_does_not_change_global_random_state(self):
        before = np.random.get_state()
        FoosballEnv().reset(seed=42)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_gymnasium_contract(self):
        env = FoosballGymEnv()
        try:
            check_env(env, skip_render_check=True)
        finally:
            env.close()

    def test_metrics_keep_parallel_episodes_and_own_touches_separate(self):
        env = make_vec_env(FoosballGymEnv, n_envs=2)
        try:
            callback = FoosballMetricsCallback()
            model = SimpleNamespace(get_env=lambda: env, num_timesteps=0,
                                    logger=configure(folder=None, format_strings=[]))
            callback.init_callback(model)
            callback.on_training_start({}, {})
            callback.update_locals({
                "infos": [
                    {"events": {"touches": [(0, 0), (2, 0)]}},
                    {"events": {"touches": [(5, 0), (2, 0)]}, "score": {"A": 0, "B": 1}},
                ],
                "dones": [False, True],
            })
            callback.on_step()
            callback.update_locals({
                "infos": [
                    {"events": {"touches": [(1, 0)]}, "score": {"A": 1, "B": 0}},
                    {"events": {"touches": [(0, 0), (3, 0)]}},
                ],
                "dones": [True, False],
            })
            callback.on_step()
            self.assertEqual(list(callback._ep_touches), [1, 2])
            self.assertEqual(list(callback._ep_goals_scored), [0, 1])
            self.assertEqual(list(callback._ep_goals_conceded), [1, 0])
            np.testing.assert_array_equal(callback._ep_touches_now, [0, 2])
        finally:
            env.close()

    def test_training_save_load_and_evaluation(self):
        def short_eval(*args, **kwargs):
            kwargs.update(eval_freq=1024)
            return FoosballEvalCallback(*args, **kwargs)

        def short_checkpoint(*args, **kwargs):
            kwargs["save_freq"] = 1024
            return CheckpointCallback(*args, **kwargs)

        for curriculum, timesteps in ((False, 4096), (True, 6144)):
            with self.subTest(curriculum=curriculum), tempfile.TemporaryDirectory() as directory:
                save_dir, log_dir = Path(directory) / "models", Path(directory) / "logs"
                with (contextlib.redirect_stdout(io.StringIO()),
                      patch("train.FoosballEvalCallback", side_effect=short_eval),
                      patch("train.CheckpointCallback", side_effect=short_checkpoint)):
                    model = train_ppo(timesteps, n_envs=2, curriculum=curriculum,
                                      save_dir=str(save_dir), log_dir=str(log_dir), seed=7)
                self.assertEqual(model.num_timesteps, timesteps)
                self.assertGreater(model._n_updates, 0)
                self.assertGreater(model.lr_schedule(1.0), model.lr_schedule(0.5))
                self.assertGreater(model.lr_schedule(0.5), model.lr_schedule(0.0))
                self.assertAlmostEqual(model.policy.optimizer.param_groups[0]["lr"], 3e-5)
                self.assertTrue(all(torch.isfinite(p).all() for p in model.policy.parameters()))
                self.assertTrue(model.policy.squash_output)
                path = save_dir / "foosball_ppo_final.zip"
                self.assertTrue((save_dir / "best" / "best_model.zip").is_file())
                self.assertTrue(list(save_dir.glob("foosball_ppo_*_steps.zip")))
                loaded = PPO.load(path)
                obs, _ = FoosballEnv().reset(seed=100)
                action, _ = model.predict(obs, deterministic=True)
                np.testing.assert_array_equal(action, loaded.predict(obs, deterministic=True)[0])
                self.assertTrue(np.all(np.abs(action) <= 1))
                for phase in ("phase0", "phase1", "phase2") if curriculum else ("direct",):
                    self.assertGreater(len(load_results(str(log_dir / phase))), 0)
                with contextlib.redirect_stdout(io.StringIO()):
                    stats = evaluate_model(str(path), n_episodes=1, seed=100)
                    repeated = evaluate_model(str(path), n_episodes=1, seed=100)
                self.assertEqual(stats, repeated)
                self.assertTrue(np.isfinite(stats["rewards"]).all())

    def test_invalid_training_arguments(self):
        for kwargs in ({"total_timesteps": 0}, {"n_envs": 0}, {"seed": -1},
                       {"total_timesteps": 2, "curriculum": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                train_ppo(**kwargs)


if __name__ == "__main__":
    unittest.main()
