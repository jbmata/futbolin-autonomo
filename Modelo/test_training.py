import contextlib
import io
import math
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

from actuators import ActuatorBank, BarActuator, StepperBarActuator
from env import FoosballEnv, HeuristicOpponent, action_to_commands
from example import HeuristicAgent
from hardware import RobotSpec
from physics import PhysicsEngine, PhysicsParams
from vision import ball_occluded
from world import BAR_DEFINITIONS
from reward import RewardConfig, compute_reward, potential
from train import (FoosballEvalCallback, FoosballGymEnv, FoosballMetricsCallback,
                   evaluate_model, final_reward_config, make_env_curriculum, train_ppo)
from world import (BALL_RADIUS, FOOT_ROUNDING, KICK_ANGLE_MAX, OBS_DIM_V1, OBS_DIM_V2,
                   create_standard_world, get_team_bar_indices)

ATK_A = 5   # delanteros A, x = +0.20


def _drive(state, bar_idx, actuator):
    def update(h):
        bar = state.bars[bar_idx]
        bar.linear_pos, bar.angle, bar.linear_vel, bar.angular_vel = actuator.update(h)
    return update


def _sync(state, actuator):
    def on_bar_change(bar_idx):
        bar = state.bars[bar_idx]
        actuator._angular.set_true(bar.angle, bar.angular_vel)
        actuator._linear.set_true(bar.linear_pos, bar.linear_vel)
    return on_bar_change


class PhysicsTests(unittest.TestCase):
    def setUp(self):
        self.physics = PhysicsEngine(noise=False)

    def _kick(self, ball_x, ball_vx, start_angle, target_cmd, steps=20):
        state = create_standard_world()
        act = StepperBarActuator(add_noise=False, n_players=3)   # motor del robot
        act._angular.reset(start_angle)
        state.bars[ATK_A].angle = start_angle
        act.set_command(0.0, target_cmd)
        state.ball.reset(x=ball_x, y=0.0, vx=ball_vx, vy=0.0)
        touched, max_vx = False, -np.inf
        for _ in range(steps):
            events = self.physics.step(state, _drive(state, ATK_A, act), _sync(state, act))
            touched |= any(b == ATK_A for b, _ in events["touches"])
            max_vx = max(max_vx, state.ball.vx)
        return max_vx, touched

    def test_forward_kick_sends_ball_forward_from_behind_and_from_rest(self):
        # Bola rodando hacia delante por detrás del pie, y bola parada delante.
        for ball_x, ball_vx in ((0.185, 0.2), (0.225, 0.0)):
            with self.subTest(ball_x=ball_x):
                # Armado a ~-108°: desde 180° la cabeza de la figura toca antes la bola
                max_vx, touched = self._kick(ball_x, ball_vx, -0.6 * math.pi, 0.6)
                self.assertTrue(touched)
                self.assertGreater(max_vx, 1.5)

    def test_raised_foot_lets_ball_pass_and_lowered_foot_blocks(self):
        for angle, passes in ((math.pi / 2, True), (0.0, False)):
            with self.subTest(angle=angle):
                state = create_standard_world()
                state.bars[ATK_A].angle = angle
                state.ball.reset(x=0.30, y=0.0, vx=-1.0, vy=0.0)
                for _ in range(8):
                    self.physics.step(state)
                if passes:
                    self.assertLess(state.ball.x, 0.17)
                    self.assertLess(state.ball.vx, 0.0)
                else:
                    self.assertGreater(state.ball.x, 0.22)
                    self.assertGreater(state.ball.vx, 0.0)

    def test_kick_zone_matches_geometry(self):
        state = create_standard_world()
        bar = state.bars[ATK_A]
        bar.angle = KICK_ANGLE_MAX - 0.02
        self.assertLess(bar.foot_tip_height(), 2 * BALL_RADIUS + FOOT_ROUNDING)
        bar.angle = KICK_ANGLE_MAX + 0.05
        self.assertGreater(bar.foot_tip_height(), 2 * BALL_RADIUS + FOOT_ROUNDING)

    def test_taco_pins_ball_under_sole(self):
        # Bajar el taco sobre una bola parada bajo la suela: la bola queda pisada.
        state = create_standard_world()
        act = BarActuator(add_noise=False)
        act._angular.reset(1.6)
        state.bars[ATK_A].angle = 1.6
        act.set_command(0.0, 0.25)
        state.ball.reset(x=0.27, y=0.0, vx=0.0, vy=0.0)
        pinned = 0
        for _ in range(40):
            events = self.physics.step(state, _drive(state, ATK_A, act), _sync(state, act))
            pinned += events["pinned"]
        self.assertGreater(pinned, 20)
        self.assertLess(abs(state.ball.vx), 0.05)
        self.assertAlmostEqual(state.ball.x, 0.27, delta=0.02)
        self.assertGreater(state.bars[ATK_A].angle, 0.25 * math.pi + 0.1)

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
                        self.assertLess(state.ball.vx * direction, 0.0)
                        self.assertLess(abs(state.ball.vx), 2.0 * 0.7)

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

    def test_physics_is_mirror_symmetric(self):
        rng = np.random.default_rng(0)
        for trial in range(8):
            e1, e2 = (FoosballEnv(opponent=None, noise=False, obs_version=2) for _ in range(2))
            for e in (e1, e2):
                e.actuators = ActuatorBank(kinds=["human"] * 8, add_noise=False,
                                           n_players=[n for _, n, _, _ in BAR_DEFINITIONS],
                                           spec=e.spec)
            e1.reset(seed=trial); e2.reset(seed=trial)
            b = e1.state.ball
            e2.state.ball.reset(x=-b.x, y=-b.y, vx=-b.vx, vy=-b.vy)
            for _ in range(300):
                cmds = rng.uniform(-1, 1, size=(8, 2))
                for i in range(8):
                    e1.actuators.set_commands(i, *cmds[i])
                    e2.actuators.set_commands(7 - i, -cmds[i, 0], -cmds[i, 1])
                ev1 = e1.physics.step(e1.state, e1._bar_update, e1._on_bar_change)
                e2.physics.step(e2.state, e2._bar_update, e2._on_bar_change)
                b1, b2 = e1.state.ball, e2.state.ball
                self.assertAlmostEqual(b1.x, -b2.x, places=8)
                self.assertAlmostEqual(b1.y, -b2.y, places=8)
                if ev1["goal"]:
                    break

    def test_ball_slows_down_and_stops(self):
        state = create_standard_world()
        state.bars.clear()
        state.ball.reset(x=0.0, y=0.0, vx=0.3, vy=0.0)
        for _ in range(300):
            self.physics.step(state)
        self.assertEqual(state.ball.vx, 0.0)
        self.assertLess(state.ball.x, 0.4)

    def test_shot_on_target(self):
        state = create_standard_world()
        state.ball.reset(x=0.3, y=0.0, vx=2.0, vy=0.1)
        self.assertTrue(PhysicsEngine.shot_on_target(state.ball, state.field, 'A'))
        self.assertFalse(PhysicsEngine.shot_on_target(state.ball, state.field, 'B'))
        state.ball.reset(x=0.3, y=0.0, vx=2.0, vy=1.5)
        self.assertFalse(PhysicsEngine.shot_on_target(state.ball, state.field, 'A'))


class RealismTests(unittest.TestCase):
    def test_stepper_loses_steps_on_hard_impact_and_encoder_recovers(self):
        for encoders in (False, True):
            with self.subTest(encoders=encoders):
                act = StepperBarActuator(add_noise=False, n_players=1,
                                         robot=RobotSpec(encoders=encoders))
                act.reset()
                act._angular.set_true(0.0, -45.0)      # golpe fuerte de la bola al portero
                for _ in range(25):
                    act.update(0.02)
                    act.end_of_step()
                lost = act.lost_steps()[1]
                self.assertGreater(lost, 0)
                self.assertEqual(lost % 4, 0)
                err = abs(act._angular.pos - act._angular.est_pos)
                if encoders:
                    self.assertLess(err, 0.02)
                else:
                    self.assertGreater(err, math.radians(5))

    def test_stepper_follows_profile_without_load(self):
        act = StepperBarActuator(add_noise=False, n_players=5)
        act.reset()
        act.set_command(1.0, 0.5)
        for _ in range(50):
            act.update(0.02)
            act.end_of_step()
        self.assertEqual(act.lost_steps(), (0, 0))
        self.assertAlmostEqual(act._linear.pos, 0.07, delta=0.002)
        self.assertAlmostEqual(act._angular.pos, math.pi / 2, delta=0.03)

    def test_light_bar_recoils_when_hit(self):
        state = create_standard_world()
        state.ball.reset(x=-0.42, y=0.0, vx=-3.0, vy=0.0)     # tiro hacia el portero A
        PhysicsEngine(noise=False).step(state)
        for _ in range(3):
            PhysicsEngine(noise=False).step(state)
        self.assertLess(state.bars[0].angular_vel, -5.0)   # el portero retrocede
        # Con masa infinita rebotaría a ~2 m/s; una barra ligera absorbe el tiro
        self.assertLess(abs(state.ball.vx), 1.0)

    def test_struck_ball_slides_then_rolls(self):
        state = create_standard_world()
        state.bars.clear()
        state.ball.reset(x=0.0, y=0.0, vx=0.0, vy=0.0)
        state.ball.vx = 2.0                                  # golpe seco: velocidad sin giro
        physics = PhysicsEngine(noise=False, params=PhysicsParams(rolling_decel=0.0, viscous_drag=0.0))
        for _ in range(30):
            physics.step(state)
        self.assertAlmostEqual(state.ball.vx, 2.0 * 5.0 / 7.0, places=3)
        self.assertAlmostEqual(state.ball.wx, state.ball.vx, places=6)

    def test_camera_lags_and_ball_hides_under_figures(self):
        env = FoosballEnv(opponent=None, noise=False)
        env.reset(seed=0)
        env.state.ball.reset(x=0.0, y=0.3, vx=1.0, vy=0.0)
        for _ in range(10):
            obs, *_ = env.step(np.zeros(8))
        seen_x = obs[0] * env.state.field.length / 2.0
        lag = env.spec.camera.latency
        self.assertAlmostEqual(env.state.ball.x - seen_x, 1.0 * lag, delta=0.03)
        state = create_standard_world()
        state.ball.reset(x=state.bars[ATK_A].bar_x, y=0.0, vx=0.0, vy=0.0)
        self.assertTrue(ball_occluded(state))
        state.ball.reset(x=0.0, y=0.3, vx=0.0, vy=0.0)
        self.assertFalse(ball_occluded(state))

    def test_asymmetric_actor_ignores_privileged_part(self):
        from observation import ACTOR_DIM
        from policies import actor_forward, actor_weights
        from stable_baselines3.common.env_util import make_vec_env
        from train import make_model
        venv = make_vec_env(FoosballGymEnv, n_envs=1)
        try:
            model = make_model(venv, verbose=0)
            obs, _ = FoosballEnv().reset(seed=1)
            other = obs.copy()
            other[ACTOR_DIM:] = np.random.default_rng(0).uniform(-1, 1, len(obs) - ACTOR_DIM)
            a1 = model.predict(obs, deterministic=True)[0]
            a2 = model.predict(other, deterministic=True)[0]
            np.testing.assert_allclose(a1, a2, atol=1e-6)
            np.testing.assert_allclose(actor_forward(actor_weights(model), obs[:ACTOR_DIM]),
                                       a1, atol=1e-5)
            self.assertEqual(model.policy.mlp_extractor.policy_net[0].in_features, ACTOR_DIM)
        finally:
            venv.close()


class EnvTests(unittest.TestCase):
    def test_observation_mirror_between_teams(self):
        a, b = create_standard_world(), create_standard_world()
        a.ball.reset(x=0.21, y=0.1, vx=-0.7, vy=0.3)
        a.bars[0].linear_pos, a.bars[0].angle, a.bars[0].angular_vel = 0.03, 0.4, 2.0
        b.ball.reset(x=-0.21, y=-0.1, vx=0.7, vy=-0.3)
        b.bars[7].linear_pos, b.bars[7].angle, b.bars[7].angular_vel = -0.03, -0.4, -2.0
        np.testing.assert_allclose(a.observation('A'), b.observation('B'), atol=1e-6)
        self.assertEqual(a.observation('A', version=1).shape, (OBS_DIM_V1,))
        self.assertEqual(a.observation('A').shape, (OBS_DIM_V2,))

    def test_action_mapping_is_mirrored_for_team_b(self):
        action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        a, b = action_to_commands(action, 'A'), action_to_commands(action, 'B')
        self.assertEqual(list(a), [0, 1, 3, 5])
        self.assertEqual(list(b), [7, 6, 4, 2])
        for (ia, ca), (ib, cb) in zip(a.items(), b.items()):
            self.assertEqual(ca, tuple(-c for c in cb))

    def test_old_models_can_use_v1_observation(self):
        env = FoosballEnv(obs_version=1)
        obs, _ = env.reset(seed=0)
        self.assertEqual(obs.shape, (OBS_DIM_V1,))
        self.assertEqual(env.observation_space.shape, (OBS_DIM_V1,))

    def test_heuristic_scores_against_absent_opponent(self):
        for team in ('A', 'B'):
            with self.subTest(team=team):
                env = FoosballEnv(opponent="absent", noise=False, controlled_team=team,
                                  realistic=False)
                agent = HeuristicAgent(env, seed=0)
                goals = 0
                for ep in range(20):
                    env.reset(seed=ep)
                    done = False
                    while not done:
                        _, _, terminated, truncated, info = env.step(agent.predict())
                        done = terminated or truncated
                    goals += info["score"][team] - info["score"]['B' if team == 'A' else 'A']
                self.assertGreaterEqual(goals, 3)   # sin autogoles y marcando

    def test_heuristic_teams_are_balanced(self):
        totals = {}
        for team in ('A', 'B'):
            env = FoosballEnv(controlled_team=team, realistic=False)
            agent = HeuristicAgent(env, seed=1)
            diff = 0
            for ep in range(60):
                env.reset(seed=1000 + ep)
                done = False
                while not done:
                    _, _, terminated, truncated, info = env.step(agent.predict())
                    done = terminated or truncated
                diff += info["score"][team] - info["score"]['B' if team == 'A' else 'A']
            totals[team] = diff / 60
        for team, diff in totals.items():
            with self.subTest(team=team):
                self.assertLess(abs(diff), 0.25)

    def test_stalled_unreachable_ball_is_relaunched(self):
        env = FoosballEnv(noise=False)
        env.reset(seed=0)
        env.state.ball.reset(x=-0.565, y=0.30, vx=0.0, vy=0.0)   # esquina, fuera de alcance
        env.physics.base_params = env.physics.params = \
            type(env.physics.params)(corner_ramp_accel=0.0)
        moved = False
        for _ in range(60):
            env.step(np.zeros(8))
            moved |= env.state.ball.speed > 0.1
        self.assertTrue(moved)

    def test_noisy_trajectories_and_relaunch_are_reproducible(self):
        first, second = FoosballEnv(), FoosballEnv()
        np.testing.assert_array_equal(first.reset(seed=7)[0], second.reset(seed=7)[0])
        rng = np.random.default_rng(8)
        for step in range(150):
            if step == 12:
                for env in (first, second):
                    env.state.ball.vx = env.state.ball.vy = 0.0
            action = rng.uniform(-1, 1, size=8)
            one, two = first.step(action), second.step(action)
            np.testing.assert_array_equal(one[0], two[0])
            self.assertEqual(one[1:4], two[1:4])
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
        for kwargs in ({}, {"controlled_team": "B"}, {"self_play": True}):
            with self.subTest(**kwargs):
                env = FoosballGymEnv(**kwargs)
                try:
                    check_env(env, skip_render_check=True)
                finally:
                    env.close()


class MatchFormatTests(unittest.TestCase):
    def _env_with_goal_chance(self):
        env = FoosballEnv(opponent=None, noise=False, max_steps=400)
        env.reset(seed=0)
        env.state.ball.reset(x=0.52, y=0.0, vx=3.0, vy=0.0)   # tiro a la portería B
        for bar in env.state.bars:                           # sin defensas: pies levantados
            bar.angle = math.pi
        return env

    def test_match_continues_after_goal_and_goalkeeper_serves(self):
        env = self._env_with_goal_chance()
        for _ in range(8):
            _, _, terminated, truncated, info = env.step(np.zeros(8))
            if info["events"]["goal"]:
                break
        self.assertEqual(info["events"]["goal"], 'A')
        self.assertEqual(info["events"]["serve"], 'B')       # saca el que lo ha recibido
        self.assertFalse(terminated)
        self.assertEqual(info["score"], {"A": 1, "B": 0})
        gk = env.state.bars[7]                               # portero de B
        ball = env.state.ball
        self.assertAlmostEqual(ball.speed, 0.0)
        self.assertTrue(0.04 <= gk.bar_x - ball.x <= 0.065)  # delante de su portero
        self.assertLessEqual(abs(ball.y), env.state.field.goal_y_max)
        for _ in range(5):                                   # el partido sigue
            _, _, terminated, _, _ = env.step(np.zeros(8))
            self.assertFalse(terminated)

    def test_old_format_terminates_on_goal(self):
        env = FoosballEnv(opponent=None, noise=False, continue_after_goal=False, max_steps=400)
        env.reset(seed=0)
        env.state.ball.reset(x=0.52, y=0.0, vx=3.0, vy=0.0)
        for bar in env.state.bars:
            bar.angle = math.pi
        for _ in range(8):
            _, _, terminated, _, info = env.step(np.zeros(8))
            if terminated:
                break
        self.assertTrue(terminated)

    def test_serve_teleport_is_not_rewarded_by_shaping(self):
        env = self._env_with_goal_chance()
        phi_before = env._phi
        for _ in range(8):
            _, r, _, _, info = env.step(np.zeros(8))
            if info["events"]["goal"]:
                break
        bd = info["reward_bd"]
        self.assertAlmostEqual(bd["shaping"], -phi_before + 0.0, places=1)   # Φ(terminal) = 0
        self.assertGreater(bd["goal_scored"], 0)
        self.assertAlmostEqual(env._phi, potential(env.state, 'A', env.reward_config))

    def test_serve_does_not_cause_own_goal_chains(self):
        from example import HeuristicAgent
        env = FoosballEnv(opponent_skill=0.7, max_steps=1500)
        agent = HeuristicAgent(env, seed=1)
        env.reset(seed=102)
        goal_steps = []
        for step in range(1500):
            _, _, terminated, truncated, info = env.step(agent.predict())
            if info["events"]["goal"]:
                goal_steps.append(step)
            if terminated or truncated:
                break
        self.assertTrue(all(b - a > 40 for a, b in zip(goal_steps, goal_steps[1:])))

    def test_heuristic_params_roundtrip_and_bounds(self):
        from env import PARAM_BOUNDS, HeuristicParams
        p = HeuristicParams()
        self.assertEqual(HeuristicParams.from_array(p.to_array()), p)
        wild = HeuristicParams.from_array(np.full(len(PARAM_BOUNDS), 1e6))
        for (name, (lo, hi)) in PARAM_BOUNDS.items():
            self.assertAlmostEqual(getattr(wild, name), hi)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "p.json")
            p.save(path)
            self.assertEqual(HeuristicParams.load(path), p)

    def test_benchmark_reports_confidence_interval(self):
        from benchmark import heuristic_policy, play
        r = play(heuristic_policy(), n_matches=4, seed=5, seconds=4.0)
        for key in ("scored", "conceded", "diff", "diff_ci", "win", "loss", "shots"):
            self.assertIn(key, r)
        self.assertTrue(np.isfinite(r["diff_ci"]))

    def test_dagger_rollout_labels_with_teacher(self):
        from dagger import _rollout
        from env import HeuristicParams
        from observation import ACTOR_DIM
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "teacher.json")
            HeuristicParams().save(path)
            o, a, r, (scored, conceded, minutes) = _rollout(
                (np.array([3, 4]), "A", ("heuristic", 0.5), 1.0, None, path, 0.1, 60))
        self.assertEqual(o.shape, (120, 240))
        self.assertEqual(a.shape, (120, 8))
        self.assertEqual(r.shape, (120,))
        self.assertTrue(np.all(np.abs(a) <= 1.0))
        self.assertAlmostEqual(minutes, 2 * 60 * 0.02 / 60.0)

    def test_optimizer_fitness_rewards_goal_difference(self):
        from optimize_teacher import _fitness, _summary
        good = np.array([[2, 0, 3, 0.5], [1, 0, 2, 0.5]], dtype=float)   # marca, recibe, tiros, minutos
        bad = np.array([[0, 2, 3, 0.5], [0, 1, 2, 0.5]], dtype=float)
        self.assertGreater(_fitness(good), _fitness(bad))
        s = _summary(good)
        self.assertAlmostEqual(s["scored"], 3.0)
        self.assertAlmostEqual(s["conceded"], 0.0)


class RewardTests(unittest.TestCase):
    def test_potential_shaping_telescopes(self):
        cfg = RewardConfig(potential_prox=0.5)
        env = FoosballEnv(reward_config=cfg, noise=False)
        env.reset(seed=3)
        phi0 = env._phi
        rng = np.random.default_rng(0)
        total, discount = 0.0, 1.0
        for _ in range(200):
            _, _, terminated, truncated, info = env.step(rng.uniform(-1, 1, 8))
            total += discount * info["reward_bd"].get("shaping", 0.0)
            discount *= cfg.gamma
            if terminated or truncated:
                break
        # Σ γ^t F_t = γ^T Φ(s_T) − Φ(s_0)
        expected = discount * env._phi - phi0
        self.assertAlmostEqual(total, expected, places=5)

    def test_potential_rewards_ball_progress_and_speed(self):
        state = create_standard_world()
        cfg = RewardConfig()
        state.ball.reset(x=0.3, y=0.0, vx=1.0, vy=0.0)
        self.assertGreater(potential(state, 'A', cfg), 0)
        self.assertLess(potential(state, 'B', cfg), 0)
        self.assertAlmostEqual(potential(state, 'A', cfg), -potential(state, 'B', cfg))

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

    def test_shot_rewards(self):
        state = create_standard_world()
        r_own, _ = compute_reward(state, 'A', {"shot_on_target": 'A'}, None)
        r_opp, _ = compute_reward(state, 'A', {"shot_on_target": 'B'}, None)
        self.assertGreater(r_own, 0)
        self.assertLess(r_opp, 0)

    def test_curriculum_reduces_helpers(self):
        configs = [make_env_curriculum(phase)._env.reward_config for phase in range(3)]
        self.assertGreater(configs[0].potential_prox, configs[1].potential_prox)
        self.assertGreater(configs[1].potential_prox, configs[2].potential_prox)
        self.assertEqual(configs[2].potential_prox, 0.0)
        self.assertEqual(configs[0].touch_reward > 0, True)
        self.assertEqual(configs[2].touch_reward, 0.0)
        self.assertEqual(configs[2], final_reward_config(1.0))
        for cfg in configs:
            with self.subTest(cfg=cfg):
                for legacy in ("proximity_scale", "ball_direction_scale", "ball_in_atk_zone",
                               "ball_progress_scale", "forward_kick_reward", "action_penalty_scale"):
                    self.assertEqual(getattr(cfg, legacy), 0.0)

    def test_final_shaping_scales_potentials(self):
        full, half, none = (final_reward_config(s) for s in (1.0, 0.5, 0.0))
        self.assertAlmostEqual(half.potential_x, full.potential_x / 2)
        self.assertAlmostEqual(half.potential_v, full.potential_v / 2)
        self.assertEqual(none.potential_x, 0.0)
        with self.assertRaises(ValueError):
            make_env_curriculum(2, final_shaping=1.5)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

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
                    {"events": {"touches": [(5, 0), (2, 0)], "shot_on_target": 'A'},
                     "score": {"A": 0, "B": 1}},
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
            self.assertEqual(list(callback._ep_shots), [1, 0])
            self.assertEqual(list(callback._ep_goals_scored), [0, 1])
            self.assertEqual(list(callback._ep_goals_conceded), [1, 0])
            np.testing.assert_array_equal(callback._ep_touches_now, [0, 2])
        finally:
            env.close()

    def test_training_save_load_and_evaluation(self):
        def short_eval(*args, **kwargs):
            kwargs.update(eval_freq=1024, n_eval_episodes=kwargs["n_eval_episodes"])
            return FoosballEvalCallback(*args, **kwargs)

        def short_checkpoint(*args, **kwargs):
            kwargs["save_freq"] = 1024
            return CheckpointCallback(*args, **kwargs)

        import train as train_module
        original_make = train_module.make_env_curriculum

        def short_matches(phase, final_shaping=1.0, **kwargs):
            # Partidos de 6 s: con los de 30 s ningún episodio termina en una prueba tan corta
            kwargs["max_steps"] = 300
            return original_make(phase, final_shaping, **kwargs)

        for curriculum, timesteps in ((False, 4096), (True, 6144)):
            with self.subTest(curriculum=curriculum), tempfile.TemporaryDirectory() as directory:
                save_dir, log_dir = Path(directory) / "models", Path(directory) / "logs"
                with (contextlib.redirect_stdout(io.StringIO()),
                      patch("train.FoosballEvalCallback", side_effect=short_eval),
                      patch("train.make_env_curriculum", side_effect=short_matches),
                      patch("train.CheckpointCallback", side_effect=short_checkpoint)):
                    model = train_ppo(timesteps, n_envs=2, curriculum=curriculum,
                                      save_dir=str(save_dir), log_dir=str(log_dir), seed=7,
                                      phase_fractions=(1, 1, 1) if curriculum else None)
                self.assertGreaterEqual(model.num_timesteps, timesteps)
                self.assertGreater(model._n_updates, 0)
                self.assertGreater(model.lr_schedule(1.0), model.lr_schedule(0.5))
                self.assertGreater(model.lr_schedule(0.5), model.lr_schedule(0.0))
                self.assertAlmostEqual(model.policy.optimizer.param_groups[0]["lr"], 3e-5)
                self.assertTrue(all(torch.isfinite(p).all() for p in model.policy.parameters()))
                path = save_dir / "foosball_ppo_final.zip"
                self.assertTrue((save_dir / "best" / "best_model.zip").is_file())
                self.assertTrue(list(save_dir.glob("foosball_ppo_*_steps.zip")))
                loaded = PPO.load(path)
                obs, _ = FoosballEnv().reset(seed=100)
                action, _ = model.predict(obs, deterministic=True)
                np.testing.assert_array_equal(action, loaded.predict(obs, deterministic=True)[0])
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
