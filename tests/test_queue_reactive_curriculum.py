import copy
import ctypes
import math
import sys
import unittest
from unittest import mock

import numpy as np
import torch

from pufferlib import _C
from pufferlib import models
from pufferlib.pufferl import load_config
from pufferlib.torch_pufferl import (
    _freeze_policy_modules,
    _mps_is_available,
    sample_logits,
)
from scripts.benchmark_queue_alpha_drift import _summary
from scripts.benchmark_queue_constant_alpha_drift import summarize


ACTION_SIZES = (5, 2, 8, 6)


class QueueReactiveAlphaDriftBenchmarkTest(unittest.TestCase):
    def test_alpha_aligned_summary_has_expected_sign_and_standard_error(self):
        result = _summary(
            np.array([1.0, -1.0, 0.1, np.nan]),
            np.array([2.0, -4.0, 9.0, 3.0]),
            threshold=0.25,
        )

        self.assertEqual(result["n"], 2)
        self.assertAlmostEqual(result["alpha_aligned_drift"], 3.0)
        self.assertAlmostEqual(result["standard_error"], 1.0)
        self.assertAlmostEqual(result["unconditional_drift"], 7.0 / 3.0)
        self.assertAlmostEqual(result["alpha_delta_slope"], 3.0)

    def test_constant_alpha_summary_uses_terminal_minus_initial_drifts(self):
        result = summarize(np.array([-1.0, 0.0, 1.0, 2.0]))
        self.assertAlmostEqual(result["mean_drift"], 0.5)
        self.assertAlmostEqual(result["standard_error"], math.sqrt(5.0 / 12.0))
        self.assertEqual(result["minimum"], -1.0)
        self.assertEqual(result["median"], 0.5)
        self.assertEqual(result["maximum"], 2.0)


class QueueReactiveEncoderTest(unittest.TestCase):
    def test_core_encoder_uses_exact_phase_one_features(self):
        torch.manual_seed(101)
        encoder = models.QueueReactiveCoreEncoder(44, 12)
        observations = torch.randn(7, 44)

        expected = encoder.core(
            observations[:, encoder.CORE_OBSERVATION_INDICES].float())
        torch.testing.assert_close(encoder(observations), expected)
        self.assertEqual(
            encoder.CORE_OBSERVATION_INDICES,
            tuple(range(16)) + (18, 21, 25),
        )

        changed = observations.clone()
        changed[:, encoder.EXTRA_OBSERVATION_INDICES] += 1000
        torch.testing.assert_close(encoder(changed), expected)

        self.assertTrue(all(p.requires_grad for p in encoder.core.parameters()))
        self.assertTrue(all(not p.requires_grad for p in encoder.extra.parameters()))

    def test_expanded_encoder_loads_core_checkpoint_without_policy_jump(self):
        torch.manual_seed(202)
        core = models.QueueReactiveCoreEncoder(44, 16)
        expanded = models.QueueReactiveExpandedEncoder(44, 16)
        expanded.load_state_dict(core.state_dict(), strict=True)

        observations = torch.randn(11, 44)
        torch.testing.assert_close(expanded(observations), core(observations))
        self.assertTrue(all(p.requires_grad for p in expanded.parameters()))

        loss = expanded(observations).square().mean()
        loss.backward()
        self.assertIsNotNone(expanded.extra[-1].weight.grad)

    def test_encoder_rejects_wrong_observation_abi(self):
        with self.assertRaisesRegex(ValueError, '44-value observation ABI'):
            models.QueueReactiveCoreEncoder(43, 8)


class QueueReactiveActionTest(unittest.TestCase):
    def test_samples_only_noop_or_valid_market_actions(self):
        torch.manual_seed(303)
        decoder = models.QueueReactiveMarketDecoder(ACTION_SIZES, 8)
        with torch.no_grad():
            decoder.decoder.weight.zero_()
            decoder.decoder.bias.zero_()

        distribution, _ = decoder(torch.zeros(4096, 8))
        actions, _, _ = sample_logits(distribution)
        action_types = set(actions[:, 0].tolist())

        self.assertEqual(action_types, {0, 3})
        self.assertTrue(torch.all(actions[actions[:, 0] == 0, 1:] == 0))
        self.assertTrue(torch.all(actions[actions[:, 0] == 3, 2] < 4))

    def test_noop_logprob_ignores_unused_action_heads(self):
        type_logits = torch.tensor(
            [[0.4, -torch.inf, -torch.inf, -0.2, -torch.inf]],
            requires_grad=True,
        )
        side_logits = torch.tensor([[0.1, -0.3]], requires_grad=True)
        depth_logits = torch.tensor([[0.2, -0.1, 0.7, -0.4]], requires_grad=True)
        size_logits = torch.tensor(
            [[0.3, -0.2, 0.0, 0.1, -0.5, 0.8]], requires_grad=True)
        distribution = models.QueueReactiveMarketDistribution((
            type_logits, side_logits, depth_logits, size_logits,
        ))

        action_a = torch.tensor([[0, 0, 0, 0]])
        action_b = torch.tensor([[0, 1, 3, 5]])
        _, logprob_a, _ = sample_logits(distribution, action_a)
        _, logprob_b, _ = sample_logits(distribution, action_b)
        expected = type_logits.log_softmax(-1)[:, 0]

        torch.testing.assert_close(logprob_a, expected)
        torch.testing.assert_close(logprob_b, expected)
        logprob_a.sum().backward()
        torch.testing.assert_close(side_logits.grad, torch.zeros_like(side_logits))
        torch.testing.assert_close(depth_logits.grad, torch.zeros_like(depth_logits))
        torch.testing.assert_close(size_logits.grad, torch.zeros_like(size_logits))

    def test_market_logprob_and_conditional_entropy_match_baseline(self):
        type_logits = torch.tensor([
            [0.4, -torch.inf, -torch.inf, -0.2, -torch.inf],
            [-0.7, -torch.inf, -torch.inf, 0.9, -torch.inf],
        ])
        side_logits = torch.tensor([[0.1, -0.3], [0.8, -0.6]])
        depth_logits = torch.tensor([
            [0.2, -0.1, 0.7, -0.4],
            [-0.3, 0.2, 0.1, 0.5],
        ])
        size_logits = torch.tensor([
            [0.3, -0.2, 0.0, 0.1, -0.5, 0.8],
            [0.2, 0.7, -0.4, 0.1, -0.2, 0.0],
        ])
        distribution = models.QueueReactiveMarketDistribution((
            type_logits, side_logits, depth_logits, size_logits,
        ))
        actions = torch.tensor([[3, 1, 2, 5], [3, 0, 3, 1]])

        _, logprob, entropy = sample_logits(distribution, actions)
        expected_logprob = sum(
            logits.log_softmax(-1).gather(1, selected[:, None]).squeeze(1)
            for logits, selected in zip(
                distribution.logits, actions.T,
            )
        )
        type_probs = type_logits.softmax(-1)
        expected_entropy = torch.distributions.Categorical(
            logits=type_logits).entropy()
        expected_entropy += type_probs[:, 3] * sum(
            torch.distributions.Categorical(logits=head).entropy()
            for head in (side_logits, depth_logits, size_logits)
        )

        torch.testing.assert_close(logprob, expected_logprob)
        torch.testing.assert_close(entropy, expected_entropy)

    def test_market_decoder_checkpoint_loads_strictly_into_full_decoder(self):
        market = models.QueueReactiveMarketDecoder(ACTION_SIZES, 16)
        full = models.DefaultDecoder(ACTION_SIZES, 16)
        full.load_state_dict(market.state_dict(), strict=True)

        disabled_rows = torch.tensor([1, 2, 4, 11, 12, 13, 14])
        torch.testing.assert_close(
            market.decoder.weight[disabled_rows],
            torch.zeros_like(market.decoder.weight[disabled_rows]),
        )
        torch.testing.assert_close(
            market.decoder.bias[disabled_rows],
            torch.zeros_like(market.decoder.bias[disabled_rows]),
        )


class QueueReactiveFreezeTest(unittest.TestCase):
    def test_freezes_named_modules_only(self):
        policy = models.Policy(
            models.QueueReactiveExpandedEncoder(44, 8),
            models.DefaultDecoder(ACTION_SIZES, 8),
            models.GRU(8, num_layers=1),
        )
        _freeze_policy_modules(policy, 'encoder.core, network')

        for name, parameter in policy.named_parameters():
            if name.startswith('encoder.core.') or name.startswith('network.'):
                self.assertFalse(parameter.requires_grad, name)
            else:
                self.assertTrue(parameter.requires_grad, name)

    def test_unknown_freeze_prefix_has_actionable_error(self):
        policy = models.Policy(
            models.QueueReactiveExpandedEncoder(44, 8),
            models.DefaultDecoder(ACTION_SIZES, 8),
            models.GRU(8, num_layers=1),
        )
        with self.assertRaisesRegex(ValueError, 'matched no policy parameters'):
            _freeze_policy_modules(policy, 'does.not.exist')

    def test_queue_config_selects_phase_one_policy(self):
        with mock.patch.object(sys, 'argv', ['unittest']):
            args = load_config('queue_reactive')
        self.assertEqual(args['torch']['encoder'], 'QueueReactiveCoreEncoder')
        self.assertEqual(args['torch']['decoder'], 'QueueReactiveMarketDecoder')
        self.assertEqual(args['torch']['freeze_modules'], '')
        self.assertEqual(args['env']['market_residual_rests'], 0)
        self.assertEqual(args['env']['use_power_law_impact'], 0)
        self.assertEqual(args['env']['agent_rejection_penalty'], 0.01)
        self.assertEqual(args['env']['alpha_scale'], 2.0)


@unittest.skipUnless(
    getattr(_C, 'env_name', None) == 'queue_reactive',
    'native backend is not built for queue_reactive',
)
class QueueReactiveNativeIOCTest(unittest.TestCase):
    def test_constant_alpha_stays_fixed_and_keeps_observation_abi(self):
        with mock.patch.object(sys, 'argv', ['unittest']):
            args = load_config('queue_reactive')
        args['vec'].update(total_agents=2, num_buffers=1, num_threads=1)
        args['env'].update(
            use_alpha=0,
            use_constant_alpha=1,
            constant_alpha=0.75,
            alpha_scale=2.0,
            use_power_law_impact=0,
            episode_duration_seconds=120.0,
        )
        vec = _C.create_vec(args, 0)
        observations = np.ctypeslib.as_array(
            (ctypes.c_float * (2 * 44)).from_address(vec.obs_ptr),
        ).reshape(2, 44)
        actions = np.zeros((2, 4), dtype=np.float32)
        try:
            vec.reset()
            np.testing.assert_array_equal(observations[:, 21], np.full(2, 0.75))
            for _ in range(20):
                vec.cpu_step(actions.ctypes.data)
            np.testing.assert_array_equal(observations[:, 21], np.full(2, 0.75))
            self.assertEqual(vec.obs_size, 44)
        finally:
            vec.close()

    def test_mid_price_diagnostic_does_not_change_observation_abi(self):
        with mock.patch.object(sys, 'argv', ['unittest']):
            args = load_config('queue_reactive')
        args['vec'].update(total_agents=3, num_buffers=1, num_threads=1)
        vec = _C.create_vec(args, 0)
        diagnostics = np.empty(3, dtype=np.float32)
        try:
            vec.reset()
            self.assertEqual(vec.obs_size, 44)
            self.assertEqual(vec.diagnostic_size, 1)
            vec.diagnostic(diagnostics.ctypes.data)
        finally:
            vec.close()
        np.testing.assert_array_equal(diagnostics, np.full(3, 1519.5))

    def _step_partial_market(self, residual_rests):
        with mock.patch.object(sys, 'argv', ['unittest']):
            args = load_config('queue_reactive')
        args['vec'].update(total_agents=1, num_buffers=1, num_threads=1)
        args['env'].update(
            latency_mu=0.0,
            latency_sigma=0.0,
            latency_lower=0.0,
            latency_upper=0.0,
            event_add_prob=1.0,
            event_cancel_prob=0.0,
            event_trade_prob=0.0,
            event_create_prob=0.0,
            market_residual_rests=int(residual_rests),
        )

        vec = _C.create_vec(args, 0)
        try:
            vec.reset()
            # Buy 3,200 at best ask, whose initial displayed volume is 1,200.
            actions = np.array([[3, 0, 0, 5]], dtype=np.float32)
            vec.cpu_step(actions.ctypes.data)
            obs_buffer = (ctypes.c_float * 44).from_address(vec.obs_ptr)
            return np.ctypeslib.as_array(obs_buffer).copy()
        finally:
            vec.close()

    def test_phase_one_market_residual_is_ioc(self):
        observations = self._step_partial_market(residual_rests=False)
        self.assertGreater(observations[18], 0)  # inventory changed
        self.assertGreater(observations[26], 0)  # aggressive fill occurred
        self.assertEqual(observations[31], 1)    # it was a partial fill
        self.assertEqual(observations[27], 0)    # no active private order
        np.testing.assert_array_equal(observations[40:44], np.zeros(4))

    def test_residual_resting_remains_explicitly_available(self):
        observations = self._step_partial_market(residual_rests=True)
        self.assertEqual(observations[27], 0.5)
        self.assertGreater(observations[41], 0)

    def test_rejected_intervention_receives_immediate_penalty(self):
        with mock.patch.object(sys, 'argv', ['unittest']):
            args = load_config('queue_reactive')
        args['vec'].update(total_agents=1, num_buffers=1, num_threads=1)
        args['env'].update(
            latency_mu=0.0,
            latency_sigma=0.0,
            latency_lower=0.0,
            latency_upper=0.0,
            event_add_prob=1.0,
            event_cancel_prob=0.0,
            event_trade_prob=0.0,
            event_create_prob=0.0,
            inventory_penalty_coef=0.0,
            agent_rejection_penalty=0.125,
        )

        vec = _C.create_vec(args, 0)
        try:
            vec.reset()
            # Cancelling with no private order is deterministically rejected.
            actions = np.array([[2, 0, 0, 0]], dtype=np.float32)
            vec.cpu_step(actions.ctypes.data)
            obs_buffer = (ctypes.c_float * 44).from_address(vec.obs_ptr)
            observations = np.ctypeslib.as_array(obs_buffer).copy()
            reward_buffer = (ctypes.c_float * 1).from_address(vec.rewards_ptr)
            reward = float(np.ctypeslib.as_array(reward_buffer)[0])
        finally:
            vec.close()

        self.assertEqual(observations[30], 1)
        self.assertAlmostEqual(reward, -0.125, places=6)


class QueueReactiveMPSParityTest(unittest.TestCase):
    @unittest.skipUnless(_mps_is_available(), 'MPS is not available')
    def test_conditional_policy_matches_cpu_forward_and_gradients(self):
        torch.manual_seed(404)
        cpu_encoder = models.QueueReactiveCoreEncoder(44, 16)
        cpu_decoder = models.QueueReactiveMarketDecoder(ACTION_SIZES, 16)
        mps_encoder = copy.deepcopy(cpu_encoder).to('mps')
        mps_decoder = copy.deepcopy(cpu_decoder).to('mps')

        observations = torch.randn(32, 44)
        actions = torch.zeros(32, 4, dtype=torch.long)
        actions[1::2, 0] = 3
        actions[1::2, 1] = torch.arange(16) % 2
        actions[1::2, 2] = torch.arange(16) % 4
        actions[1::2, 3] = torch.arange(16) % 6

        cpu_hidden = cpu_encoder(observations)
        cpu_distribution, cpu_values = cpu_decoder(cpu_hidden)
        _, cpu_logprob, cpu_entropy = sample_logits(cpu_distribution, actions)
        cpu_loss = cpu_logprob.mean() + cpu_entropy.mean() + cpu_values.mean()
        cpu_loss.backward()

        mps_hidden = mps_encoder(observations.to('mps'))
        mps_distribution, mps_values = mps_decoder(mps_hidden)
        _, mps_logprob, mps_entropy = sample_logits(
            mps_distribution, actions.to('mps'))
        mps_loss = mps_logprob.mean() + mps_entropy.mean() + mps_values.mean()
        mps_loss.backward()
        torch.mps.synchronize()

        torch.testing.assert_close(mps_hidden.cpu(), cpu_hidden, atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(mps_logprob.cpu(), cpu_logprob, atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(mps_entropy.cpu(), cpu_entropy, atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(mps_values.cpu(), cpu_values, atol=3e-5, rtol=3e-5)

        for mps_parameter, cpu_parameter in zip(
                mps_encoder.parameters(), cpu_encoder.parameters()):
            if cpu_parameter.grad is None:
                self.assertIsNone(mps_parameter.grad)
            else:
                torch.testing.assert_close(
                    mps_parameter.grad.cpu(), cpu_parameter.grad,
                    atol=8e-5, rtol=8e-5,
                )
        for mps_parameter, cpu_parameter in zip(
                mps_decoder.parameters(), cpu_decoder.parameters()):
            torch.testing.assert_close(
                mps_parameter.grad.cpu(), cpu_parameter.grad,
                atol=8e-5, rtol=8e-5,
            )


if __name__ == '__main__':
    unittest.main()
