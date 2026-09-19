import copy
import unittest
from unittest import mock

import torch

from pufferlib import _C
from pufferlib import models
from pufferlib.muon import Muon
from pufferlib.torch_pufferl import (
    _actions_for_vec_step,
    _compute_puff_advantage_torch,
    _mps_is_available,
    _resolve_device,
    compute_puff_advantage,
    Profile,
)


class DeviceSelectionTest(unittest.TestCase):
    def test_explicit_cpu(self):
        self.assertEqual(_resolve_device('cpu'), torch.device('cpu'))

    def test_auto_prefers_mps_when_cuda_is_unavailable(self):
        with mock.patch('torch.cuda.is_available', return_value=False), \
                mock.patch(
                    'pufferlib.torch_pufferl._mps_is_available',
                    return_value=True,
                ):
            self.assertEqual(_resolve_device('auto'), torch.device('mps'))

    def test_unavailable_mps_has_actionable_error(self):
        with mock.patch(
                'pufferlib.torch_pufferl._mps_is_available',
                return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'MPS was requested'):
                _resolve_device('mps')


class AdvantageParityTest(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(9182)
        shape = (11, 17)
        self.values = torch.randn(shape, generator=generator)
        self.rewards = torch.randn(shape, generator=generator)
        self.terminals = (
            torch.rand(shape, generator=generator) < 0.15
        ).float()
        self.ratio = torch.rand(shape, generator=generator) * 2.5
        self.kwargs = {
            'gamma': 0.995,
            'gae_lambda': 0.91,
            'vtrace_rho_clip': 1.0,
            'vtrace_c_clip': 0.8,
        }

    def test_torch_implementation_matches_native_cpu_baseline(self):
        if not hasattr(_C, 'puff_advantage_cpu'):
            self.skipTest('the loaded native backend has no CPU advantage kernel')
        native = torch.zeros_like(self.values)
        compute_puff_advantage(
            self.values, self.rewards, self.terminals, self.ratio, native,
            **self.kwargs)

        portable = torch.zeros_like(self.values)
        _compute_puff_advantage_torch(
            self.values, self.rewards, self.terminals, self.ratio, portable,
            **self.kwargs)

        torch.testing.assert_close(portable, native, atol=2e-6, rtol=2e-6)

    @unittest.skipUnless(_mps_is_available(), 'MPS is not available')
    def test_mps_matches_native_cpu_baseline(self):
        native = torch.zeros_like(self.values)
        compute_puff_advantage(
            self.values, self.rewards, self.terminals, self.ratio, native,
            **self.kwargs)

        device = torch.device('mps')
        mps = torch.zeros_like(self.values, device=device)
        compute_puff_advantage(
            self.values.to(device), self.rewards.to(device),
            self.terminals.to(device), self.ratio.to(device), mps,
            **self.kwargs)

        torch.mps.synchronize()
        torch.testing.assert_close(mps.cpu(), native, atol=2e-5, rtol=2e-5)


class MPSPolicyParityTest(unittest.TestCase):
    @unittest.skipUnless(_mps_is_available(), 'MPS is not available')
    def test_mps_profiler_completes(self):
        profile = Profile('mps')
        profile.mark(0)
        value = torch.ones(256, device='mps').square().sum()
        profile.mark(1)
        profile.elapsed(Profile.TRAIN, 0, 1)
        self.assertEqual(value.item(), 256.0)
        self.assertGreaterEqual(profile.read_and_reset()[Profile.TRAIN], 0.0)

    @unittest.skipUnless(_mps_is_available(), 'MPS is not available')
    def test_training_step_matches_cpu_baseline(self):
        torch.manual_seed(2026)
        cpu_policy = models.Policy(
            models.DefaultEncoder(7, 16),
            models.DefaultDecoder([3, 4], 16),
            models.MinGRU(16, num_layers=2),
        )
        mps_policy = copy.deepcopy(cpu_policy).to('mps')
        observations = torch.randn(4, 6, 7)

        cpu_logits, cpu_values = cpu_policy(observations)
        mps_logits, mps_values = mps_policy(observations.to('mps'))
        for actual, expected in zip(mps_logits, cpu_logits):
            torch.testing.assert_close(
                actual.cpu(), expected, atol=3e-4, rtol=3e-4)
        torch.testing.assert_close(
            mps_values.cpu(), cpu_values, atol=3e-4, rtol=3e-4)

        cpu_loss = (
            sum(x.square().mean() for x in cpu_logits)
            + cpu_values.square().mean()
        )
        mps_loss = (
            sum(x.square().mean() for x in mps_logits)
            + mps_values.square().mean()
        )
        cpu_loss.backward()
        mps_loss.backward()

        cpu_optimizer = Muon(cpu_policy.parameters(), lr=1e-3, momentum=0.9)
        mps_optimizer = Muon(mps_policy.parameters(), lr=1e-3, momentum=0.9)
        cpu_optimizer.step()
        mps_optimizer.step()
        torch.mps.synchronize()

        for mps_param, cpu_param in zip(
                mps_policy.parameters(), cpu_policy.parameters()):
            torch.testing.assert_close(
                mps_param.cpu(), cpu_param, atol=6e-4, rtol=6e-4)

    @unittest.skipUnless(_mps_is_available(), 'MPS is not available')
    def test_actions_are_copied_to_cpu_for_cpu_vector_backend(self):
        action = torch.tensor([1, 2, 3], device='mps')
        cpu_action = _actions_for_vec_step(action, device='cpu')
        self.assertEqual(cpu_action.device.type, 'cpu')
        self.assertEqual(cpu_action.dtype, torch.float32)
        torch.testing.assert_close(
            cpu_action, torch.tensor([[1.0], [2.0], [3.0]]))


if __name__ == '__main__':
    unittest.main()
