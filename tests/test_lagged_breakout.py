import ctypes
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class LaggedBreakoutEncoderTest(unittest.TestCase):
    def test_velocity_reparameterization_preserves_policy_features(self):
        import torch
        from pufferlib.models import DefaultEncoder, LaggedBreakoutEncoder
        torch.manual_seed(123)
        original = DefaultEncoder(190, 64)
        normalized = LaggedBreakoutEncoder(190, 64)
        normalized.load_state_dict(original.state_dict())
        with torch.no_grad():
            normalized.encoder.weight[:, 4:6] /= 60
        observations = torch.randn(512, 190)
        torch.testing.assert_close(normalized(observations), original(observations))
        self.assertEqual(set(original.state_dict()), set(normalized.state_dict()))


class LaggedBreakoutPhysicsTest(unittest.TestCase):
    def test_native_physics_contract(self):
        compiler = shlex.split(os.environ.get("CC", "clang"))
        if not shutil.which(compiler[0]):
            self.skipTest("A C compiler is required")
        if sys.platform == "darwin":
            raylib = Path("/opt/homebrew/opt/raylib")
            if not raylib.exists():
                raylib = ROOT / "raylib-5.5_macos"
            links = ["-framework", "Cocoa", "-framework", "IOKit",
                     "-framework", "CoreVideo", "-framework", "OpenGL"]
        else:
            raylib = ROOT / "raylib-5.5_linux_amd64"
            links = ["-lGL", "-lpthread", "-ldl"]
        if not (raylib / "include/raylib.h").exists():
            self.skipTest("Build an environment first to install raylib")
        with tempfile.TemporaryDirectory(prefix="lagged-breakout-tests-") as temp:
            binary = Path(temp) / "physics_test"
            subprocess.run(compiler + [
                "-std=gnu11", "-O2", "-Wall", "-Wextra",
                "-I", str(ROOT), "-I", str(raylib / "include"),
                str(ROOT / "tests/lagged_breakout_test.c"),
                str(raylib / "lib/libraylib.a"), "-lm", *links, "-o", str(binary),
            ], check=True, capture_output=True, text=True)
            result = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            print(result.stdout)


class LaggedBreakoutVectorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from pufferlib import _C
        except ImportError as error:
            raise unittest.SkipTest("Build lagged_breakout first") from error
        if _C.env_name != "lagged_breakout":
            raise unittest.SkipTest("Build lagged_breakout first")
        cls.backend = _C

    def config(self, seed=73):
        from pufferlib.pufferl import load_config
        with mock.patch.object(sys, "argv", [sys.argv[0]]):
            config = load_config("lagged_breakout")
        config["vec"].update(total_agents=8, num_buffers=1, num_threads=1)
        config["env"].update(action_lag=7, volatility=6.0, seed=seed)
        return config

    def observations(self, vec):
        array = (ctypes.c_float * (vec.total_agents * vec.obs_size)).from_address(vec.obs_ptr)
        return np.ctypeslib.as_array(array).reshape(vec.total_agents, vec.obs_size)

    def test_vector_layout_seeds_and_reset(self):
        first = self.backend.create_vec(self.config(), 0)
        second = self.backend.create_vec(self.config(), 0)
        different = self.backend.create_vec(self.config(74), 0)
        try:
            self.assertEqual(first.obs_size, 190)
            self.assertEqual(first.act_sizes, [3])
            self.assertEqual(first.obs_dtype, "FloatTensor")
            for vec in (first, second, different):
                vec.reset()
            actions = np.full((8, 1), 2, dtype=np.float32)
            for _ in range(50):
                for vec in (first, second, different):
                    vec.cpu_step(actions.ctypes.data)
                np.testing.assert_array_equal(self.observations(first), self.observations(second))
            self.assertFalse(np.array_equal(self.observations(first), self.observations(different)))
            self.assertFalse(np.array_equal(self.observations(first)[0], self.observations(first)[1]))
            first.reset()
            np.testing.assert_array_equal(self.observations(first)[:, 126:], 0)
            self.assertTrue(np.isfinite(self.observations(first)).all())
        finally:
            for vec in (first, second, different):
                vec.close()

    def test_invalid_native_config_is_rejected_in_release_build(self):
        for key, value in (("action_lag", -1), ("action_lag", 65),
                           ("action_lag", 1.5), ("volatility", -1),
                           ("frameskip", 0), ("reward_interval", 3)):
            code = (
                "from pufferlib import _C; "
                "config = {'vec': {'total_agents': 1, 'num_buffers': 1}, "
                f"'env': {{{key!r}: {value!r}}}}}; "
                "_C.create_vec(config, 0)"
            )
            result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0, key)
            self.assertIn(key, result.stderr)


if __name__ == "__main__":
    unittest.main()
