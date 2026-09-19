"""Local launcher checks; no Modal authentication or remote compute required."""

import importlib.util
from pathlib import Path
import unittest

MODAL_AVAILABLE = importlib.util.find_spec("modal") is not None
if MODAL_AVAILABLE:
    import modal
    from scripts import modal_lagged_breakout as launcher


@unittest.skipUnless(MODAL_AVAILABLE, "Install the optional Modal client to test the launcher")
class ModalLauncherTest(unittest.TestCase):
    def command(self, **changes):
        arguments = dict(
            run_name="trial_01", lags="0,2,4,8", seeds="42,43", timesteps=1024,
            agents=64, horizon=16, minibatch_size=1024, volatility=6.0,
            eval_episodes=1, max_eval_steps=64,
        )
        arguments.update(changes)
        return launcher.experiment_command(**arguments)

    def test_upload_has_training_sources_but_no_local_state(self):
        included = modal.FilePatternMatcher(*launcher.SOURCE_PATTERNS)
        for path in (
            "pufferlib/torch_pufferl.py", "pufferlib/models.py", "pufferlib/muon.py",
            "src/bindings.cu", "src/vecenv.h", "vendor/ini.c",
            "ocean/breakout/breakout.h", "ocean/lagged_breakout/binding.c",
            "ocean/lagged_breakout/lagged_breakout.h", "config/default.ini",
            "config/lagged_breakout.ini", "scripts/compare_lagged_breakout.py",
            "scripts/build_modal_lagged_breakout.sh",
            "scripts/train_lagged_breakout_timed.py",
        ):
            with self.subTest(path=path):
                self.assertTrue((launcher.ROOT / path).is_file())
                self.assertTrue(included(Path(path)))
        for path in (
            ".env", ".git/config", ".venv/bin/python", ".venv-modal/bin/python",
            "pufferlib/_C.cpython-312-darwin.so", "pufferlib/__pycache__/models.pyc",
            "build/bindings.o", "benchmarks/lagged_breakout/trial/policy.bin",
            "resources/breakout/breakout_weights.bin", "logs/training.log",
        ):
            with self.subTest(path=path):
                self.assertFalse(included(Path(path)))

    def test_rejects_paths_in_run_name(self):
        for value in ("", ".", "..", "../existing", "/tmp/run", "run/name", "run name"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "run-name"):
                self.command(run_name=value)

    def test_rejects_invalid_experiment_inputs(self):
        for changes in (
            {"lags": "0,0"}, {"lags": "65"}, {"lags": ""}, {"lags": "x"},
            {"seeds": "-1"}, {"seeds": "2147483648"}, {"seeds": "42,42"},
            {"timesteps": 0}, {"agents": 0}, {"horizon": 1},
            {"minibatch_size": 1000}, {"minibatch_size": 2048},
            {"volatility": float("nan")}, {"volatility": float("inf")},
            {"volatility": -1}, {"max_eval_steps": 0},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.command(**changes)

    def test_command_preserves_explicit_cuda_and_volume_destination(self):
        command = self.command(lags="0, 4", seeds="42, 44")
        self.assertEqual(command[command.index("--device") + 1], "cuda")
        self.assertEqual(command[command.index("--output") + 1], "/results/trial_01")
        self.assertEqual(command[command.index("--lags") + 1:command.index("--seeds")], ["0", "4"])
        self.assertEqual(command[command.index("--seeds") + 1:command.index("--timesteps")], ["42", "44"])

    def timed(self, **changes):
        arguments = dict(run_name="longer", source_run="parent", lags="0,2,4,8", seeds="42",
                         seconds=300, agents=4096, horizon=64, minibatch_size=16384,
                         volatility=6, learning_rate=0.005, threads=4, autotune=False)
        arguments.update(changes)
        return launcher.timed_commands(**arguments)

    def test_parallel_jobs_have_distinct_outputs_and_matching_parents(self):
        jobs = self.timed()
        self.assertEqual(len(jobs), 4)
        self.assertEqual(len({job[2] for job in jobs}), 4)
        for lag, (name, command, relative) in zip((0, 2, 4, 8), jobs):
            self.assertEqual(relative, f"longer/lag_{lag}/seed_42")
            self.assertEqual(command[command.index("--source") + 1], f"/results/parent/lag_{lag}/seed_42")
            self.assertEqual(command[command.index("--seconds") + 1], "300")

    def test_timed_jobs_reject_invalid_budget_and_parent(self):
        for changes in ({"seconds": -1}, {"seconds": float("nan")}, {"seconds": 3601},
                        {"source_run": "../parent"}, {"source_run": "longer"},
                        {"learning_rate": 0}, {"threads": 5}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.timed(**changes)


if __name__ == "__main__":
    unittest.main()
