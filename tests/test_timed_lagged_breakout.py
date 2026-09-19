"""Wall-clock stopping tests; no CUDA or Modal account needed."""

from types import SimpleNamespace
import unittest

from scripts.train_lagged_breakout_timed import run_for_seconds


class TimedTrainingTest(unittest.TestCase):
    def trainer(self):
        self.now = 0.0
        self.rates = []
        trainer = SimpleNamespace(epoch=0, optimizer=SimpleNamespace(param_groups=[{"lr": 0}]))
        trainer.rollouts = lambda: None

        def train():
            self.now += 3
            trainer.epoch += 1
            self.rates.append(trainer.optimizer.param_groups[0]["lr"])

        trainer.train = train
        return trainer

    def test_deadline_finishes_update_without_extra_iteration(self):
        trainer = self.trainer()
        reports = []
        result = run_for_seconds(trainer, 5, 0.01, lambda *args: reports.append(args), clock=lambda: self.now)
        self.assertEqual(result, {"seconds": 6, "iterations": 2, "status": "completed"})
        self.assertEqual(len(reports), 1)
        self.assertEqual(self.rates[0], 0.01)
        self.assertLess(self.rates[1], self.rates[0])
        self.assertGreaterEqual(self.rates[1], 0.001)

    def test_interrupt_stops_after_current_update(self):
        trainer = self.trainer()
        result = run_for_seconds(trainer, 300, 0.01, lambda *args: None,
                                 stopped=lambda: trainer.epoch == 1, clock=lambda: self.now)
        self.assertEqual(result, {"seconds": 3, "iterations": 1, "status": "interrupted"})


if __name__ == "__main__":
    unittest.main()
