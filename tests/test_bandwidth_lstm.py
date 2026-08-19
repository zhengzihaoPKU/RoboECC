from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

try:
    import torch

    from roboecc.bandwidth_lstm import (
        BandwidthLSTM,
        LSTMConfig,
        TrainingConfig,
        TrainedBandwidthPredictor,
        train_predictor,
    )

    HAS_TORCH = True
except ModuleNotFoundError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "PyTorch optional dependency is not installed")
class BandwidthLSTMTests(unittest.TestCase):
    def test_forward_shape(self) -> None:
        model = BandwidthLSTM(LSTMConfig(sequence_length=5, hidden_size=4))
        self.assertEqual(tuple(model(torch.ones(3, 5)).shape), (3,))

    def test_train_save_load_predict(self) -> None:
        values = [50.0 + 8.0 * math.sin(index / 6.0) for index in range(80)]
        result = train_predictor(
            values,
            model_config=LSTMConfig(sequence_length=6, hidden_size=8),
            training_config=TrainingConfig(
                epochs=4,
                batch_size=16,
                patience=4,
                seed=3,
            ),
            device="cpu",
        )
        prediction = result.predictor.predict(values)
        self.assertGreater(prediction, 0.0)
        self.assertEqual(len(result.train_loss), 4)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "bandwidth_lstm.pt"
            result.predictor.save(checkpoint, metadata={"test": True})
            loaded = TrainedBandwidthPredictor.load(checkpoint)
            self.assertAlmostEqual(loaded.predict(values), prediction, places=5)


if __name__ == "__main__":
    unittest.main()

