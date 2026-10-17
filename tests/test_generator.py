from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from rag.generator import (
    MIN_GENERATION_TRAINING_STEPS,
    NO_TRAINED_CHECKPOINT_MESSAGE,
    TrustLMGenerator,
)
from verification.verifier import Verifier


class TestTrustLMGenerator(unittest.TestCase):
    def test_missing_checkpoint_reports_status_without_random_model(self):
        with patch.object(TrustLMGenerator, "_find_checkpoint", return_value=None):
            generator = TrustLMGenerator(max_new_tokens=1)

        self.assertEqual(
            generator.generate("Answer using the evidence."),
            NO_TRAINED_CHECKPOINT_MESSAGE,
        )
        self.assertIsNone(generator._model)
        self.assertFalse(generator.is_available)

    def test_smoke_test_checkpoint_is_not_used_for_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "smoke-test.pt"
            checkpoint_path.touch()
            generator = TrustLMGenerator(checkpoint_path=checkpoint_path)

            with patch(
                "torch.load",
                return_value={"global_step": MIN_GENERATION_TRAINING_STEPS - 1},
            ):
                answer = generator.generate("Answer using the evidence.")

        self.assertEqual(answer, NO_TRAINED_CHECKPOINT_MESSAGE)
        self.assertIsNone(generator._model)

    def test_trained_checkpoint_loads_local_model_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "trained.pt"
            checkpoint_path.touch()
            checkpoint = {
                "global_step": MIN_GENERATION_TRAINING_STEPS,
                "model_config": {},
                "model_state_dict": {"weight": torch.tensor([1.0])},
            }
            local_model = Mock()
            with (
                patch("torch.load", return_value=checkpoint),
                patch("model.trustlm.TrustLM", return_value=local_model),
            ):
                generator = TrustLMGenerator(checkpoint_path=checkpoint_path)
                generator._load_model()

        local_model.load_state_dict.assert_called_once_with(checkpoint["model_state_dict"])
        local_model.eval.assert_called_once()
        self.assertTrue(generator.is_available)

    def test_verifier_does_not_use_external_service(self):
        result = Verifier().verify("an answer", "evidence")

        self.assertFalse(result["enabled"])
        self.assertEqual(result["claims"], [])
        self.assertIn("not implemented", result["reason"])


if __name__ == "__main__":
    unittest.main()
