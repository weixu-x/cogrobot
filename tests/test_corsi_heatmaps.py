from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


class HeatmapUtilityTests(unittest.TestCase):
    def test_gaussian_heatmap_generation(self):
        from corsi.heatmaps import target_heatmap_for_block

        heatmap = target_heatmap_for_block(0, size=32, sigma=2.0, normalize=True)

        self.assertEqual(tuple(heatmap.shape), (32, 32))
        self.assertAlmostEqual(float(heatmap.sum()), 1.0, places=5)
        self.assertGreater(float(heatmap.max()), float(heatmap.mean()))

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_heatmap_loss_respects_mask(self):
        from corsi.heatmaps import spatial_heatmap_loss

        logits = torch.zeros(1, 2, 1, 4, 4, requires_grad=True)
        targets = torch.zeros(1, 2, 4, 4)
        targets[0, 0, 1, 2] = 1.0
        targets[0, 1, 3, 3] = 1.0
        mask = torch.tensor([[True, False]])

        loss = spatial_heatmap_loss(logits, targets, mask, loss_type="spatial_ce")
        loss.backward()

        self.assertGreater(float(loss.item()), 0.0)
        self.assertGreater(float(logits.grad[0, 0].abs().sum().item()), 0.0)
        self.assertEqual(float(logits.grad[0, 1].abs().sum().item()), 0.0)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_argmax_decode(self):
        from corsi.heatmaps import decode_heatmap_argmax

        logits = torch.zeros(1, 2, 1, 4, 5)
        logits[0, 0, 0, 2, 3] = 10.0
        logits[0, 1, 0, 1, 4] = 7.0

        xy = decode_heatmap_argmax(logits)

        self.assertEqual(xy.tolist(), [[[3, 2], [4, 1]]])

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_nearest_block_decode(self):
        from corsi.heatmaps import nearest_block_decode, standard_block_heatmap_xy

        centers = torch.tensor(standard_block_heatmap_xy(32), dtype=torch.float32)
        xy = centers[[0, 4, 8]].reshape(1, 3, 2)

        indices, distances = nearest_block_decode(xy, block_xy=centers)

        self.assertEqual(indices.tolist(), [[0, 4, 8]])
        self.assertTrue(torch.allclose(distances, torch.zeros_like(distances), atol=1e-6))

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_one_batch_heatmap_training_run(self):
        from corsi.heatmaps import sequence_to_target_heatmaps, spatial_heatmap_loss
        from corsi.models.lstm_visual import VisualLSTMConfig, VisualSeq2SeqLSTM

        torch.manual_seed(7)
        config = VisualLSTMConfig(
            cnn_feature_dim=8,
            token_embedding_dim=4,
            hidden_dim=16,
            input_image_size=16,
            output_type="heatmap",
            use_attention=False,
            use_step_embedding=True,
            max_decode_steps=3,
            step_embedding_dim=4,
            heatmap_size=16,
        )
        model = VisualSeq2SeqLSTM(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        frames = torch.rand(2, 3, 3, 16, 16)
        lengths = torch.tensor([3, 2], dtype=torch.long)
        targets = torch.tensor([[0, 4, 8], [3, 5, -100]], dtype=torch.long)
        target_heatmaps = torch.tensor(
            np.stack(
                [
                sequence_to_target_heatmaps([0, 4, 8], size=16, sigma=1.5, normalize=True),
                sequence_to_target_heatmaps([3, 5, 0], size=16, sigma=1.5, normalize=True),
                ],
                axis=0,
            ),
            dtype=torch.float32,
        )
        mask = torch.tensor([[True, True, True], [True, True, False]])

        outputs = model(frames, lengths, targets)
        loss = spatial_heatmap_loss(outputs["heatmap_logits"], target_heatmaps, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.assertIsNone(outputs["logits"])
        self.assertEqual(tuple(outputs["heatmap_logits"].shape), (2, 3, 1, 16, 16))
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
