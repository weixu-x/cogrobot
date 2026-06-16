from __future__ import annotations

import math
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class AttentionModuleTests(unittest.TestCase):
    def test_attention_shapes_and_padding(self):
        from corsi.models.attention import AttentionConfig, AttentionModule

        module = AttentionModule(AttentionConfig(), hidden_dim=8, attention_dim=4)
        decoder_state = torch.randn(2, 8)
        encoder_outputs = torch.randn(2, 5, 8)
        encoder_mask = torch.tensor([[True, True, True, False, False], [True, True, True, True, True]])

        context, weights, diagnostics = module(decoder_state, encoder_outputs, encoder_mask, step_index=1)

        self.assertEqual(tuple(context.shape), (2, 8))
        self.assertEqual(tuple(weights.shape), (2, 5))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(2), atol=1e-5))
        self.assertTrue(torch.allclose(weights[0, 3:], torch.zeros(2), atol=1e-6))
        self.assertIn("attention_entropy", diagnostics)

    def test_local_distance_increases_diagonal_mass(self):
        from corsi.models.attention import AttentionConfig, AttentionModule, LocalAttentionConfig

        torch.manual_seed(3)
        decoder_state = torch.randn(2, 8)
        encoder_outputs = torch.randn(2, 6, 8)
        encoder_mask = torch.ones(2, 6, dtype=torch.bool)
        global_module = AttentionModule(AttentionConfig(), hidden_dim=8, attention_dim=4)
        local_module = AttentionModule(
            AttentionConfig(
                attention_type="local_distance",
                local_attention=LocalAttentionConfig(lambda_distance=5.0),
            ),
            hidden_dim=8,
            attention_dim=4,
        )
        local_module.load_state_dict(global_module.state_dict())

        _, global_weights, _ = global_module(decoder_state, encoder_outputs, encoder_mask, step_index=2)
        _, local_weights, _ = local_module(decoder_state, encoder_outputs, encoder_mask, step_index=2)

        global_mass = global_weights[:, 1:4].sum(dim=-1).mean()
        local_mass = local_weights[:, 1:4].sum(dim=-1).mean()
        self.assertGreater(float(local_mass.detach()), float(global_mass.detach()))

    def test_zero_noise_equals_global_attention(self):
        from corsi.models.attention import AttentionConfig, AttentionModule, NoisyAttentionConfig

        torch.manual_seed(5)
        global_module = AttentionModule(AttentionConfig(), hidden_dim=8, attention_dim=4)
        noisy_module = AttentionModule(
            AttentionConfig(
                attention_type="noisy_global",
                noisy_attention=NoisyAttentionConfig(enabled=True, noise_std=0.0),
            ),
            hidden_dim=8,
            attention_dim=4,
        )
        noisy_module.load_state_dict(global_module.state_dict())
        decoder_state = torch.randn(2, 8)
        encoder_outputs = torch.randn(2, 5, 8)
        encoder_mask = torch.ones(2, 5, dtype=torch.bool)

        global_context, global_weights, _ = global_module(decoder_state, encoder_outputs, encoder_mask, step_index=0)
        noisy_context, noisy_weights, _ = noisy_module(decoder_state, encoder_outputs, encoder_mask, step_index=0)

        self.assertTrue(torch.allclose(global_weights, noisy_weights, atol=1e-6))
        self.assertTrue(torch.allclose(global_context, noisy_context, atol=1e-6))

    def test_response_suppression(self):
        from corsi.models.attention import ResponseSuppressionConfig
        from corsi.models.lstm_visual import VisualLSTMConfig, VisualSeq2SeqLSTM

        model = VisualSeq2SeqLSTM(
            VisualLSTMConfig(
                response_suppression=ResponseSuppressionConfig(enabled=True, beta=2.5, hard_mask=False)
            )
        )
        logits = torch.zeros(1, 9)
        previous = torch.zeros(1, 9, dtype=torch.bool)
        previous[0, 3] = True

        suppressed = model._apply_response_suppression(logits, previous)

        self.assertTrue(math.isclose(float(suppressed[0, 3]), -2.5, abs_tol=1e-6))
        self.assertTrue(math.isclose(float(suppressed[0, 2]), 0.0, abs_tol=1e-6))


class ErrorTaxonomyTests(unittest.TestCase):
    def test_repeat_error_ignores_valid_target_repeats(self):
        from corsi.analysis.error_taxonomy import classify_trial

        row = classify_trial([1, 2, 1, 3], [1, 2, 1, 3])

        self.assertEqual(row["repeat_error_count"], 0)

    def test_repeat_error_counts_prediction_excess_over_target(self):
        from corsi.analysis.error_taxonomy import classify_trial

        row = classify_trial([1, 2, 3, 4], [1, 2, 2, 2])

        self.assertEqual(row["repeat_error_count"], 2)


if __name__ == "__main__":
    unittest.main()
