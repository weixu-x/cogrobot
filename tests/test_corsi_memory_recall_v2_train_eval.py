import pytest
import torch

from corsi.experiments.corsi_memory_recall_v2.analysis import (
    apply_presentation_order_shuffle,
    compute_sequence_metrics,
    run_causal_sanity_checks,
)
from corsi.experiments.corsi_memory_recall_v2.train import (
    CHECKPOINT_VERSION,
    load_checkpoint,
    save_checkpoint,
    validate_training_scope,
)


def test_sequence_metrics_cover_masks_lengths_eos_and_taxonomy():
    targets = torch.tensor(
        [
            [0, 1, 2, 9, -100],
            [3, 4, 9, -100, -100],
            [5, 6, 7, 9, -100],
            [1, 2, 9, -100, -100],
            [4, 5, 9, -100, -100],
        ]
    )
    mask = targets.ne(-100)
    predictions = torch.tensor(
        [
            [0, 1, 2, 9, 8],
            [3, 8, 4, 9, 0],
            [5, 7, 6, 9, 0],
            [1, 9, 0, 0, 0],
            [4, 8, 9, 0, 0],
        ]
    )
    metrics = compute_sequence_metrics(predictions, targets, mask, eos_token_id=9, ignore_index=-100)
    assert metrics["sequence_count"] == 5
    assert metrics["token_count"] == 17
    assert metrics["full_sequence_accuracy"] == pytest.approx(0.2)
    assert metrics["eos_accuracy"] == pytest.approx(0.6)
    assert metrics["predicted_length_accuracy"] == pytest.approx(0.6)
    assert metrics["per_length_accuracy"]["2"]["count"] == 3
    assert metrics["serial_position_accuracy"]["0"]["accuracy"] == pytest.approx(1.0)
    assert metrics["error_taxonomy"]["substitution"] >= 1
    assert metrics["error_taxonomy"]["omission"] >= 1
    assert metrics["error_taxonomy"]["insertion"] >= 1
    assert metrics["error_taxonomy"]["transposition"] >= 1


def test_presentation_order_shuffle_reverses_only_model_inputs():
    batch = {
        "model_inputs": {
            "images": torch.arange(2 * 3 * 1).reshape(2, 3, 1),
            "segment_mask": torch.tensor([[True, True, False], [True, True, True]]),
            "frame_mask": torch.arange(2 * 3 * 2).reshape(2, 3, 2).bool(),
        },
        "targets": {"tokens": torch.tensor([[0, 1, 9], [2, 3, 9]])},
    }
    shuffled = apply_presentation_order_shuffle(batch)
    torch.testing.assert_close(shuffled["model_inputs"]["images"], batch["model_inputs"]["images"].flip(1))
    torch.testing.assert_close(
        shuffled["model_inputs"]["segment_mask"], batch["model_inputs"]["segment_mask"].flip(1)
    )
    torch.testing.assert_close(shuffled["targets"]["tokens"], batch["targets"]["tokens"])
    assert not torch.equal(shuffled["model_inputs"]["images"], batch["model_inputs"]["images"])


class FakeRecallModel:
    def __init__(self):
        self.calls = []

    def decode(self, model_inputs, intervention=None):
        self.calls.append(intervention)
        batch = model_inputs["segment_mask"].shape[0]
        if intervention == "memory_zero":
            row = torch.tensor([8, 8, 9])
        elif intervention == "memory_shuffle":
            row = torch.tensor([2, 1, 9])
        elif intervention == "presentation_order_shuffle":
            row = torch.tensor([1, 0, 9])
        else:
            row = torch.tensor([0, 1, 9])
        return row.repeat(batch, 1)


def test_causal_sanity_check_mechanics_with_fake_model_hooks():
    batch = {
        "model_inputs": {
            "images": torch.zeros(2, 2, 1, 1, 1, 1),
            "segment_mask": torch.ones(2, 2, dtype=torch.bool),
            "frame_mask": torch.ones(2, 2, 1, dtype=torch.bool),
        },
        "targets": {
            "tokens": torch.tensor([[0, 1, 9], [0, 1, 9]]),
            "token_mask": torch.ones(2, 3, dtype=torch.bool),
        },
    }
    model = FakeRecallModel()
    checks = run_causal_sanity_checks(model, batch, eos_token_id=9)
    assert checks["normal"]["full_sequence_accuracy"] == pytest.approx(1.0)
    assert checks["memory_zero"]["full_sequence_accuracy_delta"] > 0.0
    assert checks["memory_shuffle"]["token_accuracy_delta"] > 0.0
    assert checks["presentation_order_shuffle"]["decoded_order_changed"] is True
    assert model.calls == [None, "memory_zero", "memory_shuffle", "presentation_order_shuffle"]


def test_checkpoint_payload_contains_resume_metadata(tmp_path):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    manifest = {
        "schema_version": "scala_corsi_memory_recall_v2_canonical_v1",
        "canonical_fingerprint": "abc123",
        "source_manifest_sha256": "raw456",
        "samples": [{"seq_id": "a"}],
        "split": {"train": ["a"], "val": [], "test": []},
        "normalization": {"source": "train_only"},
    }
    path = tmp_path / "checkpoint.pt"
    payload = save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scaler=None,
        epoch=2,
        stage=2,
        best_metric=0.75,
        config={"learning_rate": 0.01},
        manifest=manifest,
        seed=13,
    )
    assert payload["checkpoint_version"] == CHECKPOINT_VERSION
    loaded = load_checkpoint(path, model=torch.nn.Linear(3, 2), map_location="cpu")
    assert loaded["optimizer_state_dict"] is not None
    assert loaded["scaler_state_dict"] is None
    assert set(loaded["rng_state"]) == {"torch", "cuda", "numpy", "python"}
    assert loaded["fingerprint_state"]["canonical_fingerprint"] == "abc123"
    assert loaded["fingerprint_state"]["split_counts"] == {"train": 1, "val": 0, "test": 0}
    assert loaded["config"]["learning_rate"] == 0.01


def test_training_scope_refuses_full_training_by_default():
    with pytest.raises(ValueError, match="Refusing full V2 training"):
        validate_training_scope(max_epochs=30, overfit_episodes=0)
    validate_training_scope(max_epochs=3, overfit_episodes=0)
    validate_training_scope(max_epochs=30, overfit_episodes=2)
    validate_training_scope(max_epochs=30, overfit_episodes=0, allow_full_training=True)
