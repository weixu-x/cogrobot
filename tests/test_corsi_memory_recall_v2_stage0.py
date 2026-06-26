from __future__ import annotations

import torch

from corsi.experiments.corsi_memory_recall_v2.stage0 import (
    run_gate2_oracle_recall,
    run_gate3_frozen_probe,
    run_gate4_tiny_overfit,
    stage0_audit,
)


def _config() -> dict[str, int]:
    return {
        "max_sequence_length": 3,
        "num_blocks": 5,
        "eos_token_id": 5,
        "ignore_index": -100,
        "k_samples_per_segment": 3,
        "joint_names": [f"joint{i}" for i in range(7)],
    }


def test_stage0_audit_documents_missing_preexisting_gates():
    audit = stage0_audit()
    assert audit["preexisting_exact_gate_implementation_found"] is False
    assert len(audit["missing_before_stage0_module"]) == 4
    assert "final_memory" in audit["M_definition"]


def test_gate2_oracle_recall_passes_tiny_upper_bound(tmp_path):
    result = run_gate2_oracle_recall(
        config=_config(),
        output_root=tmp_path,
        device=torch.device("cpu"),
        seed=12,
        sequence_count=12,
        max_steps=300,
        learning_rate=1e-2,
        exact_accuracy_threshold=1.0,
    )
    assert result.passed
    assert result.metrics["exact_sequence_accuracy"] == 1.0
    assert "checkpoint" in result.artifacts


def test_gate3_frozen_probe_passes_tiny_memory_diagnostic(tmp_path):
    result = run_gate3_frozen_probe(
        config=_config(),
        output_root=tmp_path,
        device=torch.device("cpu"),
        seed=13,
        sequence_count=16,
        encoder_steps=500,
        probe_steps=400,
        learning_rate=1e-2,
        token_accuracy_threshold=0.98,
        exact_accuracy_threshold=0.90,
    )
    assert result.passed
    assert result.metrics["token_order_accuracy"] >= 0.98
    assert result.metrics["length9_exact_sequence_accuracy"] >= 0.90


def test_gate4_visual_motor_tiny_overfit_reduces_loss(tmp_path):
    result = run_gate4_tiny_overfit(
        config=_config(),
        output_root=tmp_path,
        device=torch.device("cpu"),
        seed=14,
        max_steps=300,
        learning_rate=5e-3,
        loss_threshold=8e-3,
    )
    assert result.passed
    assert result.metrics["final_loss"] <= 8e-3
    assert result.metrics["initial_loss"] > result.metrics["final_loss"]
