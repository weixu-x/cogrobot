import inspect

import pytest
import torch

from corsi.experiments.corsi_memory_recall_v2.losses import (
    combined_v2_loss,
    final_memory_length_loss,
    final_memory_order_loss,
    memory_order_prefix_loss,
    sequence_cross_entropy_loss,
    weak_coord_loss,
)
from corsi.experiments.corsi_memory_recall_v2.model import (
    CorsiMemoryRecallV2Config,
    CorsiMemoryRecallV2Model,
)


def _small_config(**overrides):
    values = {
        "cnn_dim": 16,
        "visual_hidden_dim": 12,
        "motor_hidden_dim": 10,
        "item_dim": 14,
        "memory_dim": 8,
        "recall_hidden_dim": 11,
        "recall_token_dim": 6,
        "max_sequence_length": 4,
        "memory_noise_std": 0.0,
    }
    values.update(overrides)
    return CorsiMemoryRecallV2Config(**values)


def _inputs(batch_size=2, segments=3, frames=4, image_size=128):
    torch.manual_seed(7)
    images = torch.randn(batch_size, segments, frames, 3, image_size, image_size)
    segment_mask = torch.ones(batch_size, segments, dtype=torch.bool)
    frame_mask = torch.ones(batch_size, segments, frames, dtype=torch.bool)
    if segments > 2:
        segment_mask[-1, -1] = False
        frame_mask[-1, -1] = False
    return images, segment_mask, frame_mask


def test_forward_shapes_on_segmented_rgb_cpu():
    model = CorsiMemoryRecallV2Model(_small_config())
    images, segment_mask, frame_mask = _inputs(batch_size=2, segments=3, frames=4)

    outputs = model(
        images=images,
        segment_mask=segment_mask,
        frame_mask=frame_mask,
        max_recall_steps=5,
    )

    assert outputs["logits"].shape == (2, 5, 10)
    assert outputs["coord"].shape == (2, 5, 2)
    assert outputs["pred_joint"].shape == (2, 3, 4, 7)
    assert outputs["pred_ee_pose"].shape == (2, 3, 4, 7)
    assert outputs["pred_ee_xy"].shape == (2, 3, 4, 2)
    assert outputs["item_embeddings"].shape == (2, 3, 14)
    assert outputs["final_memory"].shape == (2, 8)
    assert outputs["memory_order_logits"].shape == (2, 3, 4, 9)
    assert outputs["final_memory_order_logits"].shape == (2, 4, 9)
    assert outputs["final_memory_length_logits"].shape == (2, 4)
    assert outputs["pred_joint"][-1, -1].abs().sum().item() == 0.0


def test_cnn_output_dimension_is_configurable():
    model = CorsiMemoryRecallV2Model(_small_config(cnn_dim=10))
    images, segment_mask, frame_mask = _inputs(batch_size=1, segments=1, frames=2)

    outputs = model.encode_presentation(
        images=images,
        segment_mask=segment_mask,
        frame_mask=frame_mask,
    )

    assert outputs["frame_features"].shape == (1, 1, 2, 10)


def test_visual_and_motor_lstm_reset_for_each_segment():
    model = CorsiMemoryRecallV2Model(_small_config())
    model.eval()
    image = torch.randn(1, 1, 4, 3, 128, 128)
    images = image.repeat(1, 2, 1, 1, 1, 1)
    segment_mask = torch.ones(1, 2, dtype=torch.bool)
    frame_mask = torch.ones(1, 2, 4, dtype=torch.bool)

    encoded = model.encode_presentation(
        images=images,
        segment_mask=segment_mask,
        frame_mask=frame_mask,
        return_traces=True,
    )

    torch.testing.assert_close(encoded["item_visual"][:, 0], encoded["item_visual"][:, 1])
    torch.testing.assert_close(encoded["item_motor"][:, 0], encoded["item_motor"][:, 1])
    torch.testing.assert_close(
        encoded["traces"]["visual_h"][:, 0],
        encoded["traces"]["visual_h"][:, 1],
    )
    torch.testing.assert_close(
        encoded["traces"]["motor_h"][:, 0],
        encoded["traces"]["motor_h"][:, 1],
    )


def test_model_contract_excludes_forbidden_inputs_and_motor_gets_visual_hidden_only():
    signature = inspect.signature(CorsiMemoryRecallV2Model.forward)
    assert set(signature.parameters) == {
        "self",
        "images",
        "segment_mask",
        "frame_mask",
        "max_recall_steps",
        "return_traces",
    }
    for forbidden in ["joints", "joint", "tokens", "block_id", "rank", "length", "block_xy"]:
        assert forbidden not in signature.parameters

    model = CorsiMemoryRecallV2Model(_small_config())
    assert model.motor_lstm.input_size == model.config.visual_hidden_dim
    assert model.motor_lstm.input_size != model.config.joint_dim


def test_recall_token_is_constant_at_every_step():
    model = CorsiMemoryRecallV2Model(_small_config())
    recall_inputs = model.get_recall_inputs(batch_size=3, steps=5)

    assert recall_inputs.shape == (3, 5, model.config.recall_token_dim)
    for step in range(1, recall_inputs.shape[1]):
        torch.testing.assert_close(recall_inputs[:, 0], recall_inputs[:, step])

    recall_inputs.sum().backward()
    assert model.recall_token.grad is not None


def test_recall_depends_only_on_final_memory_bridge():
    model = CorsiMemoryRecallV2Model(_small_config())
    model.eval()
    images, segment_mask, frame_mask = _inputs(batch_size=2, segments=3, frames=4)

    outputs = model(
        images=images,
        segment_mask=segment_mask,
        frame_mask=frame_mask,
        max_recall_steps=4,
    )
    recalled = model.recall_from_memory(outputs["final_memory"], max_recall_steps=4)

    torch.testing.assert_close(outputs["logits"], recalled["logits"])
    torch.testing.assert_close(outputs["coord"], recalled["coord"])

    zero_memory = torch.zeros_like(outputs["final_memory"])
    recalled_a = model.recall_from_memory(zero_memory, max_recall_steps=4)
    recalled_b = model.recall_from_memory(zero_memory, max_recall_steps=4)
    torch.testing.assert_close(recalled_a["logits"], recalled_b["logits"])


def test_slot_compress_memory_write_mode_outputs_bottleneck_and_traces():
    model = CorsiMemoryRecallV2Model(_small_config(memory_write_mode="slot_compress", memory_slot_dim=5))
    images, segment_mask, frame_mask = _inputs(batch_size=2, segments=3, frames=4)

    outputs = model(
        images=images,
        segment_mask=segment_mask,
        frame_mask=frame_mask,
        max_recall_steps=5,
        return_traces=True,
    )

    assert outputs["final_memory"].shape == (2, 8)
    assert outputs["memory_order_logits"].shape == (2, 3, 4, 9)
    assert outputs["final_memory_order_logits"].shape == (2, 4, 9)
    assert outputs["final_memory_length_logits"].shape == (2, 4)
    for key in ["h_t", "c_t", "input_gate", "forget_gate", "candidate", "output_gate"]:
        assert outputs["traces"]["memory"][key].shape == (2, 3, 8)

    items = outputs["item_embeddings"][:1, :2].detach()
    mask = torch.ones(1, 2, dtype=torch.bool)
    forward_memory = model.run_memory(items, mask)["final_memory"]
    reversed_memory = model.run_memory(items.flip(dims=[1]), mask)["final_memory"]
    assert not torch.allclose(forward_memory, reversed_memory)


def test_unknown_memory_write_mode_is_rejected():
    model = CorsiMemoryRecallV2Model(_small_config(memory_write_mode="unknown"))
    item_embeddings = torch.zeros(1, 1, model.config.item_dim)
    segment_mask = torch.ones(1, 1, dtype=torch.bool)

    with pytest.raises(ValueError, match="unknown memory_write_mode"):
        model.run_memory(item_embeddings, segment_mask)


def test_sequence_ce_keeps_eos_and_ignores_padding():
    logits = torch.zeros(1, 4, 10)
    targets = torch.tensor([[2, 9, -100, -100]])
    logits[0, 0, 2] = 8.0
    logits[0, 1, 9] = 8.0
    logits[0, 2:, 0] = -1000.0

    baseline = sequence_cross_entropy_loss(logits, targets, ignore_index=-100)
    changed_padding = logits.clone()
    changed_padding[0, 2:, 4] = 1000.0
    torch.testing.assert_close(
        baseline,
        sequence_cross_entropy_loss(changed_padding, targets, ignore_index=-100),
    )

    wrong_eos = logits.clone()
    wrong_eos[0, 1, 9] = -8.0
    wrong_eos[0, 1, 3] = 8.0
    assert sequence_cross_entropy_loss(wrong_eos, targets, ignore_index=-100) > baseline


def test_weak_coord_loss_excludes_eos_and_ignore_steps():
    pred = torch.zeros(1, 3, 2)
    target = torch.zeros(1, 3, 2)
    target[:, 1] = 1000.0
    target[:, 2] = -1000.0
    tokens = torch.tensor([[4, 9, -100]])
    token_mask = torch.tensor([[True, True, False]])

    loss = weak_coord_loss(
        pred,
        target,
        target_tokens=tokens,
        token_mask=token_mask,
        eos_token_id=9,
        ignore_index=-100,
    )

    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_memory_order_prefix_loss_supervises_written_prefix_only():
    logits = torch.zeros(1, 3, 4, 9)
    targets = torch.tensor([[2, 4, 9, -100]])
    token_mask = torch.tensor([[True, True, True, False]])
    segment_mask = torch.tensor([[True, True, False]])
    logits[0, 0, 0, 2] = 8.0
    logits[0, 1, 0, 2] = 8.0
    logits[0, 1, 1, 4] = 8.0

    baseline = memory_order_prefix_loss(
        logits,
        targets,
        segment_mask=segment_mask,
        token_mask=token_mask,
        eos_token_id=9,
        ignore_index=-100,
    )

    changed_invalid = logits.clone()
    changed_invalid[0, 0, 1, 8] = 1000.0
    changed_invalid[0, 1, 2, 8] = 1000.0
    changed_invalid[0, 2, :, 8] = 1000.0
    torch.testing.assert_close(
        baseline,
        memory_order_prefix_loss(
            changed_invalid,
            targets,
            segment_mask=segment_mask,
            token_mask=token_mask,
            eos_token_id=9,
            ignore_index=-100,
        ),
    )

    wrong_written_prefix = logits.clone()
    wrong_written_prefix[0, 1, 1, 4] = -8.0
    wrong_written_prefix[0, 1, 1, 1] = 8.0
    assert (
        memory_order_prefix_loss(
            wrong_written_prefix,
            targets,
            segment_mask=segment_mask,
            token_mask=token_mask,
            eos_token_id=9,
            ignore_index=-100,
        )
        > baseline
    )


def test_memory_order_prefix_loss_balances_serial_position_exposure():
    logits = torch.zeros(1, 2, 2, 3, requires_grad=True)
    targets = torch.tensor([[0, 1]])
    token_mask = torch.tensor([[True, True]])
    segment_mask = torch.tensor([[True, True]])

    loss = memory_order_prefix_loss(
        logits,
        targets,
        segment_mask=segment_mask,
        token_mask=token_mask,
        eos_token_id=9,
        ignore_index=-100,
    )
    loss.backward()

    pos0_target_grad = logits.grad[0, 0, 0, 0] + logits.grad[0, 1, 0, 0]
    pos1_target_grad = logits.grad[0, 1, 1, 1]
    torch.testing.assert_close(pos0_target_grad, pos1_target_grad)


def test_final_memory_order_loss_ignores_eos_and_padding():
    logits = torch.zeros(1, 4, 9)
    targets = torch.tensor([[2, 4, 9, -100]])
    token_mask = torch.tensor([[True, True, True, False]])
    logits[0, 0, 2] = 8.0
    logits[0, 1, 4] = 8.0

    baseline = final_memory_order_loss(
        logits,
        targets,
        token_mask=token_mask,
        eos_token_id=9,
        ignore_index=-100,
    )

    changed_invalid = logits.clone()
    changed_invalid[0, 2, 8] = 1000.0
    changed_invalid[0, 3, 8] = 1000.0
    torch.testing.assert_close(
        baseline,
        final_memory_order_loss(
            changed_invalid,
            targets,
            token_mask=token_mask,
            eos_token_id=9,
            ignore_index=-100,
        ),
    )

    wrong_block = logits.clone()
    wrong_block[0, 1, 4] = -8.0
    wrong_block[0, 1, 1] = 8.0
    assert final_memory_order_loss(
        wrong_block,
        targets,
        token_mask=token_mask,
        eos_token_id=9,
        ignore_index=-100,
    ) > baseline


def test_final_memory_length_loss_counts_block_tokens_only():
    logits = torch.zeros(2, 4)
    targets = torch.tensor([[2, 4, 9, -100], [1, 3, 5, 9]])
    token_mask = torch.tensor([[True, True, True, False], [True, True, True, True]])
    logits[0, 1] = 8.0
    logits[1, 2] = 8.0

    baseline = final_memory_length_loss(
        logits,
        targets,
        token_mask=token_mask,
        eos_token_id=9,
        ignore_index=-100,
    )

    wrong_eos_count = logits.clone()
    wrong_eos_count[0, 2] = 1000.0
    assert final_memory_length_loss(
        wrong_eos_count,
        targets,
        token_mask=token_mask,
        eos_token_id=9,
        ignore_index=-100,
    ) > baseline


def test_combined_loss_uses_auxiliary_masks_and_weights():
    torch.manual_seed(3)
    outputs = {
        "logits": torch.randn(1, 3, 10, requires_grad=True),
        "coord": torch.zeros(1, 3, 2),
        "memory_order_logits": torch.zeros(1, 2, 4, 9, requires_grad=True),
        "final_memory_order_logits": torch.zeros(1, 4, 9, requires_grad=True),
        "final_memory_length_logits": torch.zeros(1, 4, requires_grad=True),
        "pred_joint": torch.zeros(1, 2, 2, 7),
        "pred_ee_pose": torch.zeros(1, 2, 2, 7),
        "pred_ee_xy": torch.zeros(1, 2, 2, 2),
    }
    targets = {
        "tokens": torch.tensor([[1, 9, -100]]),
        "token_mask": torch.tensor([[True, True, False]]),
        "target_xy": torch.zeros(1, 2, 2),
        "joint": torch.zeros(1, 2, 2, 7),
        "ee_pose": torch.zeros(1, 2, 2, 7),
        "ee_xy": torch.zeros(1, 2, 2, 2),
    }
    targets["joint"][:, 1] = 100.0
    frame_mask = torch.tensor([[[True, True], [False, False]]])

    losses = combined_v2_loss(
        outputs,
        targets,
        frame_mask=frame_mask,
        weights={"memory_order_final": 0.5, "memory_length": 0.5},
    )

    assert set(losses) == {
        "seq_loss",
        "memory_order_loss",
        "final_memory_order_loss",
        "final_memory_length_loss",
        "coord_loss",
        "joint_loss",
        "ee_pose_loss",
        "ee_xy_loss",
        "loss",
    }
    assert losses["memory_order_loss"].requires_grad
    assert losses["final_memory_order_loss"].requires_grad
    assert losses["final_memory_length_loss"].requires_grad
    torch.testing.assert_close(losses["coord_loss"], torch.tensor(0.0))
    torch.testing.assert_close(losses["joint_loss"], torch.tensor(0.0))
    assert losses["loss"].requires_grad


def test_trace_outputs_include_memory_gates_and_states():
    model = CorsiMemoryRecallV2Model(_small_config())
    images, segment_mask, frame_mask = _inputs(batch_size=2, segments=3, frames=4)

    outputs = model(
        images=images,
        segment_mask=segment_mask,
        frame_mask=frame_mask,
        max_recall_steps=4,
        return_traces=True,
    )

    traces = outputs["traces"]
    assert traces["presentation"]["visual_h"].shape == (2, 3, 4, model.config.visual_hidden_dim)
    assert traces["presentation"]["motor_h"].shape == (2, 3, 4, model.config.motor_hidden_dim)
    for key in ["h_t", "c_t", "input_gate", "forget_gate", "candidate", "output_gate"]:
        assert traces["memory"][key].shape == (2, 3, model.config.memory_dim)
    for key in ["h_t", "c_t", "input_gate", "forget_gate", "candidate", "output_gate"]:
        assert traces["recall"][key].shape == (2, 4, model.config.recall_hidden_dim)
