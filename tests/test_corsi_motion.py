import numpy as np
import pytest

from corsi.data.freecam_motion_dataset import FreecamMotionDataset
from corsi.data.collate_motion import collate_motion_batch
from corsi.models.sensorimotor import FreecamMotionSensorimotorModel, SensorimotorConfig


def test_freecam_motion_dataset_loads_existing_preview():
    dataset = FreecamMotionDataset("corsi_artifacts/visual_base/datasets/freecam_ee_xy_v1_preview")
    sample = dataset[0]

    assert sample["motion_schema_version"] == "freecam_motion_v1"
    assert sample["frames"].ndim == 4
    assert sample["motion_target_xy"].shape[-1] == 2
    assert sample["motion_delta_xy"].shape == sample["motion_target_xy"].shape
    assert len(sample["trajectory_metadata"]) > 0
    assert sample["trajectory_ee_xy_norm"].shape[-1] == 2


def test_freecam_motion_dataset_can_require_new_joint_fields():
    dataset = FreecamMotionDataset(
        "corsi_artifacts/visual_base/datasets/freecam_ee_xy_v1_preview",
        require_joint_state=True,
    )
    with pytest.raises(ValueError, match="arm_joint_qpos"):
        _ = dataset[0]


def test_collate_motion_batch_and_model_shape():
    dataset = FreecamMotionDataset("corsi_artifacts/visual_base/datasets/freecam_ee_xy_v1_preview")
    batch = collate_motion_batch([dataset[0], dataset[1]])
    model = FreecamMotionSensorimotorModel(
        SensorimotorConfig(
            input_image_size=64,
            cnn_feature_dim=16,
            hidden_dim=16,
            proprio_dim=0,
        )
    )
    outputs = model(batch["frames"], batch["frame_lengths"])

    assert outputs["pred_motion_xy"].shape[:2] == batch["frames"].shape[:2]
    assert outputs["pred_motion_xy"].shape[-1] == 2
    assert np.isfinite(batch["motion_target_xy"].numpy()).all()
