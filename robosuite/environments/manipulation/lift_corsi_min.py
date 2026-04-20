from typing import Optional, Sequence, Tuple

import numpy as np

from robosuite.environments.base import register_env
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.tasks import ManipulationTask

from robosuite.models.arenas.corsi_table_arena import CorsiTableArena


@register_env
class CorsiSceneDemo(ManipulationEnv):
    """
    Minimal demo env:
    - robot (e.g., PandaDexRH)
    - your CorsiTableArena (table + 9 random non-overlapping blocks)
    - NO cube / NO rewards / NO success condition
    """

    def __init__(
        self,
        robots,
        block_xy_positions: Optional[Sequence[Tuple[float, float]]] = None,
        corsi_board_size_xy: Optional[Tuple[float, float]] = None,
        **kwargs,
    ):
        # 你不需要 object obs
        # kwargs.setdefault("use_object_obs", False)
        self.block_xy_positions = None if block_xy_positions is None else list(block_xy_positions)
        self.corsi_board_size_xy = None if corsi_board_size_xy is None else tuple(corsi_board_size_xy)
        super().__init__(robots=robots, **kwargs)

    def _load_model(self):
        super()._load_model()

        # robot base pose like Lift (table setup)
        table_full_size = (0.8, 0.8, 0.05)
        table_offset = np.array((0, 0, 0.8))

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = CorsiTableArena(
            fname="arenas/table_arena.xml",
            table_full_size=table_full_size,
            table_friction=(1.0, 5e-3, 1e-4),
            table_offset=table_offset,
            rows=3, cols=3,
            block_xy_positions=self.block_xy_positions,
            corsi_board_size_xy=self.corsi_board_size_xy,
        )
        mujoco_arena.set_origin([0, 0, 0])

        # ✅ 关键：不传 cube
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[],   # 空列表
        )

    def reward(self, action=None):
        return 0.0

    def _check_success(self):
        return False
