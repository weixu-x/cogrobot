import numpy as np

from robosuite.environments.base import register_env
from robosuite.environments.manipulation.lift import Lift
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.placement_samplers import UniformRandomSampler

from robosuite.models.arenas.corsi_table_arena import CorsiTableArena   # <- 按你文件路径改


@register_env
class LiftCorsiArena(Lift):
    """
    Lift task, but using CorsiTableArena instead of TableArena.
    其他逻辑完全不动（reward / reset / obs / success 都还是 Lift 的）。
    """

    def _load_model(self):
        # 不调用 Lift._load_model()，否则它会创建 TableArena
        # 这里只调用 ManipulationEnv 的 _load_model()
        super(Lift, self)._load_model()

        # ---- 下面几乎逐行复制 Lift._load_model() ----

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)
        mujoco_arena = CorsiTableArena(
            fname="arenas/table_arena.xml",
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
            rows=3,
            cols=3,
            dx=0.07,
            dy=0.07,
        )

        mujoco_arena.set_origin([0, 0, 0])

        tex_attrib = {"type": "cube"}
        mat_attrib = {"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"}
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        # self.cube = BoxObject(
        #     name="cube",
        #     size_min=[0.020, 0.020, 0.020],
        #     size_max=[0.022, 0.022, 0.022],
        #     rgba=[1, 0, 0, 1],
        #     material=redwood,
        #     rng=self.rng,
        # )

        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.cube)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                # mujoco_objects=self.cube,
                mujoco_objects=[],
                x_range=[-0.03, 0.03],
                y_range=[-0.03, 0.03],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
                rng=self.rng,
            )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            # mujoco_objects=self.cube,
            mujoco_objects=[],
        )
