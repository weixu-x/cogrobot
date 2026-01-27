import numpy as np

from robosuite.robots import register_robot_class
from robosuite.models.robots import Panda
from robosuite.models.grippers import InspireRightHand, register_gripper
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
import mujoco


@register_gripper
class InspireRightHandWithInit(InspireRightHand):
    @property
    def init_qpos(self):
        base_qpos = np.array([-1.5, -1.5, -1.5, -1.5, -3.0, 3.0])
        indices = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5])
        return base_qpos[indices]


# @register_robot_class("FixedBaseRobot")
class PandaDexRH(Panda):
    @property
    def default_gripper(self):
        return {"right": "InspireRightHandWithInit"}

    @property
    def gripper_mount_pos_offset(self):
        return {"right": [0.0, 0.0, 0.0]}

    @property
    def gripper_mount_quat_offset(self):
        return {"right": [-0.5, 0.5, 0.5, -0.5]}
    
# Create environment
env = suite.make(
    env_name="Lift",
    robots="PandaDexRH",
    controller_configs=load_composite_controller_config(controller="BASIC"),
    has_renderer=True,
    has_offscreen_renderer=False,
    render_camera="agentview",
    use_camera_obs=False,
    control_freq=20,
)

# Run the simulation, and visualize it
env.reset()
mujoco.viewer.launch(env.sim.model._model, env.sim.data._data)
