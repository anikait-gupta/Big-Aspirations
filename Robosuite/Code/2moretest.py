import robosuite as suite
import mujoco
from mujoco import viewer
import numpy as np
import time
from robosuite.models import MujocoWorldBase
from robosuite.models.robots import Sawyer
from robosuite.models.grippers import gripper_factory
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject

world = MujocoWorldBase()

mujoco_robot = Sawyer()
gripper = gripper_factory('RethinkGripper')
mujoco_robot.add_gripper(gripper)
mujoco_robot.set_base_xpos([0, 0, 0])
world.merge(mujoco_robot)

# Add arena (table)
mujoco_arena = TableArena()
mujoco_arena.set_origin([0.8, 0, 0])
world.merge(mujoco_arena)

# Create free-moving ball with default free joint
sphere = BallObject(
    name="sphere",
    size=[0.04],
    rgba=[0, 0.5, 0.5, 1]
).get_obj()
# Create the Lift environment with Sawyer + Rethink gripper
env = suite.make(
    "Lift",
    robots="Sawyer",
    gripper_types="RethinkGripper",
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    use_object_obs=True,
)

env.reset()
model = env.sim.model
data = env.sim.data

# Set Sawyer arm pose
desired_pose = {
    "robot0_right_j0": 0.0,
    "robot0_right_j1": -1.0,
    "robot0_right_j2": 0.0,
    "robot0_right_j3": 1.5,
    "robot0_right_j4": 0.0,
    "robot0_right_j5": 1.57,
    "robot0_right_j6": 0.0,
}

joint_names = list(model.joint_names)
for i, name in enumerate(joint_names):
    if name in desired_pose:
        data.qpos[i] = desired_pose[name]
env.sim.forward()

# Get gripper position
body_names = [name.decode("utf-8") if isinstance(name, bytes) else name for name in model.body_names]
gripper_name = "gripper0_right_eef"
gripper_idx = body_names.index(gripper_name)
gripper_pos = data.xpos[gripper_idx].copy()

# Set cube position at gripper position as ball proxy
joint_name = "cube_joint0"
joint_idx = model.joint_names.index(joint_name)
start_idx = joint_idx * 7  # 7 qpos for free joint
#data.qpos[start_idx:start_idx + 3] = gripper_pos
env.sim.forward()

# Step simulation to stabilize
for _ in range(10):
    env.sim.step()

# Create native mujoco model and data for viewer visualization
mujoco_model = world.get_model(mode="mujoco")
mujoco_data = mujoco.MjData(mujoco_model)

# Launch mujoco viewer loop with native mujoco model/data
with viewer.launch_passive(mujoco_model, mujoco_data) as mujoco_viewer:
    while mujoco_data.time < 100:
        mujoco.mj_step(mujoco_model, mujoco_data)
        mujoco_viewer.sync()
        time.sleep(0.01)
