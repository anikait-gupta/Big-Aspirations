from robosuite.models import MujocoWorldBase
from robosuite.models.robots import Sawyer
from robosuite.models.grippers import gripper_factory
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject
from robosuite.models.bases import RethinkMount
import mujoco
import time
import numpy as np

# Initialize Mujoco world
world = MujocoWorldBase()

# Create Sawyer robot and add gripper
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

# Set ball initial position close to gripper (world coordinates)
sphere.set('pos', '0.1 0 2')

# Add ball directly to worldbody to allow free movement
world.worldbody.append(sphere)

# Build the MuJoCo model and data
model = world.get_model(mode="mujoco")
data = mujoco.MjData(model)

# Step simulation a few times to stabilize
for _ in range(10):
    mujoco.mj_step(model, data)

# Launch MuJoCo viewer simulation loop
with mujoco.viewer.launch_passive(model, data) as viewer:
    while data.time < 100:
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)
