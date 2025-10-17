from robosuite.models import MujocoWorldBase
from robosuite.models.robots import Sawyer
from robosuite.models.grippers import gripper_factory
from robosuite.models.bases import RethinkMount
import mujoco
import time
import numpy as np

# Import the EmptyArena instead of TableArena
from robosuite.models.arenas import EmptyArena
from robosuite.models.objects import BallObject

# Initialize Mujoco world
world = MujocoWorldBase()

# Create Sawyer robot and its official base
rethink_base = RethinkMount()
mujoco_robot = Sawyer()
mujoco_robot.add_base(rethink_base)

# Add gripper to the robot
gripper = gripper_factory('RethinkGripper')
mujoco_robot.add_gripper(gripper)

# Position the combined model. Move it down to be on the floor.
mujoco_robot.set_base_xpos([0, 0, 0])
world.merge(mujoco_robot)

# Add an empty arena (just a floor)
mujoco_arena = EmptyArena()
world.merge(mujoco_arena)

# Create free-moving ball with default free joint
sphere = BallObject(
    name="sphere",
    size=[0.03],
    rgba=[0, 0.5, 0.5, 1]
).get_obj()

# Set ball initial position on the floor
sphere.set('pos', '1.1 0.16 1.245') # Adjust the z-position for the floor

# Add ball directly to worldbody
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
