from robosuite.models import MujocoWorldBase
from robosuite.models.robots import Sawyer
from robosuite.models.grippers import gripper_factory
from robosuite.models.bases import RethinkMount
from robosuite.utils.mjcf_utils import new_body, new_geom, new_site, find_elements
import mujoco
import time
import numpy as np
from robosuite.models.arenas import EmptyArena
from robosuite.models.objects import BallObject

# Initialize Mujoco world
world = MujocoWorldBase()

# Create Sawyer robot and its base
rethink_base = RethinkMount()
mujoco_robot = Sawyer()
mujoco_robot.add_base(rethink_base)

# Add custom gripper - attach directly to robot's worldbody
gripper_body = new_body(name="cylinder_gripper", pos="0 0 0.3")

# OUTER CYLINDER
outer_cylinder = new_geom(
    type="cylinder",
    name="outer_cylinder",
    size=[0.045, 0.07],
    pos=[0, 0, 0.07],
    rgba=[0.7, 0.7, 0.8, 1],
    mass=0.15,
    friction=[1.2, 0.005, 0.0001]
)
gripper_body.append(outer_cylinder)

# INNER CYLINDER
inner_cylinder = new_geom(
    type="cylinder",
    name="inner_cylinder",
    size=[0.040, 0.065],
    pos=[0, 0, 0.075],
    rgba=[0.4, 0.4, 0.5, 0.3],
    contype=0,
    conaffinity=0
)
gripper_body.append(inner_cylinder)

# BOTTOM CAP
bottom_cap = new_geom(
    type="cylinder",
    name="gripper_bottom",
    size=[0.045, 0.003],
    pos=[0, 0, 0.003],
    rgba=[0.7, 0.7, 0.8, 1],
    mass=0.05,
    friction=[1.2, 0.005, 0.0001]
)
gripper_body.append(bottom_cap)

# Add grip site
grip_site = new_site(
    name="grip_site",
    pos=[0, 0, 0.07],
    size=[0.005],
    rgba=[1, 0, 0, 1]
)
gripper_body.append(grip_site)

# Attach gripper to robot
mujoco_robot.worldbody.append(gripper_body)
print("Cylinder gripper added to robot")

# Position the combined model
mujoco_robot.set_base_xpos([0, 0, 0])
world.merge(mujoco_robot)

# Add an empty arena (just a floor)
mujoco_arena = EmptyArena()
world.merge(mujoco_arena)

# Custom table (30in x 60in x 1in, suspended 40in above floor)
table_body = new_body(name="custom_table", pos="0 0 1.016")

table_geom = new_geom(
    type="box",
    name="table_top",
    size=[0.381, 0.762, 0.0127],
    pos=[0, 0, 0],
    rgba=[0.6, 0.4, 0.2, 1],
    friction=[1.0, 0.005, 0.0001],
    mass=50.0 
)
table_body.append(table_geom)
world.worldbody.append(table_body)

# Create free-moving ball
sphere = BallObject(
    name="sphere",
    size=[0.03],
    rgba=[0, 0.5, 0.5, 1]
).get_obj()

# Set ball initial position on the table
sphere.set('pos', '0.1 0 1.05')
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
