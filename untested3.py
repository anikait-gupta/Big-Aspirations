from robosuite.models import MujocoWorldBase
from robosuite.models.robots import Sawyer
from robosuite.models.grippers import gripper_factory
from robosuite.models.bases import RethinkMount
from robosuite.utils.mjcf_utils import new_body, new_geom, new_site, find_elements
import mujoco
import time
import numpy as np
# Import the EmptyArena instead of TableArena
from robosuite.models.arenas import EmptyArena
from robosuite.models.objects import BallObject

# Initialize Mujoco world
world = MujocoWorldBase()

# Create Sawyer robot and its base
rethink_base = RethinkMount()
mujoco_robot = Sawyer()
mujoco_robot.add_base(rethink_base)

# I add custom cylinder gripper in later, commented out
# gripper = gripper_factory('RethinkGripper')  # REMOVED
# mujoco_robot.add_gripper(gripper)            # REMOVED
ee_body = find_elements(
    root=world.worldbody,
    tags="body",
    attribs={"name": "robot0_right_j6"},  # Sawyer's wrist link might be wrong, but it needs to be the "end effector"
    return_first=True
)

if ee_body is not None:
    print("eeee")
    # Create cylinder gripper body
    gripper_body = new_body(name="cylinder_gripper", pos="0 0 0")
    
    # OUTER CYLINDER (the main structure)
    outer_cylinder = new_geom(
        type="cylinder",
        name="outer_cylinder",
        size=[0.045, 0.07],  # [radius, half-height] = 4.5cm radius, 7cm tall
        pos=[0, 0, 0.07],
        rgba=[0.7, 0.7, 0.8, 1],  # Gray-blue color
        mass=0.15,
        friction=[1.2, 0.005, 0.0001]
    )
    gripper_body.append(outer_cylinder)
    
    # INNER CYLINDER (hollow space inside)
    inner_cylinder = new_geom(
        type="cylinder",
        name="inner_cylinder",
        size=[0.040, 0.065],  # Slightly smaller
        pos=[0, 0, 0.075],
        rgba=[0.4, 0.4, 0.5, 0.3],  # Semi-transparent
        contype=0,  # No collision
        conaffinity=0
    )
    gripper_body.append(inner_cylinder)
    
    # BOTTOM CAP (closes the bottom of the cup)
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
    
    # Add grip site (reference point)
    grip_site = new_site(
        name="grip_site",
        pos=[0, 0, 0.07],
        size=[0.005],
        rgba=[1, 0, 0, 1]  # Red marker
    )
    gripper_body.append(grip_site)
    
    # Attach gripper to robot
    ee_body.append(gripper_body)
    print("Cylinder gripper attached to robot")
else:
    print("Could not find end effector body")


# Position the combined model
mujoco_robot.set_base_xpos([0, 0, 0])
world.merge(mujoco_robot)

# Add an empty arena (just a floor)
mujoco_arena = EmptyArena()
world.merge(mujoco_arena)


# custom table (30in x 60in x 1in, suspended 40in above floor)

table_body = new_body(name="custom_table", pos="0 0 1.016")  # 40 inches = 1.016m

table_geom = new_geom(
    type="box",
    name="table_top",
    size=[0.381, 0.762, 0.0127],  # Half-sizes: [15in, 30in, 0.5in] in meters
    pos=[0, 0, 0],
    rgba=[0.6, 0.4, 0.2, 1],  # Brown wood color
    friction=[1.0, 0.005, 0.0001],
    mass=50.0 
)
table_body.append(table_geom)
world.worldbody.append(table_body)


# Add custom gripper

# Find the robot's end effector body

# Create free-moving ball with default free joint
sphere = BallObject(
    name="sphere",
    size=[0.03],
    rgba=[0, 0.5, 0.5, 1]
).get_obj()

# Set ball initial position on the table, change to gripper position later once it works
sphere.set('pos', '0.1 0 1.05')  # On top of table at 40in + 1in + ball radius

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




