from robosuite.models import MujocoWorldBase
from robosuite.models.robots import Sawyer
from robosuite.models.grippers import gripper_factory
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject
import time
import numpy as np
from robosuite.robots import register_robot_class
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
import mujoco


@register_robot_class("WheeledRobot")
class MobilePanda(Sawyer):
    @property
    def default_base(self):
        return "OmronMobileBase"

    @property
    def default_arms(self):
        return {"right": "Sawyer"}

# Create environment
env = suite.make(
    env_name="Lift",
    robots="Sawyer",
    controller_configs=load_composite_controller_config(controller="BASIC"),
    has_renderer=True,
    has_offscreen_renderer=False,
    render_camera="agentview",
    use_camera_obs=False,
    control_freq=20,
)

# Run the simulation, and visualize it
env.reset()

# Initialize Mujoco world
world = MujocoWorldBase()

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

# Set robot joints to a stable home pose (radians)
home_pose = np.array([100, 100, 100, 100, 100, 100, 100])
for i in range(len(home_pose)):
    data.qpos[i] = home_pose[i]





# Step simulation a few times to stabilize
for _ in range(10):
    mujoco.mj_step(model, data)

# Launch MuJoCo viewer simulation loop
with mujoco.viewer.launch_passive(model, data) as viewer:
    while data.time < 100:
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)
