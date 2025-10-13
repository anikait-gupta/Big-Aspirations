from robosuite.models import MujocoWorldBase
from robosuite.models.robots import *
from robosuite.models.grippers import gripper_factory
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject
from robosuite.utils.mjcf_utils import new_joint
import mujoco
import time

world = MujocoWorldBase()
mujoco_robot = Sawyer()

gripper = gripper_factory('RethinkGripper')
mujoco_robot.add_gripper(gripper)

mujoco_robot.set_base_xpos([0, 0, 0])
world.merge(mujoco_robot)


mujoco_arena = TableArena()
mujoco_arena.set_origin([0.8, 0, 0])
world.merge(mujoco_arena)

sphere = BallObject(
    name="sphere",
    size=[0.04],
    rgba=[0, 0.5, 0.5, 1]).get_obj()
sphere.set('pos', '1.0 0 1.0')
world.worldbody.append(sphere)

model = world.get_model(mode="mujoco")

data = mujoco.MjData(model)

# print(gripper.speed())

# The `mujoco.viewer` module is used to create and manage the window for visualization.
with mujoco.viewer.launch_passive(model, data) as _viewer:
    # The 'mj_step' is called inside the loop to advance the simulation.
    while data.time < 100:
        mujoco.mj_step(model, data)
        # The 'sync' call updates the viewer window with the current simulation state.
        _viewer.sync()
        # A small pause is added to make the simulation visible.
        time.sleep(0.01)