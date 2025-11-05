# # example.py
# from robosuite.models import MujocoWorldBase
# from robosuite.models.robots import Sawyer
# from new_Sawyer import MyNewSaywer
# from robosuite.models.grippers import gripper_factory, register_gripper
# from robosuite.models.arenas import TableArena
# from robosuite.models.objects import BallObject
# from robosuite.utils.mjcf_utils import new_joint
# import mujoco
# import time
# import math
# import numpy as np
# from robosuite.models.bases import RethinkMount
# from custom_gripper import CustomGripper # Import the custom gripper class

# # This import is redundant as you registered it in custom_gripper.py
# # from robosuite.models.grippers import ALL_GRIPPERS, GRIPPER_MAPPING, register_gripper
# import os
# current_directory = os.getcwd()
# world = MujocoWorldBase()
# mujoco_robot = MyNewSaywer()
# rethink_base = RethinkMount()
# mujoco_robot.add_base(rethink_base)

# # The registration should already happen when custom_gripper.py is imported.
# # You can remove the direct call to register_gripper(CustomGripper) here.

# # Instantiate the gripper using the factory
# gripper = gripper_factory("CustomGripper")

# mujoco_robot.add_gripper(gripper)

# mujoco_robot.set_base_xpos([0, 0, 0])
# mujoco_robot.set_base_ori([0, 0, math.pi])
# world.merge(mujoco_robot)

# mujoco_arena = TableArena(table_full_size=(3, 2, .05))
# mujoco_arena.set_origin([-1.8, 0, 0])
# world.merge(mujoco_arena)

# sphere = BallObject(
#     name="sphere",
#     size=[0.04],
#     rgba=[0, 0.5, 0.5, 1]).get_obj()
# sphere.set('pos', '-1.0 0 1.0')
# world.worldbody.append(sphere)

# # model = world.get_model(mode = "mujoco")

# # data = mujoco.MjData(model)
# # while data.time < 1:
# #     mujoco.mj_step(model, data)

# # model = world.get_model(mode="mujoco")
# # sim = MjSim(model)
# # viewer = MjViewer(sim)
# # viewer.vopt.geomgroup[0] = 0 # disable visualization of collision mesh
# # for i in range(10000):
# #     sim.data.ctrl[:] = 0
# #     sim.step()
# #     viewer.render()


# model = world.get_model(mode="mujoco")
# data = mujoco.MjData(model)

# # Step simulation a few times to stabilize
# for _ in range(10):
#     mujoco.mj_step(model, data)

# # Launch MuJoCo viewer simulation loop
# with mujoco.viewer.launch_passive(model, data) as viewer:
#     while data.time < 100:
#         mujoco.mj_step(model, data)
#         viewer.sync()
#         time.sleep(0.01)


from robosuite.models import MujocoWorldBase
from robosuite.models.robots import Sawyer
from new_Sawyer import MyNewSaywer
from robosuite.models.grippers import gripper_factory
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject
from robosuite.utils.mjcf_utils import new_joint
# from mujoco_py import MjSim, MjViewer
import mujoco
import time
import math
import numpy as np
from robosuite.models.bases import RethinkMount
from robosuite.models.grippers import ALL_GRIPPERS, GRIPPER_MAPPING, register_gripper
from custom_gripper import CustomGripper
import os

current_directory = os.getcwd()



world = MujocoWorldBase()
mujoco_robot = MyNewSaywer()
rethink_base = RethinkMount()
mujoco_robot.add_base(rethink_base)

# register_gripper(CustomGripper(current_directory + "/custom_gripper.xml"))
register_gripper(CustomGripper)
# gripper = gripper_factory('CustomGripper')
gripper = CustomGripper(current_directory + "/custom_gripper.xml")
mujoco_robot.add_gripper(gripper)

# register_gripper

mujoco_robot.set_base_xpos([0, 0, 0])
mujoco_robot.set_base_ori([0, 0, math.pi])
world.merge(mujoco_robot)

mujoco_arena = TableArena(table_full_size=(3, 2, .05))
mujoco_arena.set_origin([-1.8, 0, 0])
world.merge(mujoco_arena)

sphere = BallObject(
    name="sphere",
    size=[0.04],
    rgba=[0, 0.5, 0.5, 1]).get_obj()
sphere.set('pos', '-1.23 -0.16 1.282')
world.worldbody.append(sphere)

# model = world.get_model(mode = "mujoco")

# data = mujoco.MjData(model)
# while data.time < 1:
#     mujoco.mj_step(model, data)

# model = world.get_model(mode="mujoco")
# sim = MjSim(model)
# viewer = MjViewer(sim)
# viewer.vopt.geomgroup[0] = 0 # disable visualization of collision mesh
# for i in range(10000):
#     sim.data.ctrl[:] = 0
#     sim.step()
#     viewer.render()


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
