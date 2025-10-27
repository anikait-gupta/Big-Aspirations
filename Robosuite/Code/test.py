import os
from custom_gripper import CustomGripper
from robosuite.models.grippers import gripper_factory

current_directory = os.getcwd()
print(current_directory)

print(CustomGripper.__name__)
print(1)
c = CustomGripper(current_directory + "/custom_gripper.xml")
print(c.__name__)

gripper = gripper_factory('PandaGripper')
print(type(gripper))