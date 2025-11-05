

# # custom_gripper.py
# import numpy as np
# from robosuite.models.grippers.gripper_model import GripperModel
# from robosuite.utils.mjcf_utils import xml_path_completion
# from robosuite.models.grippers import register_gripper

# # This needs to inherit from the base GripperModel and use the path completion utility.
# class CustomGripper(GripperModel):
#    """
#    Custom Gripper
#    """
#    def __init__(self, idn=0):
#        # Use xml_path_completion to find the XML file relative to the assets path
#        super().__init__(xml_path_completion("grippers/custom_gripper.xml"), idn=idn)

#    @property
#    def speed(self):
#        return 0.2

#    @property
#    def dof(self):
#        return 1
   
#    @property
#    def init_qpos(self):
#        return np.array([0.0]) # A single value for a 1-DOF gripper

#    def format_action(self, action):
#        # Action mapping for a 1-DOF gripper
#        assert len(action) == self.dof
#        self.current_action = np.clip(
#            self.current_action + self.speed * np.sign(action), -1.0, 1.0
#        )
#        return self.current_action

# # Register the custom gripper under a specific name
# register_gripper(CustomGripper)




"""
Gripper for Franka's Panda (has two fingers).
"""
import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.models.grippers import register_gripper


class CustomGripperBase(GripperModel):
    """
    Gripper for custom

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, xml_path="custom_gripper.xml", idn=0):
        super().__init__(xml_path_completion(xml_path), idn=idn)
        # change the xml file here

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([])
        #return np.array([0.020833, -0.020833])

    """
    @property
    def _important_geoms(self):
        return {
            "left_finger": ["finger1_collision", "finger1_pad_collision"],
            "right_finger": ["finger2_collision", "finger2_pad_collision"],
            "left_fingerpad": ["finger1_pad_collision"],
            "right_fingerpad": ["finger2_pad_collision"],
        }
    """


class CustomGripper(CustomGripperBase):
    """
    Modifies PandaGripperBase to only take one action.
    """

    def __init__(self, xml_path="custom_gripper.xml", *args, **kwargs):
        super().__init__(xml_path, kwargs)
        self.__name__ = CustomGripper.__name__

    def format_action(self, action):
        """
        Maps continuous action into binary output
        -1 => open, 1 => closed

        Args:
            action (np.array): gripper-specific action

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        assert len(action) == self.dof
        self.current_action = np.clip(
            self.current_action + np.array([-1.0, 1.0]) * self.speed * np.sign(action), -1.0, 1.0
        )
        return self.current_action

    @property
    def speed(self):
        return 0.2

    @property
    def dof(self):
        return 1

register_gripper(CustomGripper)