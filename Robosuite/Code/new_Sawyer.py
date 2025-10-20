from robosuite.models.robots import Sawyer
import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class MyNewSaywer(Sawyer):
    def __init__():
        super.__init__(xml_path_completion("your_robot_path.xml"))