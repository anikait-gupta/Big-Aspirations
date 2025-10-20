# Create custom cylinder gripper class
class CylinderGripper(GripperModel):
    """Custom cylinder gripper model"""
    
    def __init__(self, idn=0):
        # Create the gripper XML structure
        super().__init__(xml_path_completion("grippers/rethink_gripper.xml"), idn=idn)
        
        # Clear out the default gripper structure
        for child in list(self.worldbody):
            self.worldbody.remove(child)
        
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
        
        # Add to worldbody
        self.worldbody.append(gripper_body)
        
    @property
    def init_qpos(self):
        return np.array([])
    
    @property
    def _important_geoms(self):
        return {
            "left_finger": ["outer_cylinder"],
            "right_finger": ["gripper_bottom"],
        }

gripper = CylinderGripper()
mujoco_robot.add_gripper(gripper, arm_name="right_hand")
print("Cylinder gripper attached to robot")