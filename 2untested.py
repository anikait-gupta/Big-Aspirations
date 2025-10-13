from robosuite.models import MujocoWorldBase
from robosuite.models.robots import Sawyer
from robosuite.models.grippers import gripper_factory
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject
from robosuite.utils.mjcf_utils import new_body, new_geom, new_joint, new_site, find_elements
import mujoco
import numpy as np

class BallRollingEnv:
    """Environment wrapper for ball rolling task"""
    
    def __init__(self, render=False):
        self.render_mode = render
        
        # Target configuration
        self.target_pos = np.array([0.5, 0.0, 0.82])  # X, Y, Z on table
        self.target_radius = 0.10
        
        # Environment dimensions
        self.action_dim = 7  # 7 DOF Sawyer
        self.obs_dim = 20    # 7 joint pos + 7 joint vel + 3 ball pos + 3 ball-to-target
        
        # Build world
        self._build_world()
        
        # Viewer for rendering
        self.viewer = None
    
    def _create_cylinder_gripper(self):
        """Create a custom cylinder gripper geometry"""
        gripper_body = new_body(name="cylinder_gripper", pos=[0, 0, 0])
        
        outer_cylinder = new_geom(
            geom_type="cylinder",
            name="outer_cylinder",
            size=[0.045, 0.07],
            pos=[0, 0, 0.07],
            rgba=[0.7, 0.7, 0.8, 1],
            mass=0.15,
            friction=[1.2, 0.005, 0.0001],
        )
        gripper_body.append(outer_cylinder)
        
        inner_cylinder = new_geom(
            geom_type="cylinder",
            name="inner_cylinder",
            size=[0.040, 0.065],
            pos=[0, 0, 0.075],
            rgba=[0.4, 0.4, 0.5, 0.3],
            contype=0,
            conaffinity=0,
            group=1,
        )
        gripper_body.append(inner_cylinder)
        
        bottom_cap = new_geom(
            geom_type="cylinder",
            name="gripper_bottom",
            size=[0.045, 0.003],
            pos=[0, 0, 0.003],
            rgba=[0.7, 0.7, 0.8, 1],
            mass=0.05,
            friction=[1.2, 0.005, 0.0001],
        )
        gripper_body.append(bottom_cap)
        
        grip_site = new_site(
            name="grip_site",
            pos=[0, 0, 0.07],
            size=[0.005],
            rgba=[1, 0, 0, 1],
        )
        gripper_body.append(grip_site)
        
        ball_site = new_site(
            name="ball_contact_site",
            pos=[0, 0, 0.04],
            size=[0.002],
            rgba=[0, 1, 0, 1],
        )
        gripper_body.append(ball_site)
        
        return gripper_body
    
    def _attach_cylinder_gripper(self):
        """Attach custom cylinder gripper to Sawyer end effector"""
        # Correct end effector body name for Sawyer robot
        ee_body = find_elements(
            root=self.world.worldbody,
            tags="body",
            attribs={"name": "right_hand"},  # 'right_hand' is Sawyer's end effector body in robosuite
            return_first=True
        )
        
        if ee_body is None:
            print("ERROR: Could not find end effector body 'right_hand'")
            return False
        
        gripper = self._create_cylinder_gripper()
        ee_body.append(gripper)
        print("✓ Cylinder gripper attached to Sawyer end effector")
        return True
    
    def _build_world(self):
        self.world = MujocoWorldBase()
        
        # Create Sawyer robot and add default gripper (we will also attach custom cylinder gripper)
        mujoco_robot = Sawyer()
        default_gripper = gripper_factory('RethinkGripper')
        mujoco_robot.add_gripper(default_gripper)
        mujoco_robot.set_base_xpos([0, 0, 0])
        self.world.merge(mujoco_robot)
        
        # Add table arena
        mujoco_arena = TableArena()
        mujoco_arena.set_origin([0.8, 0, 0])
        self.world.merge(mujoco_arena)
        
        # Add ball object using robosuite BallObject class for better integration
        ball = BallObject(radius=0.04, density=500, rgba=[1,0.3,0.3,1])
        ball_body = ball.get_root()
        ball_body.set('pos', '0.85 0 0.82')  # Place ball on table near robot
        self.world.worldbody.append(ball_body)
        
        # Attach cylinder gripper to Sawyer's right_hand
        self._attach_cylinder_gripper()
        
        # Add target visualization to arena
        self._add_target_visualization(mujoco_arena)
        
        # Build Mujoco model and data
        self.model = self.world.get_model(mode="mujoco")
        self.data = mujoco.MjData(self.model)
        
        # Get IDs for simulation elements
        self.ball_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, ball.root_body)
        self.gripper_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'grip_site')
    
    def _add_target_visualization(self, arena):
        from robosuite.utils.mjcf_utils import new_geom
        
        target_geom = new_geom(
            geom_type="cylinder",
            size=[self.target_radius, 0.001],
            rgba=[0.2, 1.0, 0.2, 0.4],
            pos=[self.target_pos[0] - 0.8, self.target_pos[1], 0.001],  # relative to arena origin
            contype=0,
            conaffinity=0,
            group=1
        )
        arena.arena.append(target_geom)
    
    def reset(self):
        """Reset environment to initial state"""
        home_pose = np.array([0.0, -0.5, 0.0, 1.5, 0.0, 0.8, 0.0])
        self.data.qpos[:7] = home_pose  # Sawyer joints
        
        # Ball qpos: free joint has 7 dims (x,y,z + quaternion)
        ball_start_idx = 7  # Assuming ball qpos starts after robot joints
        grip_pos = self.data.site_xpos[self.gripper_site_id].copy()
        self.data.qpos[ball_start_idx:ball_start_idx+3] = grip_pos + np.array([0.05, 0, 0.05])
        self.data.qpos[ball_start_idx+3:ball_start_idx+7] = np.array([1, 0, 0, 0])  # Identity quaternion
        
        self.data.qvel[:] = 0.0
        
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        
        return self._get_observation()
    
    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.data.ctrl[:self.action_dim] = action
        
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
        
        obs = self._get_observation()
        reward = self._compute_reward()
        done = self._check_done()
        
        info = {
            'ball_pos': self._get_ball_pos(),
            'distance_to_target': self._get_distance_to_target(),
            'success': self._check_success()
        }
        return obs, reward, done, info
    
    def _get_observation(self):
        obs = np.concatenate([
            self.data.qpos[:7],
            self.data.qvel[:7],
            self._get_ball_pos(),
            self._get_ball_to_target()
        ])
        return obs.astype(np.float32)
    
    def _get_ball_pos(self):
        return self.data.xpos[self.ball_body_id].copy()
    
    def _get_ball_to_target(self):
        return self.target_pos - self._get_ball_pos()
    
    def _get_distance_to_target(self):
        ball_pos = self._get_ball_pos()
        return np.linalg.norm(ball_pos[:2] - self.target_pos[:2])
    
    def _compute_reward(self):
        dist = self._get_distance_to_target()
        reward = -dist
        if dist <= self.target_radius:
            reward += 10.0
        if self._get_ball_pos()[2] < 0.7:  # ball fell off table height
            reward -= 5.0
        return reward
    
    def _check_success(self):
        return self._get_distance_to_target() <= self.target_radius
    
    def _check_done(self):
        if self._get_ball_pos()[2] < 0.7:
            return True
        if self._check_success():
            return True
        return False
    
    def render(self):
        if self.viewer is None and self.render_mode:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
