from robosuite.models import MujocoWorldBase
from robosuite.models.robots import Sawyer
from robosuite.models.grippers import gripper_factory
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject
from robosuite.utils.mjcf_utils import new_body, new_geom, new_joint, new_site, find_elements
import mujoco
import time
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
        self.obs_dim = 20    # Will be: 7 joint pos + 7 joint vel + 3 ball pos + 3 ball-to-target
        
        # Build world
        self._build_world()
        
        # Viewer for rendering
        self.viewer = None
    
    def _create_cylinder_gripper(self):
        """
        Create a gripper made of two nested cylinders (outer and inner)
        Returns the gripper body element
        """
        # Create gripper body
        gripper_body = new_body(name="cylinder_gripper", pos="0 0 0")
        
        # OUTER CYLINDER (the main structure)
        outer_cylinder = new_geom(
            geom_type="cylinder",
            name="outer_cylinder",
            size=[0.045, 0.07],  # [radius, half-height] = 4.5cm radius, 7cm tall
            pos=[0, 0, 0.07],    # Position: 7cm up from attachment point
            rgba=[0.7, 0.7, 0.8, 1],  # Gray-blue color, opaque
            mass=0.15,           # 150 grams
            friction=[1.2, 0.005, 0.0001],  # High friction for gripping
        )
        gripper_body.append(outer_cylinder)
        
        # INNER CYLINDER (hollow space inside)
        inner_cylinder = new_geom(
            geom_type="cylinder",
            name="inner_cylinder",
            size=[0.040, 0.065],  # Slightly smaller: 4cm radius, 6.5cm tall
            pos=[0, 0, 0.075],   # Slightly higher to create bottom
            rgba=[0.4, 0.4, 0.5, 0.3],  # Darker, semi-transparent
            contype=0,           # No collision (it's hollow space)
            conaffinity=0,       # No collision
            group=1,             # Visual only
        )
        gripper_body.append(inner_cylinder)
        
        # BOTTOM CAP (closes the bottom of the cup)
        bottom_cap = new_geom(
            geom_type="cylinder",
            name="gripper_bottom",
            size=[0.045, 0.003],  # Same radius as outer, very thin
            pos=[0, 0, 0.003],   # At the bottom
            rgba=[0.7, 0.7, 0.8, 1],  # Same color as outer
            mass=0.05,           # 50 grams
            friction=[1.2, 0.005, 0.0001]
        )
        gripper_body.append(bottom_cap)
        
        # Add grip site (reference point for positioning)
        grip_site = new_site(
            name="grip_site",
            pos=[0, 0, 0.07],    # Center of cup
            size=[0.005],        # Small marker
            rgba=[1, 0, 0, 1]    # Red marker (for debugging)
        )
        gripper_body.append(grip_site)
        
        # Add ball contact site (where ball should be)
        ball_site = new_site(
            name="ball_contact_site",
            pos=[0, 0, 0.04],    # Lower in cup where ball rests
            size=[0.002],
            rgba=[0, 1, 0, 1]    # Green marker
        )
        gripper_body.append(ball_site)
        
        return gripper_body
    
    def _attach_cylinder_gripper(self):
        """Attach the cylinder gripper to the Sawyer robot's end effector"""
        # Find the robot's end effector body
        ee_body = find_elements(
            root=self.world.worldbody,
            tags="body",
            attribs={"name": "right_l6"},  # Sawyer's wrist link unsure if this is it
            return_first=True
        )
        
        if ee_body is None:
            print("ERROR: Could not find end effector body 'right_l6'")
            print("Available bodies:")
            for body in self.world.worldbody.findall(".//body"):
                if body.get("name"):
                    print(f"  - {body.get('name')}")
            return False
        
        # Create and attach gripper
        gripper = self._create_cylinder_gripper()
        ee_body.append(gripper)
        
        print("✓ Cylinder gripper attached to robot end effector")
        return True
        
    def _build_world(self):
        """Build the MuJoCo world (your existing code)"""
        # Initialize Mujoco world
        self.world = MujocoWorldBase()
        
        # Turn off gravity (optional - uncomment to disable)
        # self.world.worldbody.find("option").set("gravity", "0 0 0")
        
        # Create Sawyer robot and add gripper
        mujoco_robot = Sawyer()
        gripper = gripper_factory('RethinkGripper')
        mujoco_robot.add_gripper(gripper)
        mujoco_robot.set_base_xpos([0, 0, 0])
        self.world.merge(mujoco_robot)
        
        # Add arena (table)
        mujoco_arena = TableArena()
        mujoco_arena.set_origin([0.8, 0, 0])
        self.world.merge(mujoco_arena)
        
        # Create ball using custom sphere geometry
        from robosuite.utils.mjcf_utils import new_body, new_geom, new_joint
        
        sphere_body = new_body(name="sphere")
        sphere_geom = new_geom(
            geom_type="sphere",
            size=[0.04],  # 4cm radius
            rgba=[1, 0.3, 0.3, 1],  # Red ball
            density=500,  # Mass properties
            friction=[1.0, 0.005, 0.0001],
            pos=[0, 0, 0]
        )
        sphere_body.append(sphere_geom)
        
        # Add free joint for unconstrained movement
        sphere_joint = new_joint(
            joint_type="free",
            name="sphere_joint"
        )
        sphere_body.append(sphere_joint)
        
        # Set initial position
        sphere_body.set('pos', '0.1 0 0.8')
        
        self.world.worldbody.append(sphere_body)
        
        # Add target visualization (green circle)
        self._add_target_visualization()
        
        # Build the MuJoCo model and data
        self.model = self.world.get_model(mode="mujoco")
        
        # Turn off gravity (set to 0 in all axes)
        # self.model.opt.gravity[:] = [0, 0, 0]  # Uncomment to disable gravity
        
        self.data = mujoco.MjData(self.model)
        
        # Get important IDs
        self.ball_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'sphere')
        self.gripper_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'grip_site')
        
    def _add_target_visualization(self):
        """Add green target circle to world"""
        from robosuite.utils.mjcf_utils import new_geom
        
        # Create target geom (flat cylinder)
        target_geom = new_geom(
            geom_type="cylinder",
            size=[self.target_radius, 0.001],  # radius and height
            rgba=[0.2, 1.0, 0.2, 0.4],  # green, semi-transparent
            pos=[self.target_pos[0] - 0.8, self.target_pos[1], 0.001],  # Relative to table
            contype=0,  # No collision
            conaffinity=0,
            group=1  # Visual only
        )
        
        # Add to arena
        self.world.worldbody.append(target_geom)
    
    def _print_gripper_info(self):
        """Print information about the cylinder gripper"""
        try:
            gripper_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'cylinder_gripper')
            grip_site_id = self.gripper_site_id
            
            print("\n=== Cylinder Gripper Info ===")
            print(f"Gripper Body ID: {gripper_id}")
            print(f"Gripper Position: {self.data.xpos[gripper_id]}")
            print(f"Grip Site ID: {grip_site_id}")
            print(f"Grip Site Position: {self.data.site_xpos[grip_site_id]}")
            
            outer_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'outer_cylinder')
            inner_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'inner_cylinder')
            bottom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'gripper_bottom')
            
            print(f"Outer Cylinder ID: {outer_id}")
            print(f"Inner Cylinder ID: {inner_id}")
            print(f"Bottom Cap ID: {bottom_id}")
            print("============================\n")
        except Exception as e:
            print(f"Could not print gripper info: {e}\n")
        
    def reset(self):
        """Reset environment to initial state"""
        # Set robot to home pose (better pose than before)
        home_pose = np.array([0.0, -0.5, 0.0, 1.5, 0.0, 0.8, 0.0])
        self.data.qpos[:7] = home_pose
        
        # Get gripper position
        gripper_pos = self.data.site_xpos[self.gripper_site_id].copy()
        
        # Place ball in/near gripper
        # Ball has a free joint, so qpos is [x, y, z, qw, qx, qy, qz]
        ball_start_idx = 7  # After 7 robot joints
        self.data.qpos[ball_start_idx:ball_start_idx+3] = gripper_pos + np.array([0.05, 0, 0.05])
        self.data.qpos[ball_start_idx+3:ball_start_idx+7] = np.array([1, 0, 0, 0])  # Identity quaternion
        
        # Zero all velocities
        self.data.qvel[:] = 0.0
        
        # Step simulation to stabilize
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        
        return self._get_observation()
    
    def step(self, action):
        """Execute action and return (obs, reward, done, info)"""
        # Apply action to robot (position control)
        # Clip action to reasonable range
        action = np.clip(action, -1.0, 1.0)
        
        # Set control (you may need to adjust this based on actuator setup)
        self.data.ctrl[:self.action_dim] = action
        
        # Step simulation multiple times for one action
        for _ in range(5):  # 5 substeps per action
            mujoco.mj_step(self.model, self.data)
        
        # Get new observation
        obs = self._get_observation()
        
        # Compute reward
        reward = self._compute_reward()
        
        # Check if episode is done
        done = self._check_done()
        
        # Additional info
        info = {
            'ball_pos': self._get_ball_pos(),
            'distance_to_target': self._get_distance_to_target(),
            'success': self._check_success()
        }
        
        return obs, reward, done, info
    
    def _get_observation(self):
        """Get current state observation"""
        obs = np.concatenate([
            self.data.qpos[:7],           # Robot joint positions (7)
            self.data.qvel[:7],           # Robot joint velocities (7)
            self._get_ball_pos(),         # Ball position (3)
            self._get_ball_to_target()    # Vector from ball to target (3)
        ])
        return obs.astype(np.float32)
    
    def _get_ball_pos(self):
        """Get ball 3D position"""
        return self.data.xpos[self.ball_body_id].copy()
    
    def _get_ball_to_target(self):
        """Get vector from ball to target"""
        ball_pos = self._get_ball_pos()
        return self.target_pos - ball_pos
    
    def _get_distance_to_target(self):
        """Get 2D distance from ball to target (only X-Y)"""
        ball_pos = self._get_ball_pos()
        return np.linalg.norm(ball_pos[:2] - self.target_pos[:2])
    
    def _compute_reward(self):
        """Compute reward based on ball distance to target"""
        distance = self._get_distance_to_target()
        
        # Shaped reward: negative distance
        reward = -distance
        
        # Large bonus for success
        if distance <= self.target_radius:
            reward += 10.0
        
        # Small penalty for ball falling off table
        ball_pos = self._get_ball_pos()
        if ball_pos[2] < 0.7:  # Below table
            reward -= 5.0
        
        return reward
    
    def _check_success(self):
        """Check if ball is in target area"""
        distance = self._get_distance_to_target()
        return distance <= self.target_radius
    
    def _check_done(self):
        """Check if episode should terminate"""
        ball_pos = self._get_ball_pos()
        
        # Done if ball falls off table
        if ball_pos[2] < 0.7:
            return True
        
        # Done if success (optional - can remove to keep rolling)
        if self._check_success():
            return True
        
        return False
    
    def render(self):
        """Render the environment"""
        if self.viewer is None and self.render_mode:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        if self.viewer is not None:
            self.viewer.sync()
    
    def close(self):
        """Close the environment"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# ============================================================================
# Test the environment
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Check if we should render
    use_viewer = '--render' in sys.argv
    
    # Create environment
    print("Creating environment...")
    env = BallRollingEnv(render=False)
    
    print("Environment created!")
    print(f"Action dimension: {env.action_dim}")
    print(f"Observation dimension: {env.obs_dim}")
    
    # Test reset
    print("\nTesting reset...")
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial ball position: {env._get_ball_pos()}")
    print(f"Target position: {env.target_pos}")
    print(f"Distance to target: {env._get_distance_to_target():.3f}m")
    
    # Run a test episode with random actions
    print("\nRunning test episode with random actions...")
    print("(To see visualization, run with: python moretest.py --render)")
    
    if use_viewer:
        # With viewer
        print("Opening MuJoCo viewer...")
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            episode_reward = 0
            step_count = 0
            
            for step in range(500):
                action = np.random.randn(env.action_dim) * 0.1
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                step_count += 1
                
                viewer.sync()
                time.sleep(0.01)
                
                if step % 50 == 0:
                    print(f"Step {step}: Reward={reward:.3f}, Distance={info['distance_to_target']:.3f}m, Success={info['success']}")
                
                if done:
                    print(f"\nEpisode finished!")
                    print(f"Total steps: {step_count}")
                    print(f"Total reward: {episode_reward:.2f}")
                    print(f"Final distance: {info['distance_to_target']:.3f}m")
                    print(f"Success: {info['success']}")
                    break
    else:
        # Without viewer (headless)
        episode_reward = 0
        step_count = 0
        
        for step in range(500):
            action = np.random.randn(env.action_dim) * 0.1
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
            if step % 50 == 0:
                print(f"Step {step}: Reward={reward:.3f}, Distance={info['distance_to_target']:.3f}m, Success={info['success']}")
            
            if done:
                print(f"\nEpisode finished!")
                print(f"Total steps: {step_count}")
                print(f"Total reward: {episode_reward:.2f}")
                print(f"Final distance: {info['distance_to_target']:.3f}m")
                print(f"Success: {info['success']}")
                break
    
    env.close()
    print("\n✓ Environment test complete!")
    print("Environment is working correctly and ready for training.")