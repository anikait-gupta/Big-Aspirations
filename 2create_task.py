import time
import numpy as np
from robosuite.controllers import load_composite_controller_config
from robosuite.utils.input_utils import InputHandler
import mujoco

# Your custom environment import (assuming defined in ball_rolling_env.py)
# from ball_rolling_env import BallRollingEnv

def main():
    # Load pose-space controller config for Sawyer teleop
    controller_config = load_composite_controller_config(controller="OSC_POSE")

    # Create environment instance with rendering enabled
    env = BallRollingEnv(render=True)

    # Initialize Mujoco viewer for rendering and InputHandler for teleop
    env.reset()
    input_handler = InputHandler(env)

    # Storage for demonstration data
    joint_positions = []
    joint_velocities = []
    actions = []
    ball_positions = []

    done = False
    print("Starting teleoperation. Use keyboard/SpaceMouse to control the robot. Press ESC to quit.")
    
    while not done:
        # Get action from teleoperation input (keyboard, SpaceMouse, etc.)
        action = input_handler.get_action()

        # Step simulation with action
        obs, reward, done, info = env.step(action)

        # Record demonstration data
        joint_positions.append(env.data.qpos[:7].copy())      # 7 Sawyer joints
        joint_velocities.append(env.data.qvel[:7].copy())
        actions.append(action.copy())
        ball_positions.append(info['ball_pos'])

        # Render environment
        env.render()

        # Control loop sleep for target control frequency (~20 Hz)
        time.sleep(0.05)

        # Optional: Exit on ESC key
        if input_handler.is_exit_requested():
            done = True

    # Save collected demonstration data after teleoperation ends
    np.savez_compressed(
        "demo_data.npz",
        joint_positions=np.array(joint_positions),
        joint_velocities=np.array(joint_velocities),
        actions=np.array(actions),
        ball_positions=np.array(ball_positions),
    )
    print("Demonstration data saved to demo_data.npz")

    env.close()
    print("Teleoperation session ended.")

if __name__ == "__main__":
    main()
