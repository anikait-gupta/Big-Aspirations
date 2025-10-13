import time
import numpy as np
from robosuite.controllers import load_composite_controller_config
from robosuite.utils.input_utils import InputHandler
from ball_rolling_env import BallRollingEnv
import mujoco

# Your custom environment import (assuming defined in ball_rolling_env.py)
# from ball_rolling_env import BallRollingEnv

# For demo we substitute BallRollingEnv by your class name
# Make sure your environment class is updated per the previous instructions

def main():
    # Load pose-space controller config for Sawyer teleop
    controller_config = load_composite_controller_config(controller="OSC_POSE")

    # Create environment instance with rendering enabled
    env = BallRollingEnv(render=True)

    # Initialize Mujoco viewer for rendering and InputHandler for teleop
    env.reset()
    input_handler = InputHandler(env)

    done = False
    print("Starting teleoperation. Use keyboard/SpaceMouse to control the robot. Press ESC to quit.")
    
    while not done:
        # Get action from teleoperation input (keyboard, SpaceMouse, etc.)
        action = input_handler.get_action()

        # Step simulation with action
        obs, reward, done, info = env.step(action)

        # Render environment
        env.render()

        # Control loop sleep for target control frequency (~20 Hz)
        time.sleep(0.05)

        # Optional: Exit on ESC key
        if input_handler.is_exit_requested():
            done = True

    env.close()
    print("Teleoperation session ended.")

if __name__ == "__main__":
    main()
