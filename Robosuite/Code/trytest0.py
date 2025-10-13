import numpy as np
import robosuite as suite

# create environment instance
env = suite.make(
    env_name="Door", # try with other tasks like "Stack" and "Door"
    robots="Sawyer",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    horizon = 2000,
    ignore_done=True,  # This makes the horizon infinite
)

# reset the environment
env.reset()

# Access the horizon variable directly
print(f"Current horizon: {env.horizon}")

for i in range(10000):
    action = np.random.randn(*env.action_spec[0].shape) * 1
    action = [0,0.5,0,0,0,0,-0.01]
    print(action)
    obs, reward, done, info = env.step(action)  # take action in the environment
    env.render()  # render on display
    if done:
        # The episode is over. Reset the environment.
        obs = env.reset()