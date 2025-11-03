from robosuite.models import MujocoWorldBase
from robosuite.models.robots import Sawyer
from robosuite.models.bases import RethinkMount
from custom_gripper import CustomGripper
import mujoco
import time
import numpy as np
from robosuite.models.arenas import EmptyArena
from robosuite.models.objects import BallObject
from robosuite.utils.mjcf_utils import new_body, new_geom
import os
import h5py
from datetime import datetime
import pygame


for _ in range(10):
    mujoco.mj_step(model, data)


# ==============================================================================
# PS4 CONTROLLER SETUP
# ==============================================================================

# Initialize pygame for controller input
pygame.init()
pygame.joystick.init()

# Check for connected controllers
if pygame.joystick.get_count() == 0:
    print("\n❌ No PS4 controller detected!")
    print("Please connect your PS4 controller and try again.")
    print("\nOn Linux, you may need to:")
    print("  sudo apt-get install joystick")
    print("  jstest /dev/input/js0")
    exit(1)

# Initialize the first controller
controller = pygame.joystick.Joystick(0)
controller.init()

print(f"\n✅ Controller connected: {controller.get_name()}")
print(f"   Axes: {controller.get_numaxes()}")
print(f"   Buttons: {controller.get_numbuttons()}")


# ==============================================================================
# PS4 CONTROLLER MAPPING
# ==============================================================================

# PS4 Controller Layout:
# Axes:
#   0: Left Stick X (left=-1, right=+1)
#   1: Left Stick Y (up=-1, down=+1)
#   2: L2 Trigger (0 to 1)
#   3: Right Stick X
#   4: Right Stick Y
#   5: R2 Trigger (0 to 1)
#
# Buttons:
#   0: X (Cross)
#   1: O (Circle)
#   2: □ (Square)
#   3: △ (Triangle)
#   4: Share
#   5: PS Button
#   6: Options
#   7: L3 (Left stick press)
#   8: R3 (Right stick press)
#   9: L1
#   10: R1
#   11: D-Pad Up
#   12: D-Pad Down
#   13: D-Pad Left
#   14: D-Pad Right

PS4_AXES = {
    'LEFT_X': 0,
    'LEFT_Y': 1,
    'L2': 2,
    'RIGHT_X': 3,
    'RIGHT_Y': 4,
    'R2': 5
}

PS4_BUTTONS = {
    'X': 0,
    'CIRCLE': 1,
    'SQUARE': 2,
    'TRIANGLE': 3,
    'SHARE': 4,
    'PS': 5,
    'OPTIONS': 6,
    'L3': 7,
    'R3': 8,
    'L1': 9,
    'R1': 10,
    'DPAD_UP': 11,
    'DPAD_DOWN': 12,
    'DPAD_LEFT': 13,
    'DPAD_RIGHT': 14
}


# ==============================================================================
# CONTROL PARAMETERS
# ==============================================================================

control_speed = 0.015      # Cartesian movement speed
joint_speed = 0.08         # Joint control speed
deadzone = 0.15            # Ignore stick input below this threshold

# Recording state
is_recording = False
recording_start_time = None
recorded_data = {
    'joint_positions': [],
    'joint_velocities': [],
    'ee_positions': [],
    'ball_positions': [],
    'timestamps': [],
    'controller_inputs': []
}

should_quit = False
last_button_state = {}  # Track button states for toggle detection


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def apply_deadzone(value, threshold=0.15):
    """Apply deadzone to joystick input"""
    if abs(value) < threshold:
        return 0.0
    # Scale the remaining range
    sign = 1 if value > 0 else -1
    return sign * (abs(value) - threshold) / (1.0 - threshold)


def get_ee_position(model, data):
    """Get end effector position"""
    try:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'grip_site')
        return data.site_xpos[site_id].copy()
    except:
        return np.zeros(3)


def get_ball_position(model, data):
    """Get ball position"""
    try:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'sphere')
        return data.xpos[body_id].copy()
    except:
        return np.zeros(3)


def move_ee_cartesian(model, data, delta_pos):
    """Move end effector in Cartesian space"""
    if abs(delta_pos[0]) > 0.001 or abs(delta_pos[1]) > 0.001:
        angle_change = np.arctan2(delta_pos[1], delta_pos[0]) * 0.1
        data.qpos[0] += angle_change
    
    if abs(delta_pos[2]) > 0.001:
        data.qpos[1] += delta_pos[2] * 0.5
    
    if abs(delta_pos[0]) > 0.001:
        data.qpos[2] += delta_pos[0] * 0.3
    
    if abs(delta_pos[0]) > 0.001 or abs(delta_pos[1]) > 0.001:
        reach = np.sqrt(delta_pos[0]**2 + delta_pos[1]**2)
        data.qpos[3] += reach * 0.5
    
    data.qpos[4] += delta_pos[0] * 0.1
    
    # Clamp joints to limits
    for i in range(7):
        if model.jnt_limited[i]:
            data.qpos[i] = np.clip(data.qpos[i], 
                                   model.jnt_range[i, 0], 
                                   model.jnt_range[i, 1])


def move_joint(model, data, joint_idx, delta):
    """Move a specific joint"""
    if joint_idx < 7:
        data.qpos[joint_idx] += delta
        if model.jnt_limited[joint_idx]:
            data.qpos[joint_idx] = np.clip(data.qpos[joint_idx],
                                           model.jnt_range[joint_idx, 0],
                                           model.jnt_range[joint_idx, 1])


def save_recording():
    """Save recorded data to HDF5 file"""
    if len(recorded_data['timestamps']) == 0:
        print("\n⚠️  No data recorded!")
        return
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"robot_recording_{timestamp_str}.hdf5"
    
    print(f"\n{'='*60}")
    print(f"Saving {len(recorded_data['timestamps'])} frames to {filename}...")
    
    with h5py.File(filename, 'w') as f:
        for key, value in recorded_data.items():
            data_array = np.array(value)
            f.create_dataset(key, data=data_array)
            print(f"  ✓ {key}: shape {data_array.shape}")
        
        f.attrs['num_joints'] = model.nq
        f.attrs['timestep'] = model.opt.timestep
        f.attrs['recording_duration'] = recorded_data['timestamps'][-1]
        f.attrs['num_frames'] = len(recorded_data['timestamps'])
    
    print(f"\n✅ Recording saved!")
    print(f"   Location: ./{filename}")
    print(f"   Duration: {recorded_data['timestamps'][-1]:.2f}s")


def button_pressed(button_id):
    """Check if button was just pressed (not held)"""
    current_state = controller.get_button(button_id)
    was_pressed = last_button_state.get(button_id, False)
    last_button_state[button_id] = current_state
    return current_state and not was_pressed


# ==============================================================================
# MAIN SIMULATION LOOP
# ==============================================================================

print("\n" + "="*60)
print("PS4 CONTROLLER ROBOT CONTROL + RECORDING")
print("="*60)
print("CONTROLS:")
print("  Left Stick      - Move in X/Y plane (forward/back/left/right)")
print("  Right Stick Y   - Move up/down (Z axis)")
print("  L1/R1          - Rotate base joint")
print("  D-Pad Up/Down  - Move shoulder joint")
print("  D-Pad Left/Right - Move elbow joint")
print("\nRECORDING:")
print("  □ (Square)     - Start/Stop recording")
print("  △ (Triangle)   - Reset to home position")
print("  OPTIONS        - Quit and save")
print("="*60 + "\n")

frame_count = 0
last_print_time = time.time()

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and not should_quit:
        step_start = time.time()
        
        # Process pygame events (required for controller input)
        pygame.event.pump()
        
        # Read controller inputs
        delta_pos = np.zeros(3)
        
        # Left stick - X/Y movement
        left_x = apply_deadzone(controller.get_axis(PS4_AXES['LEFT_X']), deadzone)
        left_y = apply_deadzone(controller.get_axis(PS4_AXES['LEFT_Y']), deadzone)
        
        delta_pos[0] = -left_y * control_speed  # Forward/backward (inverted)
        delta_pos[1] = left_x * control_speed   # Left/right
        
        # Right stick Y - Z movement
        right_y = apply_deadzone(controller.get_axis(PS4_AXES['RIGHT_Y']), deadzone)
        delta_pos[2] = -right_y * control_speed  # Up/down (inverted)
        
        # Apply Cartesian movement
        if np.any(np.abs(delta_pos) > 0.0001):
            move_ee_cartesian(model, data, delta_pos)
        
        # L1/R1 - Base rotation
        if controller.get_button(PS4_BUTTONS['L1']):
            move_joint(model, data, 0, -joint_speed)
        if controller.get_button(PS4_BUTTONS['R1']):
            move_joint(model, data, 0, joint_speed)
        
        # D-Pad - Joint control
        if controller.get_button(PS4_BUTTONS['DPAD_UP']):
            move_joint(model, data, 1, joint_speed)
        if controller.get_button(PS4_BUTTONS['DPAD_DOWN']):
            move_joint(model, data, 1, -joint_speed)
        if controller.get_button(PS4_BUTTONS['DPAD_LEFT']):
            move_joint(model, data, 3, -joint_speed)
        if controller.get_button(PS4_BUTTONS['DPAD_RIGHT']):
            move_joint(model, data, 3, joint_speed)
        
        # Button actions (use button_pressed to avoid repeats)
        if button_pressed(PS4_BUTTONS['SQUARE']):
            if not is_recording:
                is_recording = True
                recording_start_time = time.time()
                print("\n🔴 RECORDING STARTED")
            else:
                is_recording = False
                print("\n⏸️  RECORDING STOPPED")
                save_recording()
        
        if button_pressed(PS4_BUTTONS['TRIANGLE']):
            # Reset to home position
            data.qpos[:7] = [0, -1.18, 0, 2.18, 0, 0.57, 3.3161]
            print("\n🏠 Reset to home position")
        
        if button_pressed(PS4_BUTTONS['OPTIONS']):
            should_quit = True
            if is_recording:
                is_recording = False
                save_recording()
            print("\n👋 Quitting...")
        
        # Step simulation
        mujoco.mj_step(model, data)
        
        # Record data if recording
        if is_recording:
            current_time = time.time() - recording_start_time
            
            # Store controller state
            controller_state = {
                'left_stick': [left_x, left_y],
                'right_stick_y': right_y,
                'delta_pos': delta_pos.copy()
            }
            
            recorded_data['joint_positions'].append(data.qpos.copy())
            recorded_data['joint_velocities'].append(data.qvel.copy())
            recorded_data['ee_positions'].append(get_ee_position(model, data))
            recorded_data['ball_positions'].append(get_ball_position(model, data))
            recorded_data['timestamps'].append(current_time)
            recorded_data['controller_inputs'].append([left_x, left_y, right_y])
            
            frame_count += 1
            
            # Print status every 2 seconds
            if time.time() - last_print_time > 2.0:
                ee_pos = get_ee_position(model, data)
                print(f"🔴 Recording... {frame_count} frames ({current_time:.1f}s) | "
                      f"EE: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
                last_print_time = time.time()
        
        # Sync viewer
        viewer.sync()
        
        # Maintain real-time
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

# Cleanup
pygame.quit()

# Save if still recording
if is_recording:
    save_recording()

print("\n✅ Simulation end