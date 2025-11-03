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
from pynput import keyboard


# Stabilize
for _ in range(10):
    mujoco.mj_step(model, data)


# ==============================================================================
# CONTROL AND RECORDING SETUP
# ==============================================================================

# Control parameters
control_speed = 0.01  # meters per step
z_speed = 0.01
joint_speed = 0.05    # radians per step

# Keyboard state
keys_pressed = {
    'w': False, 'a': False, 's': False, 'd': False,
    't': False, 'g': False,
    'i': False, 'j': False, 'k': False, 'l': False,
    'u': False, 'o': False
}

# Recording state
is_recording = False
recording_start_time = None
recorded_data = {
    'joint_positions': [],
    'joint_velocities': [],
    'ee_positions': [],
    'ball_positions': [],
    'timestamps': [],
    'control_inputs': []
}

should_quit = False


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

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
    """
    Move end effector in Cartesian space using simple joint control
    delta_pos: [dx, dy, dz] in meters
    """
    # Get current joint positions (first 7 are arm joints for Sawyer)
    current_qpos = data.qpos[:7].copy()
    
    # Simple incremental movement (not true IK, but works for small movements)
    # Adjust joints 0-6 based on desired Cartesian movement
    
    # Joint 0 (base rotation) - controls X-Y plane rotation
    if abs(delta_pos[0]) > 0.001 or abs(delta_pos[1]) > 0.001:
        angle_change = np.arctan2(delta_pos[1], delta_pos[0]) * 0.1
        data.qpos[0] += angle_change
    
    # Joint 1 (shoulder lift) - helps with Z and reach
    if abs(delta_pos[2]) > 0.001:
        data.qpos[1] += delta_pos[2] * 0.5
    
    # Joint 2 (elbow rotation)
    if abs(delta_pos[0]) > 0.001:
        data.qpos[2] += delta_pos[0] * 0.3
    
    # Joint 3 (elbow flex) - primary for reaching forward/back
    if abs(delta_pos[0]) > 0.001 or abs(delta_pos[1]) > 0.001:
        reach = np.sqrt(delta_pos[0]**2 + delta_pos[1]**2)
        data.qpos[3] += reach * 0.5
    
    # Joint 4 (wrist rotation)
    data.qpos[4] += delta_pos[0] * 0.1
    
    # Clamp joints to their limits
    for i in range(7):
        if model.jnt_limited[i]:
            data.qpos[i] = np.clip(data.qpos[i], 
                                   model.jnt_range[i, 0], 
                                   model.jnt_range[i, 1])


def move_joint(model, data, joint_idx, delta):
    """Move a specific joint by delta amount"""
    if joint_idx < 7:  # Only modify arm joints
        data.qpos[joint_idx] += delta
        # Clamp to limits
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
    print(f"   Frames: {len(recorded_data['timestamps'])}")


# ==============================================================================
# KEYBOARD CALLBACKS
# ==============================================================================

def on_press(key):
    """Handle key press"""
    global is_recording, recording_start_time, should_quit, keys_pressed
    
    try:
        # Letter keys
        if hasattr(key, 'char') and key.char:
            k = key.char.lower()
            
            # Movement keys
            if k in keys_pressed:
                keys_pressed[k] = True
            
            # Recording control
            if k == 'r':
                if not is_recording:
                    is_recording = True
                    recording_start_time = time.time()
                    print("\n🔴 RECORDING STARTED")
                else:
                    is_recording = False
                    print("\n⏸️  RECORDING STOPPED")
                    save_recording()
            
            # Quit
            if k == 'q':
                should_quit = True
                if is_recording:
                    is_recording = False
                    save_recording()
                print("\n👋 Quitting...")
    
    except AttributeError:
        # Special keys
        if key == keyboard.Key.space:
            # Reset robot to home position
            data.qpos[:7] = [0, -1.18, 0, 2.18, 0, 0.57, 3.3161]
            print("\n🏠 Reset to home position")
        elif key == keyboard.Key.esc:
            should_quit = True


def on_release(key):
    """Handle key release"""
    global keys_pressed
    
    try:
        if hasattr(key, 'char') and key.char:
            k = key.char.lower()
            if k in keys_pressed:
                keys_pressed[k] = False
    except AttributeError:
        pass


# Start keyboard listener
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()


# ==============================================================================
# MAIN SIMULATION LOOP
# ==============================================================================

print("\n" + "="*60)
print("ROBOT CONTROL + RECORDING")
print("="*60)
print("CARTESIAN CONTROL (End Effector):")
print("  W/S - Move forward/backward (X)")
print("  A/D - Move left/right (Y)")
print("  T/G - Move up/down (Z)")
print("\nJOINT CONTROL:")
print("  I/K - Joint 0 (base rotation)")
print("  J/L - Joint 1 (shoulder)")
print("  U/O - Joint 3 (elbow)")
print("\nRECORDING:")
print("  R - Start/Stop recording")
print("  Q - Quit and save")
print("  SPACE - Reset to home position")
print("="*60 + "\n")

frame_count = 0
last_print_time = time.time()

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and not should_quit:
        step_start = time.time()
        
        # Process keyboard input for movement
        delta_pos = np.zeros(3)
        
        # WASD + TG for Cartesian control
        if keys_pressed['w']:
            delta_pos[0] += control_speed  # Forward (X+)
        if keys_pressed['s']:
            delta_pos[0] -= control_speed  # Backward (X-)
        if keys_pressed['d']:
            delta_pos[1] += control_speed  # Right (Y+)
        if keys_pressed['a']:
            delta_pos[1] -= control_speed  # Left (Y-)
        if keys_pressed['t']:
            delta_pos[2] += z_speed        # Up (Z+)
        if keys_pressed['g']:
            delta_pos[2] -= z_speed        # Down (Z-)
        
        # Apply Cartesian movement
        if np.any(np.abs(delta_pos) > 0.0001):
            move_ee_cartesian(model, data, delta_pos)
        
        # Joint control
        if keys_pressed['i']:
            move_joint(model, data, 0, joint_speed)
        if keys_pressed['k']:
            move_joint(model, data, 0, -joint_speed)
        if keys_pressed['j']:
            move_joint(model, data, 1, joint_speed)
        if keys_pressed['l']:
            move_joint(model, data, 1, -joint_speed)
        if keys_pressed['u']:
            move_joint(model, data, 3, joint_speed)
        if keys_pressed['o']:
            move_joint(model, data, 3, -joint_speed)
        
        # Step simulation
        mujoco.mj_step(model, data)
        
        # Record data if recording
        if is_recording:
            current_time = time.time() - recording_start_time
            
            recorded_data['joint_positions'].append(data.qpos.copy())
            recorded_data['joint_velocities'].append(data.qvel.copy())
            recorded_data['ee_positions'].append(get_ee_position(model, data))
            recorded_data['ball_positions'].append(get_ball_position(model, data))
            recorded_data['timestamps'].append(current_time)
            recorded_data['control_inputs'].append(delta_pos.copy())
            
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

# Stop keyboard listener
listener.stop()

# Save if still recording
if is_recording:
    save_recording()

print("\n✅ Simulation ended")