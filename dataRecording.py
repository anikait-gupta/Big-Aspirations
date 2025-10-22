
# ==============================================================================
# RECORDING SETUP
# ==============================================================================

# Storage dictionary for all recorded data
recorded_data = {
    'joint_positions': [],      # qpos - all joint angles
    'joint_velocities': [],     # qvel - all joint velocities
    'timestamps': [],           # Time in seconds
    'ee_positions': [],         # End effector XYZ position
    'ball_positions': []        # Ball XYZ position
}

# Track recording state
is_recording = False
start_time = None
recording_start_real_time = None

print("\n" + "="*60)
print("TELEOPERATION RECORDING MODE")
print("="*60)
print("Controls:")
print("  SPACE - Start/Stop recording")
print("  ESC   - Quit and save")
print("  (Use MuJoCo viewer controls to move the robot)")
print("="*60 + "\n")
#Still needs propper keyboard handling

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        
        # Step the physics simulation
        mujoco.mj_step(model, data)
        
        # Check for spacebar press to toggle recording
        # This will need to be done later
       
        
        # RECORD DATA if recording is active
        if is_recording:
            # Get current timestamp relative to recording start
            current_time = time.time() - recording_start_real_time
            
            # Record joint positions (data.qpos contains ALL position variables)
            # For Sawyer: first 7 values are the arm joints
            recorded_data['joint_positions'].append(data.qpos.copy())
            
            # Record joint velocities (data.qvel contains ALL velocity variables)
            recorded_data['joint_velocities'].append(data.qvel.copy())
            
            # Record timestamp
            recorded_data['timestamps'].append(current_time)
            
            # Get end effector position (if grip_site exists)
            try:
                grip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'grip_site')
                ee_pos = data.site_xpos[grip_site_id].copy()
                recorded_data['ee_positions'].append(ee_pos)
            except:
                recorded_data['ee_positions'].append(np.zeros(3))
            
            # Get ball position
            try:
                ball_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'sphere')
                ball_pos = data.xpos[ball_body_id].copy()
                recorded_data['ball_positions'].append(ball_pos)
            except:
                recorded_data['ball_positions'].append(np.zeros(3))
        
        # Sync viewer
        viewer.sync()
        
        # Maintain real-time execution
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        
        # Simple recording toggle 
        # Needs proper keyboard handling here
        if data.time > 5 and not is_recording and start_time is None:
            is_recording = True
            start_time = data.time
            recording_start_real_time = time.time()
            print(f"[{data.time:.2f}s] RECORDING STARTED")
        
        # Auto-stop after 30 seconds of recording
        if is_recording and (time.time() - recording_start_real_time) > 30:
            is_recording = False
            print(f"[{data.time:.2f}s] RECORDING STOPPED")
            break


# ==============================================================================
# SAVE RECORDED DATA
# ==============================================================================

if len(recorded_data['timestamps']) > 0:
    # Create filename with timestamp
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"robot_recording_{timestamp_str}.hdf5"
    
    print(f"\nSaving {len(recorded_data['timestamps'])} frames to {filename}...")
    
    # Save to HDF5 file
    with h5py.File(filename, 'w') as f:
        # Save each data type as a dataset
        for key, value in recorded_data.items():
            data_array = np.array(value)
            f.create_dataset(key, data=data_array)
            print(f"  - {key}: shape {data_array.shape}")
        
        # Save metadata
        f.attrs['num_joints'] = model.nq
        f.attrs['timestep'] = model.opt.timestep
        f.attrs['recording_duration'] = recorded_data['timestamps'][-1]
    
    print(f"\n Recording saved successfully!")
    print(f"  Location: ./{filename}")
    print(f"  Duration: {recorded_data['timestamps'][-1]:.2f} seconds")
    print(f"  Frames: {len(recorded_data['timestamps'])}")
    
    # ==============================================================================
    # HOW TO LOAD THE DATA LATER
    # ==============================================================================
    print("\n" + "="*60)
    print("To load this data later, use:")
    print("="*60)
    print(f"""
import h5py
import numpy as np

# Load the recorded data
with h5py.File('{filename}', 'r') as f:
    joint_positions = f['joint_positions'][:]
    joint_velocities = f['joint_velocities'][:]
    timestamps = f['timestamps'][:]
    ee_positions = f['ee_positions'][:]
    
    print(f"Loaded {{len(timestamps)}} frames")
    print(f"Joint positions shape: {{joint_positions.shape}}")
    
# Now you can replay or analyze the trajectory
for i in range(len(timestamps)):
    # Set robot to recorded position
    # data.qpos[:] = joint_positions[i]
    # data.qvel[:] = joint_velocities[i]
    pass
    """)
else:
    print("\nNo data recorded!")