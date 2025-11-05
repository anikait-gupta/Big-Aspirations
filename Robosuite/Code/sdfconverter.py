import trimesh
import numpy as np
from mesh_to_sdf import mesh_to_sdf, sample_sdf_near_surface
import skimage.measure
import marching_cubes as mc # You might need another library for direct SDF volume export


# Load your STL file
mesh = trimesh.load('ball_thrower_0.stl')


# Ensure the mesh is watertight for best results (trimesh can sometimes automatically handle this)
# If the mesh has issues, the SDF might be inaccurate.


# Option 1: Sample SDF on a uniform grid (more comprehensive for a full SDF volume)
# Define the bounds of the grid
bounds = mesh.bounds
# Create a grid of points within those bounds
points = np.mgrid[bounds[0,0]:bounds[1,0]:64j, 
                  bounds[0,1]:bounds[1,1]:64j, 
                  bounds[0,2]:bounds[1,2]:64j]
points = points.reshape(3, -1).T


# Calculate SDF values for the points
sdf = mesh_to_sdf(mesh, points)


# The 'sdf' variable now holds the signed distance for each point in the grid.
# You would need to save this data in a format appropriate for your target application
# (e.g., a raw data file, a NumPy file (.npy), or a specific SDF format for a simulator like Gazebo).


# Example: Save the raw numpy data to a file
np.save('ball_thrower_0_sdf.npy', sdf)




