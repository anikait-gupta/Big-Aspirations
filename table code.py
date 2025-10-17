from robosuite.utils.mjcf_utils import new_body, new_geom


table_rectangle = new_body(name="custom_table", pos="0 0 1.016") #Recording the position here should make it stationary?


# Convert inches to meters: 30in = 0.762m, 60in = 1.524m, 1in = 0.0254m, 40in = 1.016m
table_geom = new_geom(
    geom_type="box",
    name="robot_table",
    size=[0.762, 0.381, 0.0127],  #  [30in/2, 60in/2, 1in/2]
    pos=[0, 0, 0],  # Stationary relative to body, line 4
    rgba=[0.6, 0.4, 0.2, 1],  # Brown wood color
    friction=[1.0, 0.005, 0.0001], #Can be changed to fit real scenario (I dont know how to judge the friction)
    mass=50.0,  # Heavy so it doesn't move, but this shouldnt matter anyway
    contype=1,  # Enables collision
    conaffinity=1
)
table_rectangle.append(table_geom)

# Adds it to world
self.world.worldbody.append(table_rectangle)