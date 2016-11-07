# micro-mouse

improvements:
* when reseting to a new path, robot does not take example of multiple movements
* robot does not continue to explore after reaching goal during first run.
* update g_values if car encounters a location it has already been, and the g_values are inaccurate -- requires propagation through graph
* if the robot can *see* (utilizing more of the sensor data) a square it has already been to, we need to propagate the g-values as well.