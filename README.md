# Structure_from_Motion
 
 The goal of this project is to construct a 3D model from two 2D photos taken in different angles with common visible points.
 
 We will first recover the fundamental matrix to calculate camera poses. Note there are 4 different combination of camera poses given one fundamental matrix. Hence, we will need to disambiguate and select the correct one combination. After we have the correct camera poses, we will use triangulation to calculate the 3D coordinates of a given common visible point.
 
 We will then illustrate how to construct a 3D model with more than 2 cameras. Namely, we will use 10 cameras to construct a 3D model in our case. We will need to use some numerical method in triangulation because we need to calculate the 3D point that has the least error to the predicted points from all 10 cameras. After that, we will apply bundle adjustment to optimize the positions of cameras and the predicted 3D points recursively to achieve the best result possible.