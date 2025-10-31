import numpy as np
import random # will be used to remove some validation set from the training set
import torch # deep learning library as it is efficient
import torchnet as tnt # type: ignore # to handle a bunch of files to put in a variable

# Plot Libraries
import open3d as o3d 

# Utility Libraries
from glob import glob
import os
import functools

# # Data Augmentation Libraries
# import cv2
# from skimage import io
# from skimage import transform as tf
# from skimage import img_as_ubyte
# from skimage.util import random_noise

# 1-Data path setup

# specify the data path and extract the file name
project_dir = "./DATA/"
pointcloud_train_files = glob(os.path.join(project_dir, "train/*.txt"))
pointcloud_test_files = glob(os.path.join(project_dir, "test/*.txt"))

# # 2- Train, Test and Validation Set creation
# Randomly selects 20% of the training file indices to be used as validation data.
valid_index = np.random.choice(len(pointcloud_train_files), int(len(pointcloud_train_files)/5), replace=False)

# Creates a list of validation files using the selected indices.
valid_list = [pointcloud_train_files[i] for i in valid_index]

# Creates a list of training files using the indices that were not selected for validation.
train_list = [pointcloud_train_files[i] for i in np.setdiff1d(range(len(pointcloud_train_files)), valid_index)]

# Defines the list of test files.
test_list = pointcloud_test_files

print("%d titles in train set- %d titles in valid set- %d  titles in valid test" % (len(train_list), len(valid_list), len(test_list)))

# Data Analysis
# Sets numpy print options to display 3 decimal places
np.set_printoptions(precision=3)

# Randomly selects a file from the list of point cloud training files
tile_selected = pointcloud_train_files[random.randrange(20)]
print("Title selected: ", tile_selected)  # Prints the name of the selected file

# Loads the data from the selected file into a numpy array
temp = np.loadtxt(tile_selected)

# Prints the median of each column of the data
print("median\n", np.median(temp, axis=0))

# Prints the standard deviation of each column of the data
print("std\n", np.std(temp, axis=0))

# Prints the minimum value of each column of the data
print("min\n", np.min(temp, axis=0))

# Prints the maximum value of each column of the data
print("max\n", np.max(temp, axis=0))


# Computing the mean and the min of a data title
# Transposes the data to facilitate index selection.
# This means that now each row represents a feature (e.g., x, y, z, intensity) and each column represents a point in the point cloud.
cloud_data = temp.transpose()

# Calculates the minimum value of each feature.
min_f = np.min(cloud_data, axis=1)
# Calculates the mean of each feature.
mean_f = np.mean(cloud_data, axis=1)

# Prints the minimum value of each feature.
print("min transpose\n", min_f)
# Prints the mean of each feature.
print("mean transpose\n", mean_f)

# Normalizes the point cloud coordinates.
# First, selects the first three rows of the data, which represent the x, y, and z coordinates.
n_coords = cloud_data[0:3]

# Subtracts the mean from the x and y coordinates, and the minimum value from the z coordinate.
# This has the effect of centering the point cloud around the origin (0,0,0).
n_coords[0] -= mean_f[0]
n_coords[1] -= mean_f[1]
n_coords[2] -= min_f[2] # "level" the point cloud along the z axis, ensuring that all point clouds are at the same scale and relative position.

# Prints the normalized coordinates.
print("n_coords\n", n_coords)

# Normalize the intensity
# to have something more robust and scalable
# which means we will use the interquartile range which is the difference between the 75th and 25th quantile

# The interquartile range (IQR) is the difference between the 75th and 25th percentile.
# We are calculating this for the intensity of the points in the point cloud.
# cloud_data[-2] refers to the second-to-last column of the data, which presumably contains the intensity values.
IQR = np.quantile(cloud_data[-2], 0.75) - np.quantile(cloud_data[-2], 0.25)

# We subtract the median from all observations and then divide by the interquartile range.
# This has the effect of scaling the data so that the median is 0 and most of the data is between -0.5 and 0.5.
# This also has the effect of reducing the impact of outliers (values that are significantly larger or smaller than most other values).
n_intensity = ((cloud_data[-2] - np.median(cloud_data[-2])) / IQR)

# Subtracts the minimum value from the normalized intensity.
# This ensures that all values are positive, with the smallest value being 0.
n_intensity -= np.min(n_intensity)

# Prints the normalized intensity.
print("n_intensity\n", n_intensity)

# Cloud load function
# for each cloud that is loaded we want to apply normalization

# The cloud_loader function receives the file name (tile_name) and a list of features to be used (features_used).
def cloud_loader(tile_name, features_used):
    # Loads the data from the file and transposes the resulting array.
    # The transposition is done to facilitate data manipulation later.
    cloud_data = np.loadtxt(tile_name).transpose()

    # Calculates the minimum value and mean of each feature.
    min_f = np.min(cloud_data, axis=1)
    mean_f = np.mean(cloud_data, axis=1)

    # Creates an empty list to store the selected features.
    features = []

    # If 'xyz' is in the list of features used, normalizes the x, y, and z coordinates and adds them to the feature list.
    if 'xyz' in features_used:
        n_coords = cloud_data[0:3]
        n_coords[0] -= mean_f[0]
        n_coords[1] -= mean_f[1]
        n_coords[2] -= min_f[2]
        features.append(n_coords)

    # If 'rgb' is in the list of features used, adds the colors to the feature list.
    if 'rgb' in features_used:
        colors = cloud_data[3:6]
        features.append(colors)

    # If 'i' is in the list of features used, normalizes the intensity and adds it to the feature list.
    if 'i' in features_used:
        IQR = np.quantile(cloud_data[-2], 0.75) - np.quantile(cloud_data[-2], 0.25)
        n_intensity = ((cloud_data[-2] - np.median(cloud_data[-2])) / IQR)
        n_intensity -= np.min(n_intensity)
        features.append(n_intensity)

    # The last row of the data is assumed to be the ground truth and is converted to a PyTorch tensor.
    gt = cloud_data[-1]
    gt = torch.from_numpy(gt).long()

    # The selected features are stacked vertically and converted to a PyTorch tensor.
    cloud_data = torch.from_numpy(np.vstack(features))

    # The function returns the preprocessed point cloud and the ground truth.
    return cloud_data, gt


print("cloud_loader function created", cloud_loader(tile_selected, "xyzrgbi"))

# train, test, validation split dataset
# Defines the features that will be used to load the point cloud data.
cloud_features = "xyzrgbi"

# Creates a test dataset.
# The cloud_loader function is used to load the data, with the features defined above.
# The test_list list contains the names of the files that will be used for the test set.
test_set = tnt.dataset.ListDataset(test_list, functools.partial(cloud_loader, features_used=cloud_features))

# Creates a training dataset in a similar way to the test set.
train_set = tnt.dataset.ListDataset(train_list, functools.partial(cloud_loader, features_used=cloud_features))

# Creates a validation dataset in a similar way to the test and training sets.
valid_set = tnt.dataset.ListDataset(valid_list, functools.partial(cloud_loader, features_used=cloud_features))

# Prints the first element of the test set.
print("test_set", test_set[0])


# Title visualization function
# Defines the title visualization function.
def title_visualization(title_name, features_used='xyzrgbi'):
    # Loads the point cloud and ground truth (gt) data using the cloud_loader function.
    cloud, gt = cloud_loader(title_name, features_used)

    # Prepares the data for Open3D, a library used for 3D visualization.
    # Extracts the x, y, z coordinates from the point cloud and transposes them to the correct format.
    xyz = np.array(cloud[0:3]).transpose()
    # Creates a PointCloud object, which is used by Open3D to store and manipulate point clouds.
    pcd = o3d.geometry.PointCloud()
    # Assigns the x, y, z coordinates to the point cloud.
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # If RGB colors are present in the data, extracts them and normalizes them to the [0, 1] range.
    if 'rgb' in features_used:
        rgb = np.array(cloud[3:6]/255).transpose()
        # Assigns the colors to the point cloud.
        pcd.colors = o3d.utility.Vector3dVector(rgb)

    # Estimates the normals of the point cloud, which are used to improve 3D visualization.
    pcd.estimate_normals(fast_normal_computation=True)
    # Draws the point cloud using Open3D.
    o3d.visualization.draw_geometries([pcd])

    return

# Selects a title from the test list.
selection = test_list[9]
# Calls the title visualization function with the selected title.
title_visualization(selection)