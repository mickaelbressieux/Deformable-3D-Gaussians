# Code associated with the Capstone Project "Dynamic Scene Understanding: A Novel Approach to Optimize Training Efficiency in 3D Gaussian Splatting"

Author: Mickaël Bressieux

Source: Deformable 3DGS: https://github.com/ingra14m/Deformable-3D-Gaussians
For more information on 98% of the code, refer to its github README.


Novel things:

Most of the new code is in DSU_utils. The threshold hyperparameters are hardcoded there.

The number of splits and the iteration at which they happen must be specified as an argument (--dynamic_seg_iterations in train.py)
I added 100 intermediate renderings for full images and also static/dynamic seg (--render_intermediate to specify the wanted iterations in train.py, be careful of densification as it happens for 100 iterations)
RANSAC algo can be executed and its rigid object detection will be saved in "inliers.npy" (--rigid_object_iterations)


