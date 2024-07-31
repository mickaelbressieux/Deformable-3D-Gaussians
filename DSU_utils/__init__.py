import numpy as np
import os
import torch
import math

from sklearn.cluster import KMeans

import pdb


def create_dynamic_mask(d_xyz: torch.Tensor, nb_clusters: int = 2) -> torch.Tensor:
    """
    Create a dynamic mask based on the total distance travelled by each gaussians
    :param d_xyz: displacement of each gaussians, size (nb_timesteps, nb_gaussians, 3)
    :return: Mask, size (nb_gaussians)
    """

    dist = torch.norm(d_xyz[1:, :, :] - d_xyz[:-1, :, :], dim=2)
    dist = torch.sum(dist, dim=0)

    dist = dist.cpu().numpy()

    # Use KMeans to find the cluster with the highest distance
    kmeans = KMeans(n_clusters=nb_clusters, random_state=0).fit(dist.reshape(-1, 1))
    mask = np.zeros_like(dist)

    # find the cluster with the highest distance
    max_cluster = np.argmax(kmeans.cluster_centers_)
    mask[kmeans.labels_ == max_cluster] = 1

    mask = torch.tensor(mask).to(d_xyz.device)

    print(f"Number of gaussians in the dynamic mask: {mask.sum()}")
    print(f"Number of gaussians not in the dynamic mask: {(1 - mask).sum()}")

    return mask


def identify_rigid_object(
    xyz: torch.Tensor, d_xyz: torch.Tensor, datapath: str
) -> torch.Tensor:
    """
    Identify rigid objects in the scene
    :param xyz: gaussians, size (nb_gaussians, 3)
    :param d_xyz: displacement of each gaussians, size (nb_timesteps, nb_gaussians, 3)
    :param datapath: path to save the rigid motion
    :return: mask, size (nb_gaussians)
    """
    xyz = xyz.cpu().numpy()
    d_xyz = d_xyz.cpu().numpy()
    d_xyz = d_xyz.transpose(1, 2, 0)

    # Use RANSAC to estimate if some gaussians are moving in rigid motion
    # A rigid motion is a motion that can be described by a rotation and a translation
    # We can estimate the rotation and translation using RANSAC
    # Then we can remove the first rigid motion from the data and repeat the process to find the next rigid motion

    nb_rigid_motions = 4

    inliers_arr = []
    rigid_rot = []
    rigid_t = []

    for i in range(nb_rigid_motions):

        # step 2: use RANSAC to find the rotation and translation that best fit the data
        ransac = Ransac_Def3DGS(
            xyz, d_xyz, error_threshold=0.01, num_samples=1000, max_trials=3
        )
        R, t = ransac.fit()

        # Get the inliers and outliers
        inliers = ransac.inliers_max
        outliers = ~inliers

        # Print the number of inliers
        print(f"Number of inliers: {inliers.sum()}")
        print(f"Number of outliers: {outliers.sum()}")

        # Check if RANSAC found a valid consensus set
        if ransac.inliers_max.sum() == 0:
            print(
                "RANSAC could not find a valid consensus set. Try adjusting the parameters."
            )
            print(f"Stopping regression at iteration {i}")
            break
        else:
            # step 7: remove the inliers and repeat the process
            xyz = xyz[outliers]
            d_xyz = d_xyz[outliers]

        if outliers.sum() == 0:
            print(f"No more outliers. Stopping regression at iteration {i}")
            break

        # step 8: store the rigid motion
        rigid_rot.append(R)
        rigid_t.append(t)

        # step 9: store the inliers and restore initial size using former inliers
        inliers_arr.append(inliers)

    rigid_rot = np.array(rigid_rot)
    rigid_t = np.array(rigid_t)
    rigid_objects = recreate_full_inliers(inliers_arr)

    np.save(datapath + "/rigid_rot.npy", rigid_rot)
    np.save(datapath + "/rigid_t.npy", rigid_t)
    np.save(datapath + "/inliers.npy", inliers)

    rigid_objects = torch.tensor(rigid_objects)

    return rigid_objects


def create_all_d_xyz(deform, xyz, scene):
    viewpoint_stack = scene.getTrainCameras().copy()
    timestamps = []
    for view in viewpoint_stack:
        timestamps.append(view.fid)
    timestamps.sort()

    all_d_xyz = []

    for fid in timestamps:
        N = xyz.shape[0]
        time_input = fid.unsqueeze(0).expand(
            N, -1
        )  # take the frame id and expand it to the number of gaussians.

        d_xyz, _, _ = deform.step(
            xyz.detach(), time_input
        )  # we detach the gaussians to avoid backpropagation through them

        all_d_xyz.append(d_xyz)

    all_d_xyz = torch.stack(all_d_xyz, dim=0).to(xyz.device)

    return all_d_xyz


class Ransac_Def3DGS:
    def __init__(
        self, xyz, d_xyz, error_threshold=0.05, num_samples=1000, max_trials=100
    ):

        self.xyz = xyz
        self.d_xyz = d_xyz

        self.nb_rigid_motions = 4

        self.inliers_arr = []
        self.rigid_rot = []
        self.rigid_t = []

        self.R = None
        self.t = None

        self.inliers = None

        self.error_threshold = error_threshold
        self.num_samples = num_samples
        self.max_trials = max_trials

        self.inliers_max = None

    def recreate_full_inliers(self, inliers_arr):
        for j in range(len(inliers_arr)):
            num = len(inliers_arr) - j - 1
            idx = np.array(np.where(inliers_arr[num] == True))[0]

            final_size = inliers_arr[num].shape[0]

            if j == 0:
                temp_inliers = np.full(final_size, len(inliers_arr))
            else:
                temp_inliers = final_inliers.copy()

            final_inliers = np.full(final_size, num)

            k = 0
            for i in range(final_size):
                if i not in idx:
                    final_inliers[i] = temp_inliers[k]
                    k += 1

        return final_inliers

    def rigid_transform_3D_all_timestamps(self, A, B):
        # A is the canonical space, size nb_gaussx3
        # B is the 3D gaussian displaced gaussians, size timestampxnb_gaussx3

        assert A.shape[0] == B.shape[0]

        R = np.zeros((B.shape[2], 3, 3))
        t = np.zeros((B.shape[2], 3))

        A_centered = A - A.mean(axis=0)
        B_centered = B - B.mean(axis=0)[np.newaxis, :, :]

        H = np.einsum("ki,kjl->ijl", A_centered, B_centered)  # correlation H = A^T B

        for i in range(B_centered.shape[2]):
            U, S, Vt = np.linalg.svd(H[:, :, i])
            R[i] = Vt.T @ U.T
            if np.linalg.det(R[i]) < 0:
                Vt[-1, :] *= -1
                R[i] = Vt.T @ U.T
            t[i] = B[:, :, i].mean(axis=0) - R[i] @ A.mean(axis=0)
        R = R.transpose(1, 2, 0)
        t = t.transpose()

        return R, t[np.newaxis, :, :]

    def apply_transformation_all_timestamps(self, xyz, R, t):
        # xyz is the canonical space, size nb_gaussx3
        # R is the rotation, size timestampx3x3
        # t is the translation, size timestampx3
        return np.einsum("jk,kli->jli", xyz, R) + t

    def loss_function(self, y_true, y_pred):
        # input:
        # y_true is nb_gaussx3xt
        # y_pred is nb_gaussx3xt

        # output:
        # error is nb_gauss and represent the maximum distance between said gaussian prediction and true position among all timesteps

        error = np.linalg.norm(y_true - y_pred, axis=1)  # error norm nb_gaussxtimesteps
        error = np.max(
            error, axis=1
        )  # max error accross time for each gaussian nb_gauss

        return error

    def fit(self):
        # fit function without using sklearn
        # output is the rotation and translation that best fits the data at each timestamp, size of rotation timestampx3x3, size of translation timestampx3

        max_num_inliers = 0
        current_R = None
        current_t = None
        max_inliers = np.array([False])

        for i in range(self.max_trials):

            # step 1: take a subset of the data
            idx = np.random.choice(
                self.d_xyz.shape[0], size=self.num_samples, replace=False
            )
            subset1 = self.xyz[idx]
            subset2 = self.xyz[idx][:, :, np.newaxis] + self.d_xyz[idx]

            # step 2: fit the model on the subset
            R, t = self.rigid_transform_3D_all_timestamps(subset1, subset2)

            # step 3: apply the transformation to the whole data
            transformed_d_xyz = self.apply_transformation_all_timestamps(self.xyz, R, t)

            # step 4: compute the error
            error = self.loss_function(
                self.xyz[:, :, np.newaxis] + self.d_xyz, transformed_d_xyz
            )

            # step 5: create a mask of inliers
            inliers = error < self.error_threshold
            print(f"before refit: {inliers.sum()}")

            # step 6: refit the model on the inliers
            R, t = self.rigid_transform_3D_all_timestamps(
                self.xyz[inliers],
                self.xyz[inliers][:, :, np.newaxis] + self.d_xyz[inliers],
            )

            # step 7: apply the transformation to the whole data
            transformed_d_xyz = self.apply_transformation_all_timestamps(self.xyz, R, t)

            # step 8: compute the error
            error = self.loss_function(
                self.xyz[:, :, np.newaxis] + self.d_xyz, transformed_d_xyz
            )

            # step 9: ceate a mask of inliers
            inliers = error < self.error_threshold
            print(f"after refit: {inliers.sum()}")

            if inliers.sum() > max_num_inliers:
                current_R = R
                current_t = t
                max_num_inliers = inliers.sum()
                max_inliers = inliers

        self.R = current_R
        self.t = current_t
        self.inliers = max_num_inliers
        self.inliers_max = max_inliers

        return self.R, self.t


def recreate_full_inliers(inliers_arr):
    for j in range(len(inliers_arr)):
        num = len(inliers_arr) - j - 1
        idx = np.array(np.where(inliers_arr[num] == True))[0]

        final_size = inliers_arr[num].shape[0]

        if j == 0:
            temp_inliers = np.full(final_size, len(inliers_arr))
        else:
            temp_inliers = final_inliers.copy()

        final_inliers = np.full(final_size, num)

        k = 0
        for i in range(final_size):
            if i not in idx:
                final_inliers[i] = temp_inliers[k]
                k += 1

    return final_inliers
