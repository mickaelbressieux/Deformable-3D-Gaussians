#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene, DeformModel
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_two_sets
import torchvision
from utils.general_utils import safe_state
from utils.pose_utils import pose_spherical, render_wander_path
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import imageio
import numpy as np
import time

import pdb as pdb

from gaussian_renderer import quaternion_multiply, quaternion_multiply_batched


def matrix_to_quaternion(R):
    # Assuming R is a 3x3 rotation matrix
    q = np.empty((4,), dtype=np.float32)
    t = np.trace(R)
    if t > 0.0:
        t = np.sqrt(t + 1.0)
        q[0] = 0.5 * t
        t = 0.5 / t
        q[1] = (R[2, 1] - R[1, 2]) * t
        q[2] = (R[0, 2] - R[2, 0]) * t
        q[3] = (R[1, 0] - R[0, 1]) * t
    else:
        i = np.argmax([R[0, 0], R[1, 1], R[2, 2]])
        j = (i + 1) % 3
        k = (j + 1) % 3
        t = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0)
        q[i + 1] = 0.5 * t
        t = 0.5 / t
        q[0] = (R[k, j] - R[j, k]) * t
        q[j + 1] = (R[j, i] + R[i, j]) * t
        q[k + 1] = (R[k, i] + R[i, k]) * t
    return torch.from_numpy(q)


def segment_dynamic_gaussian(
    model_path,
    load2gpu_on_the_fly,
    is_6dof,
    name,
    iteration,
    views,
    gaussians_dyn,
    gaussians_stat,
    pipeline,
    background,
    deform,
):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    moving_path = os.path.join(model_path, name, "ours_{}".format(iteration), "dynamic")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(moving_path, exist_ok=True)

    t_list = []

    new_d_rotation = []
    new_d_scaling = []
    new_d_xyz = []

    # render all views but segment the most dynamic gaussian
    for idx, view in enumerate(tqdm(views, desc="Rendering progress - segmenting")):
        if load2gpu_on_the_fly:
            view.load2device()
        fid = view.fid

        first_fid = 0

        if (idx == 0) and (fid != 0):
            # stop in an error:
            # raise ValueError("The first view should be at time 0")
            print(f"the canonical space is not created at time 0, but at time {fid}")
            first_fid = fid

        xyz = gaussians_dyn.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        render_full_moving_stat = True

        if fid == first_fid:
            orig_d_xyz = d_xyz

        if os.path.exists(os.path.join(model_path, "inliers.npy")):
            inliers = np.load(os.path.join(model_path, "inliers.npy"))  # N
        else:
            inliers = None

        if gaussians_stat.get_xyz.shape[0] > 0:
            results = render_two_sets(
                view,
                gaussians_dyn,
                gaussians_stat,
                pipeline,
                background,
                d_xyz,
                d_rotation,
                d_scaling,
                is_6dof,
                can_d_xyz=d_xyz - orig_d_xyz,
                root=args.model_path,
                name_iter=str(iteration),
                name_view=str(idx),
                inliers=inliers,
                render_full_moving_stat=render_full_moving_stat,
            )
        else:
            results = render(
                view,
                gaussians_dyn,
                pipeline,
                background,
                d_xyz,
                d_rotation,
                d_scaling,
                is_6dof,
                flag_segment=flag_segment,
                can_d_xyz=d_xyz - orig_d_xyz,
                root=args.model_path,
                name_iter=str(iteration),
                name_view=str(idx),
                inliers=inliers,
            )
        rendering = results["render"]
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        if fid == first_fid:
            # create the canonical space at first time step
            can_means3D = results["means3D"]
            can_xyz = gaussians_dyn.get_xyz  # for debugging
            can_rotation = results["rotation"]
            can_scaling = results["scaling"]

            can_d_xyz = d_xyz.cpu().numpy()
            can_d_rotation = d_rotation.cpu().numpy()
            can_d_scaling = d_scaling.cpu().numpy()

            new_d_xyz.append(np.zeros_like(can_d_xyz))
            new_d_rotation.append(np.zeros_like(can_d_rotation))
            new_d_scaling.append(np.zeros_like(can_d_scaling))

        else:
            # compute the displacement in canonical space
            new_d_xyz.append(d_xyz.cpu().numpy() - can_d_xyz)
            new_d_rotation.append(d_rotation.cpu().numpy() - can_d_rotation)
            new_d_scaling.append(d_scaling.cpu().numpy() - can_d_scaling)

        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(
            rendering,
            os.path.join(render_path, "{0:05d}".format(idx) + ".png"),
        )
        torchvision.utils.save_image(
            gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png")
        )
        torchvision.utils.save_image(
            depth, os.path.join(depth_path, "{0:05d}".format(idx) + ".png")
        )
        """torchvision.utils.save_image(
            results["render_moving"],
            os.path.join(moving_path, "{0:05d}".format(idx) + ".png"),
        )"""
    new_d_xyz = np.array(new_d_xyz)
    new_d_rotation = np.array(new_d_rotation)
    new_d_scaling = np.array(new_d_scaling)

    np.save(
        os.path.join(model_path, "can_d_xyz.npy"),
        new_d_xyz,
    )
    np.save(
        os.path.join(model_path, "can_d_rotation.npy"),
        new_d_rotation,
    )
    np.save(
        os.path.join(model_path, "can_d_scaling.npy"),
        new_d_scaling,
    )
    np.save(
        os.path.join(model_path, "can_means3D.npy"),
        can_means3D.cpu().numpy(),
    )
    np.save(
        os.path.join(model_path, "can_xyz.npy"),
        can_xyz.cpu().numpy(),
    )
    np.save(
        os.path.join(model_path, "can_rotation.npy"),
        can_rotation.cpu().numpy(),
    )


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    mode: str,
):

    with torch.no_grad():
        gaussians_dyn = GaussianModel(dataset.sh_degree)
        gaussians_stat = GaussianModel(dataset.sh_degree)
        scene = Scene(
            dataset,
            gaussians_dyn,
            gaussians_stat,
            load_iteration=iteration,
            shuffle=False,
        )
        deform = DeformModel(dataset.is_blender, dataset.is_6dof)
        deform.load_weights(dataset.model_path)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if mode == "segment":
            render_func = segment_dynamic_gaussian
        else:
            render_func = segment_dynamic_gaussian

        if not skip_train:
            render_func(
                dataset.model_path,
                dataset.load2gpu_on_the_fly,
                dataset.is_6dof,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians_dyn,
                gaussians_stat,
                pipeline,
                background,
                deform,
            )

        if not skip_test:
            render_func(
                dataset.model_path,
                dataset.load2gpu_on_the_fly,
                dataset.is_6dof,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians_dyn,
                gaussians_stat,
                pipeline,
                background,
                deform,
            )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--mode",
        default="segment",
        choices=[
            "segment",
        ],
    )
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args.mode,
    )
