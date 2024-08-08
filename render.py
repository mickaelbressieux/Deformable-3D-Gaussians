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


def render_set(
    model_path,
    load2gpu_on_the_fly,
    is_6dof,
    name,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    deform,
):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    t_list = []

    # render all views

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if load2gpu_on_the_fly:
            view.load2device()
        fid = view.fid
        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        results = render(
            view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof
        )
        rendering = results["render"]
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(
            rendering, os.path.join(render_path, "{0:05d}".format(idx) + ".png")
        )
        torchvision.utils.save_image(
            gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png")
        )
        torchvision.utils.save_image(
            depth, os.path.join(depth_path, "{0:05d}".format(idx) + ".png")
        )

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        fid = view.fid
        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)

        torch.cuda.synchronize()
        t_start = time.time()

        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        results = render(
            view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof
        )

        torch.cuda.synchronize()
        t_end = time.time()
        t_list.append(t_end - t_start)

    t = np.array(t_list[5:])
    fps = 1.0 / t.mean()
    print(f"Test FPS: \033[1;35m{fps:.5f}\033[0m, Num. of GS: {xyz.shape[0]}")


def one_camera_render(
    model_path,
    load2gpu_on_the_fly,
    is_6dof,
    name,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    deform,
):
    # render one view (the first one) and outputs the 3D points in camera space and the displacement of the 3D points in camera space

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    moving_path = os.path.join(model_path, name, "ours_{}".format(iteration), "dynamic")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(moving_path, exist_ok=True)

    t_list = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        if load2gpu_on_the_fly:
            view.load2device()

        fid = view.fid
        xyz = gaussians.get_xyz
        view = views[0]  # replace all the views with the first view
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        # flag_segment = (idx % 100 == 0)
        flag_segment = False

        if idx == 0:
            results = render(
                view,
                gaussians,
                pipeline,
                background,
                0.0,
                0.0,
                0.0,
                is_6dof,
                flag_segment=flag_segment,
                root=args.model_path,
                name_iter=str(iteration),
                name_view=str(idx),
            )
            rendering = results["render"]
            depth = results["depth"]
            depth = depth / (depth.max() + 1e-5)

            torchvision.utils.save_image(
                rendering,
                os.path.join(render_path, "canonical.png"),
            )

        results = render(
            view,
            gaussians,
            pipeline,
            background,
            d_xyz,
            d_rotation,
            d_scaling,
            is_6dof,
            flag_segment=flag_segment,
            root=args.model_path,
            name_iter=str(iteration),
            name_view=str(idx),
        )
        rendering = results["render"]
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

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

        if idx % 100 == 0:

            # project the 3D points from world space to camera space
            means3D = results["means3D"]
            # 3D points in world space
            xyz = gaussians.get_xyz
            Rt = torch.from_numpy(view.R).transpose(0, 1).to(device="cuda").float()
            Tt = (
                torch.from_numpy(view.T)
                .unsqueeze(1)
                .transpose(0, 1)
                .to(device="cuda")
                .float()
            )
            # 3D points in camera space
            xyz_cam = torch.matmul(xyz, Rt) + Tt
            # 3D points displaced in camera space
            means3D_cam = torch.matmul(means3D, Rt) + Tt
            # displacement of the 3D points in camera space
            d_xyz_cam = torch.matmul(d_xyz, Rt)
            # rotation of the 3D points in camera space
            R_quat = matrix_to_quaternion(view.R).to(device="cuda")
            d_rotation_cam = quaternion_multiply_batched(R_quat, d_rotation)
            # save d_xyz_cam, fid and means3D_cam to npy files

            root = args.model_path
            name_idx = str(idx).zfill(5)

            np.save(
                os.path.join(root, "d_xyz_cam_{}.npy".format(name_idx)),
                d_xyz_cam.cpu().numpy(),
            )
            np.save(
                os.path.join(root, "means3D_cam_{}.npy".format(name_idx)),
                means3D_cam.cpu().numpy(),
            )
            np.save(
                os.path.join(root, "fid_cam_{}.npy".format(name_idx)), fid.cpu().numpy()
            )


def new_canonical_space(
    model_path,
    load2gpu_on_the_fly,
    is_6dof,
    name,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    deform,
):
    # render one view (the first one) and outputs the 3D points in camera space and the displacement of the 3D points in camera space

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    moving_path = os.path.join(model_path, name, "ours_{}".format(iteration), "dynamic")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(moving_path, exist_ok=True)

    t_list = []

    new_d_xyz = []
    new_d_rotation = []
    new_d_scaling = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        if load2gpu_on_the_fly:
            view.load2device()

        fid = view.fid
        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)

        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)

        # flag_segment = (idx % 100 == 0)
        flag_segment = False

        results = render(
            view,
            gaussians,
            pipeline,
            background,
            d_xyz,
            d_rotation,
            d_scaling,
            is_6dof,
            flag_segment=flag_segment,
            root=args.model_path,
            name_iter=str(iteration),
            name_view=str(idx),
        )

        if fid == 0:
            # create the canonical space at first time step
            can_means3D = results["means3D"]
            can_xyz = gaussians.get_xyz  # for debugging
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
            # pdb.set_trace()
            new_d_xyz.append(d_xyz.cpu().numpy() - can_d_xyz)
            new_d_rotation.append(d_rotation.cpu().numpy() - can_d_rotation)
            new_d_scaling.append(d_scaling.cpu().numpy() - can_d_scaling)

        rendering = results["render"]
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

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

    new_d_xyz = np.array(new_d_xyz)
    new_d_rotation = np.array(new_d_rotation)
    new_d_scaling = np.array(new_d_scaling)

    pdb.set_trace()

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


def all_camera_render(
    model_path,
    load2gpu_on_the_fly,
    is_6dof,
    name,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    deform,
):
    # render all views and outputs the 3D points in camera space and the displacement of the 3D points in camera space

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    moving_path = os.path.join(model_path, name, "ours_{}".format(iteration), "dynamic")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(moving_path, exist_ok=True)

    t_list = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        if load2gpu_on_the_fly:
            view.load2device()

        fid = view.fid
        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        # flag_segment = (idx % 100 == 0)
        flag_segment = False

        results = render(
            view,
            gaussians,
            pipeline,
            background,
            d_xyz,
            d_rotation,
            d_scaling,
            is_6dof,
            flag_segment=flag_segment,
            root=args.model_path,
            name_iter=str(iteration),
            name_view=str(idx),
        )
        rendering = results["render"]
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

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

        if idx % 100 == 0:

            # project the 3D points from world space to camera space
            means3D = results["means3D"]
            # 3D points in world space
            xyz = gaussians.get_xyz
            Rt = torch.from_numpy(view.R).transpose(0, 1).to(device="cuda").float()
            Tt = (
                torch.from_numpy(view.T)
                .unsqueeze(1)
                .transpose(0, 1)
                .to(device="cuda")
                .float()
            )
            # 3D points in camera space
            xyz_cam = torch.matmul(xyz, Rt) + Tt
            # 3D points displaced in camera space
            means3D_cam = torch.matmul(means3D, Rt) + Tt
            # displacement of the 3D points in camera space
            d_xyz_cam = torch.matmul(d_xyz, Rt)
            # rotation of the 3D points in camera space
            R_quat = matrix_to_quaternion(view.R).to(device="cuda")
            d_rotation_cam = quaternion_multiply_batched(R_quat, d_rotation)
            # save d_xyz_cam, fid and means3D_cam to npy files

            root = args.model_path
            name_idx = str(idx).zfill(5)

            np.save(
                os.path.join(root, "d_xyz_cam_{}.npy".format(name_idx)),
                d_xyz_cam.cpu().numpy(),
            )
            np.save(
                os.path.join(root, "means3D_cam_{}.npy".format(name_idx)),
                means3D_cam.cpu().numpy(),
            )
            np.save(
                os.path.join(root, "fid_cam_{}.npy".format(name_idx)), fid.cpu().numpy()
            )


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
        flag_segment = True

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
                flag_segment=flag_segment,
                can_d_xyz=d_xyz - orig_d_xyz,
                root=args.model_path,
                name_iter=str(iteration),
                name_view=str(idx),
                inliers=inliers,
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
            #
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


def interpolate_time(
    model_path,
    load2gpt_on_the_fly,
    is_6dof,
    name,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    deform,
):
    render_path = os.path.join(
        model_path, name, "interpolate_{}".format(iteration), "renders"
    )
    depth_path = os.path.join(
        model_path, name, "interpolate_{}".format(iteration), "depth"
    )

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    frame = 150
    idx = torch.randint(0, len(views), (1,)).item()
    view = views[idx]
    renderings = []
    for t in tqdm(range(0, frame, 1), desc="Rendering progress"):
        fid = torch.Tensor([t / (frame - 1)]).cuda()
        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        results = render(
            view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof
        )
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        torchvision.utils.save_image(
            rendering, os.path.join(render_path, "{0:05d}".format(t) + ".png")
        )
        torchvision.utils.save_image(
            depth, os.path.join(depth_path, "{0:05d}".format(t) + ".png")
        )

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(
        os.path.join(render_path, "video.mp4"), renderings, fps=30, quality=8
    )


def interpolate_view(
    model_path,
    load2gpt_on_the_fly,
    is_6dof,
    name,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    timer,
):
    render_path = os.path.join(
        model_path, name, "interpolate_view_{}".format(iteration), "renders"
    )
    depth_path = os.path.join(
        model_path, name, "interpolate_view_{}".format(iteration), "depth"
    )
    # acc_path = os.path.join(model_path, name, "interpolate_view_{}".format(iteration), "acc")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    # makedirs(acc_path, exist_ok=True)

    frame = 150
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    idx = torch.randint(0, len(views), (1,)).item()
    view = views[idx]  # Choose a specific time for rendering

    render_poses = torch.stack(render_wander_path(view), 0)
    # render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, frame + 1)[:-1]],
    #                            0)

    renderings = []
    for i, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        fid = view.fid

        matrix = np.linalg.inv(np.array(pose))
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        view.reset_extrinsic(R, T)

        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = timer.step(xyz.detach(), time_input)
        results = render(
            view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof
        )
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)
        # acc = results["acc"]

        torchvision.utils.save_image(
            rendering, os.path.join(render_path, "{0:05d}".format(i) + ".png")
        )
        torchvision.utils.save_image(
            depth, os.path.join(depth_path, "{0:05d}".format(i) + ".png")
        )
        # torchvision.utils.save_image(acc, os.path.join(acc_path, '{0:05d}'.format(i) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(
        os.path.join(render_path, "video.mp4"), renderings, fps=30, quality=8
    )


def interpolate_all(
    model_path,
    load2gpt_on_the_fly,
    is_6dof,
    name,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    deform,
):
    render_path = os.path.join(
        model_path, name, "interpolate_all_{}".format(iteration), "renders"
    )
    depth_path = os.path.join(
        model_path, name, "interpolate_all_{}".format(iteration), "depth"
    )

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    frame = 150
    render_poses = torch.stack(
        [
            pose_spherical(angle, -30.0, 4.0)
            for angle in np.linspace(-180, 180, frame + 1)[:-1]
        ],
        0,
    )
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    idx = torch.randint(0, len(views), (1,)).item()
    view = views[idx]  # Choose a specific time for rendering

    renderings = []
    for i, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        fid = torch.Tensor([i / (frame - 1)]).cuda()

        matrix = np.linalg.inv(np.array(pose))
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        view.reset_extrinsic(R, T)

        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        results = render(
            view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof
        )
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        torchvision.utils.save_image(
            rendering, os.path.join(render_path, "{0:05d}".format(i) + ".png")
        )
        torchvision.utils.save_image(
            depth, os.path.join(depth_path, "{0:05d}".format(i) + ".png")
        )

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(
        os.path.join(render_path, "video.mp4"), renderings, fps=30, quality=8
    )


def interpolate_poses(
    model_path,
    load2gpt_on_the_fly,
    is_6dof,
    name,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    timer,
):
    render_path = os.path.join(
        model_path, name, "interpolate_pose_{}".format(iteration), "renders"
    )
    depth_path = os.path.join(
        model_path, name, "interpolate_pose_{}".format(iteration), "depth"
    )

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    # makedirs(acc_path, exist_ok=True)
    frame = 520
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    idx = torch.randint(0, len(views), (1,)).item()
    view_begin = views[0]  # Choose a specific time for rendering
    view_end = views[-1]
    view = views[idx]

    R_begin = view_begin.R
    R_end = view_end.R
    t_begin = view_begin.T
    t_end = view_end.T

    renderings = []
    for i in tqdm(range(frame), desc="Rendering progress"):
        fid = view.fid

        ratio = i / (frame - 1)

        R_cur = (1 - ratio) * R_begin + ratio * R_end
        T_cur = (1 - ratio) * t_begin + ratio * t_end

        view.reset_extrinsic(R_cur, T_cur)

        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = timer.step(xyz.detach(), time_input)

        results = render(
            view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof
        )
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(
        os.path.join(render_path, "video.mp4"), renderings, fps=60, quality=8
    )


def interpolate_view_original(
    model_path,
    load2gpt_on_the_fly,
    is_6dof,
    name,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    timer,
):
    render_path = os.path.join(
        model_path, name, "interpolate_hyper_view_{}".format(iteration), "renders"
    )
    depth_path = os.path.join(
        model_path, name, "interpolate_hyper_view_{}".format(iteration), "depth"
    )
    # acc_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "acc")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    frame = 1000
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    R = []
    T = []
    for view in views:
        R.append(view.R)
        T.append(view.T)

    view = views[0]
    renderings = []
    for i in tqdm(range(frame), desc="Rendering progress"):
        fid = torch.Tensor([i / (frame - 1)]).cuda()

        query_idx = i / frame * len(views)
        begin_idx = int(np.floor(query_idx))
        end_idx = int(np.ceil(query_idx))
        if end_idx == len(views):
            break
        view_begin = views[begin_idx]
        view_end = views[end_idx]
        R_begin = view_begin.R
        R_end = view_end.R
        t_begin = view_begin.T
        t_end = view_end.T

        ratio = query_idx - begin_idx

        R_cur = (1 - ratio) * R_begin + ratio * R_end
        T_cur = (1 - ratio) * t_begin + ratio * t_end

        view.reset_extrinsic(R_cur, T_cur)

        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = timer.step(xyz.detach(), time_input)

        results = render(
            view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof
        )
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(
        os.path.join(render_path, "video.mp4"), renderings, fps=60, quality=8
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

        if mode == "render":
            render_func = render_set
        elif mode == "time":
            render_func = interpolate_time
        elif mode == "view":
            render_func = interpolate_view
        elif mode == "pose":
            render_func = interpolate_poses
        elif mode == "original":
            render_func = interpolate_view_original
        elif mode == "segment":
            render_func = segment_dynamic_gaussian
        elif mode == "oneCamera":
            render_func = one_camera_render
        elif mode == "new_canonical":
            render_func = new_canonical_space
        else:
            render_func = interpolate_all

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
        default="render",
        choices=[
            "render",
            "time",
            "view",
            "all",
            "pose",
            "original",
            "segment",
            "oneCamera",
            "new_canonical",
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
