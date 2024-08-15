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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, kl_divergence
from gaussian_renderer import render, network_gui, save_npy, render_two_sets
import sys
from scene import Scene, GaussianModel, DeformModel
from utils.general_utils import safe_state, get_linear_noise_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

from random import seed, randint

import pdb

from DSU_utils import (
    create_dynamic_mask,
    identify_rigid_object,
    create_all_d,
    get_first_d,
)

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def training(dataset, opt, pipe, testing_iterations, saving_iterations):
    flag_pdb = False
    tb_writer = prepare_output_and_logger(dataset)
    gaussians_dyn = GaussianModel(dataset.sh_degree)
    gaussians_stat = GaussianModel(dataset.sh_degree)
    deform = DeformModel(dataset.is_blender, dataset.is_6dof)
    deform.train_setting(opt)

    scene = Scene(dataset, gaussians_dyn, gaussians_stat)
    gaussians_dyn.training_setup(opt)
    stat_init = False

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    smooth_term = get_linear_noise_func(
        lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000
    )

    """# remove any file starting with fid, d_xyz or means3D and ending with .npy
    os.system("rm " + args.model_path + "/fid*.npy")
    os.system("rm " + args.model_path + "/d_xyz*.npy")
    os.system("rm " + args.model_path + "/means3D*.npy")"""

    name_iter = None

    for iteration in range(1, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                (
                    custom_cam,
                    do_training,
                    pipe.do_shs_python,
                    pipe.do_cov_python,
                    keep_alive,
                    scaling_modifer,
                ) = network_gui.receive()
                if custom_cam != None:
                    net_image = render(
                        custom_cam, gaussians_dyn, pipe, background, scaling_modifer
                    )["render"]
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and (
                    (iteration < int(opt.iterations)) or not keep_alive
                ):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians_dyn.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        total_frame = len(viewpoint_stack)
        time_interval = 1 / total_frame

        rdm_idx = randint(0, len(viewpoint_stack) - 1)
        viewpoint_cam = viewpoint_stack.pop(rdm_idx)  # Randomly select a camera
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()

        flag_pdb = False

        with torch.no_grad():
            if iteration in args.dynamic_seg_iterations:

                all_d_xyz, all_d_scalings, all_d_rotations = create_all_d(
                    deform, gaussians_dyn.get_xyz, scene
                )

                mask = create_dynamic_mask(
                    all_d_xyz, all_d_scalings, gaussians_stat.get_xyz.shape[0]
                )
                stat_xyz = (gaussians_dyn._xyz + all_d_xyz[0, :, :])[~mask]
                stat_scaling = gaussians_dyn._scaling[~mask]
                stat_rotation = gaussians_dyn._rotation[~mask]
                stat_features_dc = gaussians_dyn._features_dc[~mask]
                stat_features_rest = gaussians_dyn._features_rest[~mask]
                stat_opacity = gaussians_dyn._opacity[~mask]
                stat_max_radii2D = gaussians_dyn.max_radii2D[~mask]

                if not (stat_init):
                    gaussians_stat.create_from_other_gaussian_set(
                        stat_xyz,
                        stat_features_dc,
                        stat_features_rest,
                        stat_opacity,
                        stat_scaling,
                        stat_rotation,
                        stat_max_radii2D,
                    )
                    gaussians_stat.training_setup(opt)
                    stat_init = True
                else:
                    gaussians_stat.densification_postfix(
                        stat_xyz,
                        stat_features_dc,
                        stat_features_rest,
                        stat_opacity,
                        stat_scaling,
                        stat_rotation,
                    )

                gaussians_dyn.prune_points(~mask)

                """rigid_object = identify_rigid_object(
                    gaussians_dyn.get_xyz, all_d_xyz[:, mask], args.model_path
                )"""

                flag_pdb = True

            rigid_object = None
            if iteration in args.rigid_object_iterations:
                all_d_xyz, all_d_scalings, all_d_rotations = create_all_d(
                    deform, gaussians_dyn.get_xyz, scene
                )
                rigid_object = identify_rigid_object(
                    gaussians_dyn.get_xyz, all_d_xyz, args.model_path
                )

        fid = viewpoint_cam.fid  # Frame ID

        if iteration < opt.warm_up:
            d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
        else:
            N = gaussians_dyn.get_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(
                N, -1
            )  # take the frame id and expand it to the number of gaussians.
            ast_noise = (
                0
                if dataset.is_blender
                else torch.randn(1, 1, device="cuda").expand(N, -1)
                * time_interval
                * smooth_term(iteration)
            )
            d_xyz, d_rotation, d_scaling = deform.step(
                gaussians_dyn.get_xyz.detach(), time_input + ast_noise
            )  # we detach the gaussians to avoid backpropagation through them

        if iteration in args.render_intermediate:
            count = 0
            name_iter = iteration  # used for labeling the rendered images

        render_full_moving_stat = False
        if "count" in locals():
            # save the fid, d_xyz and means3D until the gaussians are densified (cannot save different number of gaussians together)
            if count < opt.densification_interval:
                render_full_moving_stat = True
            count += 1

        # Render
        if gaussians_stat.get_xyz.shape[0] > 0:
            render_pkg_re = render_two_sets(
                viewpoint_cam,
                gaussians_dyn,
                gaussians_stat,
                pipe,
                background,
                d_xyz,
                d_rotation,
                d_scaling,
                dataset.is_6dof,
                name_iter=str(name_iter),
                name_view=str(rdm_idx),
                root=args.model_path,
                flag_pdb=flag_pdb,
                render_full_moving_stat=render_full_moving_stat,
                inliers=rigid_object,
            )
        else:
            render_pkg_re = render(
                viewpoint_cam,
                gaussians_dyn,
                pipe,
                background,
                d_xyz,
                d_rotation,
                d_scaling,
                dataset.is_6dof,
                name_iter=str(name_iter),
                name_view=str(rdm_idx),
                root=args.model_path,
                inliers=rigid_object,
            )

        with torch.no_grad():
            if iteration in args.save_npy:
                name_iter_save = iteration
                # if the npy file already exists, delete it
                if os.path.exists(
                    args.model_path + "/fid_" + str(name_iter_save) + ".npy"
                ):
                    os.remove(args.model_path + "/fid_" + str(name_iter_save) + ".npy")
                if os.path.exists(
                    args.model_path + "/means3D_" + str(name_iter_save) + ".npy"
                ):
                    os.remove(
                        args.model_path + "/means3D_" + str(name_iter_save) + ".npy"
                    )
                if os.path.exists(
                    args.model_path + "/d_xyz_" + str(name_iter_save) + ".npy"
                ):
                    os.remove(
                        args.model_path + "/d_xyz_" + str(name_iter_save) + ".npy"
                    )
                counter_save = 0

            if "counter_save" in locals():
                if counter_save < opt.densification_interval:
                    save_npy(
                        fid, "fid_" + str(name_iter_save) + ".npy", root=args.model_path
                    )
                    save_npy(
                        render_pkg_re["means3D"],
                        "means3D_" + str(name_iter_save) + ".npy",
                        root=args.model_path,
                    )
                    save_npy(
                        d_xyz,
                        "d_xyz_" + str(name_iter_save) + ".npy",
                        root=args.model_path,
                    )
                    counter_save += 1

        flag_pdb = False
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg_re["render"],
            render_pkg_re["viewspace_points"],
            render_pkg_re["visibility_filter"],
            render_pkg_re["radii"],
        )
        # depth = render_pkg_re["depth"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        if iteration < opt.warm_up:
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (  # Ldxyz
                1.0 - ssim(image, gt_image)
            )
        else:
            # New loss term to keep d_xyz close to zero using L1 loss

            """d_xyz_0, d_scaling_0, d_rotation_0 = get_first_d(
                deform, gaussians_dyn.get_xyz, scene
            )"""

            Ldxyz = torch.mean(torch.abs(d_xyz))  # L1 loss for d_xyz

            Lscaling = torch.mean(torch.abs(d_scaling))

            Lrotation = torch.mean(torch.abs(d_rotation))

            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
                1.0
                - ssim(image, gt_image)
                + opt.lambda_dxyz * Ldxyz
                + opt.lambda_scaling * Lscaling
                + opt.lambda_rotation * Lrotation
            )

        loss.backward()

        iter_end.record()

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device("cpu")

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Keep track of max radii in image-space for pruning
            if gaussians_stat.get_xyz.shape[0] > 0:
                torch.cat(
                    (gaussians_dyn.max_radii2D, gaussians_stat.max_radii2D), dim=0
                )[visibility_filter] = torch.max(
                    torch.cat(
                        (gaussians_dyn.max_radii2D, gaussians_stat.max_radii2D), dim=0
                    )[visibility_filter],
                    radii[visibility_filter],
                )
            else:
                gaussians_dyn.max_radii2D[visibility_filter] = torch.max(
                    gaussians_dyn.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )

            # Log and save
            cur_psnr = training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                render_two_sets,
                (pipe, background),
                deform,
                dataset.load2gpu_on_the_fly,
                dataset.is_6dof,
            )
            if iteration in testing_iterations:
                if cur_psnr.item() > best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                deform.save_weights(args.model_path, iteration)

            # Densification for gaussians_dyn
            if iteration < opt.densify_until_iter:
                viewspace_point_tensor_densify = render_pkg_re[
                    "viewspace_points_densify"
                ]

                gaussians_dyn.add_densification_stats(
                    torch.norm(
                        viewspace_point_tensor_densify.grad[
                            : gaussians_dyn.get_xyz.shape[0]
                        ][visibility_filter[: gaussians_dyn.get_xyz.shape[0]], :2],
                        dim=-1,
                        keepdim=True,
                    ),
                    visibility_filter[: gaussians_dyn.get_xyz.shape[0]],
                )

                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    size_threshold = (
                        20 if iteration > opt.opacity_reset_interval else None
                    )
                    gaussians_dyn.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                    )
                    print(
                        f"Number of dynamic gaussians: {gaussians_dyn.get_xyz.shape[0]}"
                    )

                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians_dyn.reset_opacity()

            # Densification for gaussians_stat
            if (
                iteration < opt.densify_until_iter
                and gaussians_stat.get_xyz.shape[0] > 0
            ):
                viewspace_point_tensor_densify = render_pkg_re[
                    "viewspace_points_densify"
                ]

                gaussians_stat.add_densification_stats(
                    torch.norm(
                        viewspace_point_tensor_densify.grad[
                            -gaussians_stat.get_xyz.shape[0] :
                        ][visibility_filter[-gaussians_stat.get_xyz.shape[0] :], :2],
                        dim=-1,
                        keepdim=True,
                    ),
                    visibility_filter[-gaussians_stat.get_xyz.shape[0] :],
                )

                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    size_threshold = (
                        20 if iteration > opt.opacity_reset_interval else None
                    )
                    gaussians_stat.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                    )
                    print(
                        f"Number of static gaussians: {gaussians_stat.get_xyz.shape[0]}"
                    )

                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians_stat.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians_dyn.optimizer.step()
                gaussians_dyn.update_learning_rate(iteration)
                deform.optimizer.step()
                gaussians_dyn.optimizer.zero_grad(set_to_none=True)
                deform.optimizer.zero_grad()
                deform.update_learning_rate(iteration)
                if gaussians_stat.get_xyz.shape[0] > 0:
                    gaussians_stat.optimizer.step()
                    gaussians_stat.update_learning_rate(iteration)
                    gaussians_stat.optimizer.zero_grad(set_to_none=True)

    print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderFunc2,
    renderArgs,
    deform,
    load2gpu_on_the_fly,
    is_6dof=False,
):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    test_psnr = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [
                    scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                    for idx in range(5, 30, 5)
                ],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                images = torch.tensor([], device="cuda")
                gts = torch.tensor([], device="cuda")
                for idx, viewpoint in enumerate(config["cameras"]):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                    fid = viewpoint.fid
                    xyz = scene.gaussians_dyn.get_xyz
                    time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                    d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
                    if scene.gaussians_stat.get_xyz.shape[0] > 0:
                        image = torch.clamp(
                            renderFunc2(
                                viewpoint,
                                scene.gaussians_dyn,
                                scene.gaussians_stat,
                                *renderArgs,
                                d_xyz,
                                d_rotation,
                                d_scaling,
                                is_6dof,
                            )["render"],
                            0.0,
                            1.0,
                        )
                    else:
                        image = torch.clamp(
                            renderFunc(
                                viewpoint,
                                scene.gaussians_dyn,
                                *renderArgs,
                                d_xyz,
                                d_rotation,
                                d_scaling,
                                is_6dof,
                            )["render"],
                            0.0,
                            1.0,
                        )
                    gt_image = torch.clamp(
                        viewpoint.original_image.to("cuda"), 0.0, 1.0
                    )
                    images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)

                    if load2gpu_on_the_fly:
                        viewpoint.load2device("cpu")
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            config["name"]
                            + "_view_{}/render".format(viewpoint.image_name),
                            image[None],
                            global_step=iteration,
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config["name"]
                                + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None],
                                global_step=iteration,
                            )

                l1_test = l1_loss(images, gts)
                psnr_test = psnr(images, gts).mean()
                if (
                    config["name"] == "test"
                    or len(validation_configs[0]["cameras"]) == 0
                ):
                    test_psnr = psnr_test
                print(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                        iteration, config["name"], l1_test, psnr_test
                    )
                )
                if tb_writer:
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration
                    )

        if tb_writer:
            tb_writer.add_histogram(
                "scene/opacity_histogram", scene.gaussians_dyn.get_opacity, iteration
            )
            tb_writer.add_scalar(
                "total_points", scene.gaussians_dyn.get_xyz.shape[0], iteration
            )
        torch.cuda.empty_cache()

    return test_psnr


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations",
        nargs="+",
        type=int,
        default=[5000, 6000, 7_000] + list(range(10000, 40001, 1000)),
    )
    parser.add_argument(
        "--save_iterations",
        nargs="+",
        type=int,
        default=[5_000, 7_000, 10_000, 20_000, 30_000, 40000],
    )
    parser.add_argument(
        "--dynamic_seg_iterations",
        nargs="+",
        type=int,
        default=list(range(5_001, 9_001, 1000)),
    )
    parser.add_argument(
        "--rigid_object_iterations",
        nargs="+",
        type=int,
        default=[],
    )
    parser.add_argument("--save_npy", nargs="+", type=int, default=[])

    parser.add_argument("--render_intermediate", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
    )

    # All done
    print("\nTraining complete.")
