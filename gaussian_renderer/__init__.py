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
import math
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.rigid_utils import from_homogenous, to_homogenous

import pdb as pdb

import torchvision


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)


def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    d_xyz,
    d_rotation,
    d_scaling,
    is_6dof=False,
    scaling_modifier=1.0,
    override_color=None,
    flag_pdb=False,
    flag_segment=False,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    screenspace_points_densify = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
        screenspace_points_densify.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    if is_6dof:
        if torch.is_tensor(d_xyz) is False:
            means3D = pc.get_xyz
        else:
            means3D = from_homogenous(
                torch.bmm(d_xyz, to_homogenous(pc.get_xyz).unsqueeze(-1)).squeeze(-1)
            )
    else:
        means3D = pc.get_xyz + d_xyz
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling + d_scaling
        rotations = pc.get_rotation + d_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if colors_precomp is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
                pc.get_features.shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    if flag_pdb:
        pdb.set_trace()

    if flag_segment:
        d_norm = torch.norm(d_xyz, dim=1)
        # get indices of the gaussians that are have associated d_norm in the top 10% of the values
        indices = torch.argsort(d_norm, descending=True)[: int(0.1 * len(d_norm))]

        # create rendered image with only those gaussians that are in the top 10% of the d_norm values
        rendered_image_moving, _, _ = rasterizer(
            means3D=means3D[indices],  # (N, 3)
            means2D=screenspace_points[indices],  # (N, 3)
            means2D_densify=screenspace_points_densify[indices],  # (N, 3)
            shs=shs[indices],  # (N, 16, 3)
            colors_precomp=colors_precomp,
            opacities=opacity[indices],  # (N, 1)
            scales=scales[indices],  # (N, 3)
            rotations=rotations[indices],  # (N, 4)
            cov3D_precomp=cov3D_precomp,
        )

        # modify the colors of the gaussians with the indices, to make them red
        shs[indices] = (
            torch.tensor([[[1, 0, 0]]]).repeat(len(indices), 1, 1).float().cuda()
        )

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, depth = rasterizer(
        means3D=means3D,  # (N, 3)
        means2D=screenspace_points,  # (N, 3)
        means2D_densify=screenspace_points_densify,  # (N, 3)
        shs=shs,  # (N, 16, 3)
        colors_precomp=colors_precomp,
        opacities=opacity,  # (N, 1)
        scales=scales,  # (N, 3)
        rotations=rotations,  # (N, 4)
        cov3D_precomp=cov3D_precomp,
    )

    if flag_pdb:
        pdb.set_trace()
        # torchvision.utils.save_image(rendered_image, "output_image.png")
        # torchvision.utils.save_image(rendered_image_moving, "output_image_moving.png")

    if flag_segment:
        return {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "viewspace_points_densify": screenspace_points_densify,
            "visibility_filter": radii > 0,
            "radii": radii,
            "depth": depth,
            "render_moving": rendered_image_moving,
        }

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "viewspace_points_densify": screenspace_points_densify,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": depth,
    }
