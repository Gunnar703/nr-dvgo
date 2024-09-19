from .ray_utils import get_rays_of_a_view
from .checkpoint_utils import load_model

import numpy as np
import torch

def compute_bbox_by_camera_frustum(
    HW: np.ndarray, 
    Ks: torch.Tensor, 
    poses: torch.Tensor, 
    near: float,
    far: float,
    inverse_y: bool, 
    flip_x: bool, 
    flip_y: bool,
):

    # Initialize lower bound as inf inf inf
    xyz_min = torch.Tensor([torch.inf, torch.inf, torch.inf])

    # Initialize upper bound as -inf -inf -inf
    xyz_max = -xyz_min

    # Find maximum/minimum points in all camera frustums
    for (h, w), K, c2w in zip(HW, Ks, poses):
        
        # Get ray origins and directions
        rays_o, _, viewdirs = get_rays_of_a_view(
            h, w, K, c2w, inverse_y, flip_x, flip_y
        )

        # Get points at min/max ray distance
        pts_nearfar = torch.stack([
            rays_o + viewdirs * near,
            rays_o + viewdirs * far
        ])

        # Set minimum/maximum bbox coordinates to minimum/maximum of all coords observed so far
        xyz_min = torch.minimum(xyz_min, pts_nearfar.amin((0, 1, 2)))
        xyz_max = torch.maximum(xyz_max, pts_nearfar.amax((0, 1, 2)))

    return xyz_min, xyz_max

@torch.no_grad()
def compute_bbox_by_coarse_geo(model_path: str, threshold: float):

    # Load in model
    model = load_model(model_path)

    # Generate XYZ coords in [0, 1]
    interp = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, model.density.shape[2]),
        torch.linspace(0, 1, model.density.shape[3]),
        torch.linspace(0, 1, model.density.shape[4]),
        indexing="ij"
    ), -1)

    # Map coords from [0, 1] -> [MIN, MAX]
    dense_xyz = (
        model.xyz_min * (1 - interp) 
        + model.xyz_max * interp
    )

    # Query density grid
    density = model.grid_sampler(dense_xyz, model.density)
    alpha = model.activate_density(density)

    # Threshold density : mask out occupied space (1 if occupied else 0)
    mask = alpha > threshold

    # All XYZ coordinates in occupied space
    active_xyz = dense_xyz[mask]

    # Get bbox min/max coords
    xyz_min = active_xyz.amin(0)
    xyz_max = active_xyz.amax(0)

    return xyz_min, xyz_max