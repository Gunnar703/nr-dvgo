import os
import json
from typing import Any
import torch
import imageio
import numpy as np

def trans_t(t: float) -> torch.Tensor:
    """Translate homogeneous camera coordinate in the z-direction"""
    return torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1]
    ], dtype=torch.float32)

def rot_phi(phi: float) -> torch.Tensor:
    """Rotate about the x-axis (phi=azimuth angle)"""
    c = np.cos(phi)
    s = np.sin(phi)

    return torch.tensor([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32)

def rot_theta(theta: float) -> torch.Tensor:
    """Rotate about the y-axis (theta=transverse angle)"""
    c = np.cos(theta)
    s = np.sin(theta)

    return torch.tensor([
        [c, 0, -s, 0],
        [0, 1, 0, 0],
        [s, 0, c, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32)

FLIP_X = torch.tensor([
    [-1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], dtype=torch.float32)

SWAP_YZ = torch.tensor([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
], dtype=torch.float32)

def pose_spherical(theta: float, phi: float, radius: float) -> torch.Tensor:
    """
    phi & theta in degrees
    """
    
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w

    c2w = FLIP_X @ SWAP_YZ @ c2w
    return c2w

def load_blender_data(base_dir: str) -> dict[str, Any]:
    # Read transforms.json
    with open(os.path.join(base_dir, "transforms.json")) as f:
        meta = json.load(f)
    
    # Initialize lists to store images/poses
    all_images = []
    all_poses = []

    # Factor by which to downsample the dataset (NOT the images)
    skip = 1

    # Get images and extrinsic properties
    for frame in meta["frames"][::skip]:
        fname = os.path.join(base_dir, frame["file_path"])
        
        # Read the image
        all_images.append(imageio.imread(fname))

        # Get the camera to world matrix
        c2w = np.array(frame["transform_matrix"])
        c2w[:, 2] *= -1
        all_poses.append(c2w)

    all_images = np.array(all_images, dtype=np.float32) / 255.0
    all_poses = np.array(all_poses, dtype=np.float32)

    # Read/calculate intrinsic properties
    h, w = all_images[0].shape[:2]
    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * w / np.tan(0.5 * camera_angle_x)

    render_poses = torch.stack([
        pose_spherical(angle, -30, 4.0)
        for angle in np.linspace(-180, 180, 40+1)[:-1]
    ], 0)

    return dict(
        images=all_images,
        poses=all_poses,
        render_poses=render_poses,
        height=h,
        width=w,
        focallength=focal,
    )

def load_data(data_base_dir: str, near: float = 0.2, far: float = 2.5) -> dict[str, Any]:
    blender_ret_dict = load_blender_data(data_base_dir)
    
    images: torch.Tensor = blender_ret_dict["images"]

    if images.shape[-1] == 4:
        images = (
            images[..., :3] * images[..., -1:] 
            + (
                1.0 
                - images[..., -1:]
            )
        )

    h = int(blender_ret_dict["height"])
    w = int(blender_ret_dict["width"])
    focal = blender_ret_dict["focallength"]

    hwf = [h, w, focal]
    hw = np.array([im.shape[:2] for im in images])
    irregular_shape = images.dtype is np.dtype("object")

    K = np.array([
        [focal, 0, 0.5*w],
        [0, focal, 0.5*h],
        [0, 0, 1]
    ])
    Ks = K[None].repeat(len(blender_ret_dict["poses"]), axis=0)

    render_poses = blender_ret_dict["render_poses"]

    return dict(
        hwf=hwf,
        HW=hw,
        Ks=Ks,
        near=near,
        far=far,
        poses=torch.tensor(blender_ret_dict["poses"], dtype=torch.float32),
        render_poses=render_poses,
        images=torch.tensor(images, dtype=torch.float32),
        depths=None,
        irregular_shape=irregular_shape,
    )
