import torch
import numpy as np

from typing import Any, Literal

# from .dvgo import DVGO

def cumprod_exclusive(p):
    # Not sure why: it will be slow at the end of training if clamping at 1e-10 is not applied
    return torch.cat([torch.ones_like(p[..., [0]]), p.clamp_min(1e-10).cumprod(-1)], -1)

def get_ray_marching_ray(alpha):
    alphainv_cum = cumprod_exclusive(1 - alpha)
    weights = alpha * alphainv_cum[..., :-1]
    return weights, alphainv_cum

def get_rays(
    H: int, 
    W: int, 
    K: torch.Tensor, 
    c2w: torch.Tensor, 
    inverse_y: bool, 
    flip_x: bool, 
    flip_y: bool, 
    mode: Literal["center", "lefttop", "random"]
) -> "tuple[torch.Tensor, torch.Tensor]":

    # Generate a grid of pixel coordinates
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W, device=c2w.device),
        torch.linspace(0, H - 1, H, device=c2w.device),
        indexing="ij"
    )
    i = i.t().float()
    j = j.t().float()

    # Adjust based on pixel sampling mode
    if mode == "lefttop":
        pass
    elif mode == "center":
        i, j = i + .5, j + .5
    elif mode == "random":
        i = i + torch.rand_like(i)
        j = j + torch.rand_like(j)
    else:
        raise NotImplementedError
    
    # Optionally flip pixels
    if flip_x: i = i.flip((1,))
    if flip_y: j = j.flip((0,))

    # Calculate ray directions in camera space
    flip_coeff = -1 if inverse_y else 1
    dirs = torch.stack(
        [
            (i - K[0][2]) / K[0][0], (j - K[1][2]) / K[1][1], 
            torch.ones_like(i) * flip_coeff
        ], -1
    )

    # Rotate from camera frame to world frame
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)

    # Translate camera frame's origin to the world frame
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    return rays_o, rays_d

def get_rays_np(H: int, W: int, K: np.ndarray, c2w: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32), 
        np.arange(H, dtype=np.float32), 
        indexing="xy"
    )

    dirs = np.stack(
        [
            (i - K[0][2]) / K[0][0], 
            -(j - K[1][2]) / K[1][1], 
            -np.ones_like(i)
        ], -1
    )

    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, 3], np.shape(rays_d))
    return rays_o, rays_d

def get_rays_of_a_view(
    H: int, 
    W: int, 
    K: torch.Tensor, 
    c2w: torch.Tensor, 
    inverse_y: bool, 
    flip_x: bool, 
    flip_y: bool, 
    mode: Literal["center", "lefttop", "random"] = "center"
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    
    # Sample rays
    rays_o, rays_d = get_rays(
        H, W, K, c2w, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y, mode=mode
    )

    # Get viewing directions
    viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)

    return rays_o, rays_d, viewdirs

def get_training_rays(
    rgb_tr: torch.Tensor,
    train_poses: torch.Tensor,
    HW: np.ndarray,
    Ks: np.ndarray,
    inverse_y: bool,
    flip_x: bool,
    flip_y: bool,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]":
    
    assert len(np.unique(HW, axis=0)) == 1  # make sure height and width are all the same
    assert len(np.unique(Ks.reshape(len(Ks), -1), axis=0)) == 1  # make sure camera intrinsics are all the same
    assert (  # make sure there is a c2w mat, K mat, and HW entry for each view
        len(rgb_tr) == len(train_poses)
        and len(rgb_tr) == len(Ks)
        and len(rgb_tr) == len(HW)
    )

    # Get height, width, and camera intrinsics
    H, W = HW[0]
    K = Ks[0]

    # Allocate space for training rays
    rays_o_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    rays_d_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    viewdirs_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    
    imsz = [1] * len(rgb_tr)  # list of chunk sizes (I think?)

    # Get rays of each view
    for i, c2w in enumerate(train_poses):
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
            H, W, K, c2w, inverse_y, flip_x, flip_y
        )

        # Copy rays of this view into training ray tensors
        rays_o_tr[i].copy_(rays_o.to(rgb_tr.device))
        rays_d_tr[i].copy_(rays_d.to(rgb_tr.device))
        viewdirs_tr[i].copy_(viewdirs.to(rgb_tr.device))
        del rays_o, rays_d, viewdirs

    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz

@torch.no_grad()
def get_training_rays_flatten(
    rgb_tr_original: torch.Tensor,
    train_poses: torch.Tensor,
    HW: np.ndarray,
    Ks: np.ndarray,
    inverse_y: bool,
    flip_x: bool, 
    flip_y: bool,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]":

    # Make sure there are train poses, camera intrinsics, and height/width data for each view
    assert (
        len(rgb_tr_original) == len(train_poses)
        and len(rgb_tr_original) == len(Ks)
        and len(rgb_tr_original) == len(HW)
    )

    # Determine which device to allocate tensors on
    device = rgb_tr_original[0].device

    # Determine total number of pixels across all images
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_original)

    # Pre-allocate placeholder tensors (for the flattened set of training rays)
    rgb_tr = torch.zeros([N, 3], device=device)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)

    # List to store the number of pixels sampled from each image
    # For this function, all entries should be the same: H * W
    imsz = []

    # Initialize pointer: starting index for the next image in the flattened tensors
    top = 0

    # Loop through images and camera poses
    for c2w, img, (h, w), K in zip(train_poses, rgb_tr_original, HW, Ks):
        # Ensure images have the expected shape
        assert img.shape[:2] == (h, w)

        # Get unflattened rays
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
            h, w, K, c2w, inverse_y, flip_x, flip_y 
        )

        # Calculate size of image
        n = h * w

        # Copy data into flattened tensors
        rgb_tr[top : top + n].copy_(img.flatten(0, 1))
        rays_o_tr[top : top + n].copy_(rays_o.flatten(0, 1).to(device))
        rays_d_tr[top : top + n].copy_(rays_d.flatten(0, 1).to(device))
        viewdirs_tr[top : top + n].copy_(viewdirs.flatten(0, 1).to(device))

        # Append number of sampled pixels to imsz
        imsz.append(n)

        # Update pointer
        top += n

    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz

@torch.no_grad()
def get_training_rays_in_maskcache_sampling(
    rgb_tr_original: torch.Tensor,
    train_poses: torch.Tensor,
    HW: np.ndarray,
    Ks: np.ndarray,
    inverse_y: bool,
    flip_x: bool,
    flip_y: bool,
    model: torch.nn.Module,
    render_kwargs: dict[str, Any]
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]":
    
    # Make sure there are train poses, camera intrinsics, and height/width data for each view
    assert (
        len(rgb_tr_original) == len(train_poses)
        and len(rgb_tr_original) == len(Ks)
        and len(rgb_tr_original) == len(HW)
    )

    chunksize = 64  # how many rays to sample at a time

    # Determine which device to allocate tensors on
    device = rgb_tr_original[0].device

    # Determine total number of pixels across all images
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_original)

    # Pre-allocate placeholder tensors (for the flattened set of training rays)
    rgb_tr = torch.zeros([N, 3], device=device)
    rays_o_tr = torch.zeros_like(rgb_tr, device=device)
    rays_d_tr = torch.zeros_like(rgb_tr, device=device)
    viewdirs_tr = torch.zeros_like(rgb_tr, device=device)

    # List to store the number of pixels sampled from each image
    # Sometimes, pixels are skipped because rays shot through them either
    #   a) never intersect the scene's bbox, or
    #   b) never intersect occupied space (as determined by model.mask_cache(...))
    imsz = []

    # Initialize pointer: starting index for the next image in the flattened tensors
    top = 0

    # Loop through images and camera poses
    for c2w, img, (h, w), K in zip(train_poses, rgb_tr_original, HW, Ks):
        # Ensure images have the expected shape
        assert img.shape[:2] == (h, w)

        # Get unflattened rays
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
            h, w, K, c2w, inverse_y, flip_x, flip_y 
        )
        rays_o, rays_d, viewdirs = rays_o.to(device), rays_d.to(device), viewdirs.to(device)

        # Figure out which rays actually intersect the volume / occupied space
        mask = torch.ones(img.shape[:2], device=device, dtype=torch.bool)
        for i in range(0, img.shape[0], chunksize):
            rays_pts, mask_outbbox = model.sample_ray(
                rays_o=rays_o[i : i + chunksize],
                rays_d=rays_d[i : i + chunksize],
                **render_kwargs,
            )
            mask_outbbox = mask_outbbox.to(device)
            rays_pts = rays_pts.to(device)

            # Add points that are in the model's maskcache to the maskout region
            mask_outbbox[~mask_outbbox] |= model.mask_cache(rays_pts[~mask_outbbox])

            # Mask out rays that never intersect occupied space
            mask[i : i + chunksize] &= (~mask_outbbox).any(-1).to(device)

        # Get the total number of pixels that need to be sampled from the current image
        n = mask.sum()

        # Copy data into flattened tensors
        rgb_tr[top : top + n].copy_(img[mask])
        rays_o_tr[top : top + n].copy_(rays_o[mask].to(device))
        rays_d_tr[top : top + n].copy_(rays_d[mask].to(device))
        viewdirs_tr[top : top + n].copy_(viewdirs[mask].to(device))

        # Append number of sampled pixels to imsz
        imsz.append(n)

        # Update pointer
        top += n

    # Trim unneeded space off tensors
    rgb_tr = rgb_tr[:top]
    rays_o_tr = rays_o_tr[:top]
    rays_d_tr = rays_d_tr[:top]
    viewdirs_tr = viewdirs_tr[:top]

    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz