import json
import os
import time
import random

import imageio
import numpy as np
import open3d as o3d

import torch
import torch.nn.functional as F

from dynamic_dvgo.static_dvgo import dvgo, optimization, ray_utils, checkpoint_utils, bbox_utils
from dataset.load_blender import load_data
from typing import Any, Literal, Optional
from tqdm import tqdm, trange
from copy import deepcopy

# Seed all RNGs
torch.manual_seed(123)
np.random.seed(123)
random.seed(123)

def scene_rep_reconstruction(
    phase: str,
    xyz_min: torch.Tensor,
    xyz_max: torch.Tensor,
    data_dict: dict[str, Any],
    basedir: str,
    expname: str,
    num_voxels: int,
    mask_cache_path: Optional[str] = None,
    batch_size: int = 8192,
    N_iters: int = 20_000,
    pervoxel_lr_downrate: int = 1,
    tv_from: int = 0,
    tv_every: int = 1,
    mask_cache_thres: float = 1e-3,
    alpha_init: float = 1e-6,
    world_bound_scale: float = 1.0,
    fast_color_thres: float = 0.0,
    no_reload: bool = False,
    scale_epochs: "list[int]" = [],
    lrate_decay: float = 20, 
    lrate_density: float = 1e-1,  # lr decay by 0.1 after every lrate_decay*1000 steps
    lrate_k0: float = 1e-1,
    stepsize: float = 0.5,
    weight_main: float = 1.0,
    weight_entropy_last: float = 0.01,
    weight_rgbper: float = 0.1,
    weight_tv_density: float = 0.0,
    weight_tv_k0: float = 0.0,
    weight_l1_loss: float = 0.0,
    no_reload_optimizer: bool = False,
    white_bkgd: bool = True,
    inverse_y: bool = False,
    flip_x: bool = False,
    flip_y: bool = False,
    ray_sampling: Literal["in_maskcache", "flatten", "random"] = "random",
    load2gpu_on_the_fly: bool = False,
    print_every: int = 100,
    save_every: int = 1000,
    verbose: bool = True,
):
    """Function to train a single voxel grid"""
    # Get device to make tensors on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Scale the world bbox if necessary
    elongation = (xyz_max - xyz_min) * (world_bound_scale - 1) / 2
    xyz_min -= elongation
    xyz_max += elongation

    HW = data_dict["HW"]
    Ks = data_dict["Ks"]
    near = data_dict["near"]
    far = data_dict["far"]
    poses = data_dict["poses"]
    images = data_dict["images"]

    # See if a checkpoint exists
    last_ckpt_path = os.path.join(basedir, expname, f"last_{phase}.tar")
    reload_ckpt_path = last_ckpt_path if (os.path.isfile(last_ckpt_path) and not no_reload) else None

    # If the grid will be progressively resized AND no ckpt path is
    # specified, compute the number of voxels to start with
    if len(scale_epochs) > 0 and reload_ckpt_path is None:
        num_voxels = int(
            num_voxels 
            / (2 ** len(scale_epochs))
        )
    
    # Create model
    model = dvgo.DVGO(
        xyz_min, 
        xyz_max, 
        num_voxels, 
        num_voxels, 
        alpha_init,
        False,
        mask_cache_path,
        mask_cache_thres,
        fast_color_thres,
        verbose
    )

    # Mask out voxels in front of the camera's near plane
    model.maskout_near_cam_vox(poses[..., :3, 3], near)
    model = model.to(device)

    # Create the optimizer
    optimizer = optimization.create_optimizer_or_freeze_model(
        model, 
        0, 
        lrate_decay, 
        lrate_density, 
        lrate_k0
    )

    # If there is a checkpoint file, load it
    if reload_ckpt_path is None:
        print("[scene_rep_reconstruction] Training from scratch.")
        start = 0
    else:
        print(f"[scene_rep_reconstruction] Loading model from {reload_ckpt_path}")
        model, optimizer, start = checkpoint_utils.load_checkpoint(
            model,
            optimizer,
            reload_ckpt_path,
            no_reload_optimizer
        )

    # Initialize rendering kwargs
    render_kwargs = dict(
        near=near,
        far=far,
        bg=int(white_bkgd),
        stepsize=stepsize,
        inverse_y=inverse_y,
        flip_x=flip_x,
        flip_y=flip_y
    )

    # Get training rays: depends on ray sampling strategy
    if data_dict["irregular_shape"]:
        rgb_tr_original = [
            images[i].to("cpu" if load2gpu_on_the_fly else device)
            for i in range(len(images))
        ]
    else:
        rgb_tr_original = images.to("cpu" if load2gpu_on_the_fly else device)

    if ray_sampling == "in_maskcache":
        (
            rgb_tr,
            rays_o_tr,
            rays_d_tr,
            viewdirs_tr,
            imsz
        ) = ray_utils.get_training_rays_in_maskcache_sampling(
            rgb_tr_original,
            poses,
            HW,
            Ks,
            inverse_y,
            flip_x,
            flip_y,
            model,
            render_kwargs
        )
    elif ray_sampling == "flatten":
        (
            rgb_tr,
            rays_o_tr,
            rays_d_tr,
            viewdirs_tr,
            imsz
        ) = ray_utils.get_training_rays_flatten(
            rgb_tr_original,
            poses,
            HW,
            Ks,
            inverse_y,
            flip_x,
            flip_y,
        )
    elif ray_sampling == "random":
        (
            rgb_tr,
            rays_o_tr,
            rays_d_tr,
            viewdirs_tr,
            imsz
        ) = ray_utils.get_training_rays(
            rgb_tr_original,
            poses,
            HW,
            Ks,
            inverse_y,
            flip_x,
            flip_y
        )
    else:
        raise NotImplementedError(f"{ray_sampling=} is not a valid option. Pick from 'in_maskcache', 'flatten', or 'random'.")
    
    # Create a sampler
    index_generator = optimization.batch_indices_generator(len(rgb_tr), batch_size)
    batch_index_sampler = lambda: next(index_generator)

    # Apply per-voxel lr
    voxel_view_counts = model.voxel_count_views(
        rays_o_tr,
        rays_d_tr,
        imsz,
        near,
        far,
        stepsize,
        pervoxel_lr_downrate,
        data_dict["irregular_shape"]
    )
    optimizer.set_pervoxel_lr(voxel_view_counts)

    # If a voxel is seen less than three times, assume it is empty
    with torch.no_grad():
        model.density[voxel_view_counts <= 2] = -100

    # Training loop
    torch.cuda.empty_cache()  # free unused space on cuda
    psnr_list = []
    start_time = time.time()
    global_step = -1

    for global_step in trange(1 + start, 1 + N_iters):

        # Scale voxel grid if on a scaling step
        if global_step in scale_epochs:
            model.scale_volume_grid(model.num_voxels * 2)
            optimizer = optimization.create_optimizer_or_freeze_model(
                model,
                global_step,
                lrate_decay,
                lrate_density,
                lrate_k0
            )

        # Randomly sample rays
        if ray_sampling in {"flatten", "in_maskcache"}:
            # rays are in tensors of the shape [N_IMAGES * H * W, 3]
            sel_i = batch_index_sampler()
            target = rgb_tr[sel_i]
            rays_o = rays_o_tr[sel_i]
            rays_d = rays_d_tr[sel_i]
            viewdirs = viewdirs_tr[sel_i]
        elif ray_sampling == "random":
            # rays are in tensors of the shape [N_IMAGES, H, W, 3]
            sel_b = torch.randint(rgb_tr.shape[0], [batch_size], device="cpu")
            sel_r = torch.randint(rgb_tr.shape[1], [batch_size], device="cpu")
            sel_c = torch.randint(rgb_tr.shape[2], [batch_size], device="cpu")
            target = rgb_tr[sel_b, sel_r, sel_c]
            rays_o = rays_o_tr[sel_b, sel_r, sel_c]
            rays_d = rays_d_tr[sel_b, sel_r, sel_c]
            viewdirs = viewdirs_tr[sel_b, sel_r, sel_c]
        else:
            raise NotImplementedError(f"If you're seeing this you did something horribly wrong. It's okay though. Do you like cheese? I like cheese.")
        
        # If data is set to be moved to the GPU 'on the fly', do so now
        target = target.to(device)
        rays_o = rays_o.to(device)
        rays_d = rays_d.to(device)
        viewdirs = viewdirs.to(device)

        # Render the rays
        render_result = model(
            rays_o,
            rays_d,
            global_step,
            **render_kwargs
        )

        # Perform gradient descent
        optimizer.zero_grad(set_to_none=True)
        loss = weight_main * F.mse_loss(
            render_result["rgb_marched"],
            target
        )

        # Compute PSNR before adding more terms onto loss
        psnr = -10.0 * torch.log10(loss.detach()).item()
        psnr_list.append(psnr)

        if weight_entropy_last > 0:
            # Add background entropy loss
            pout: torch.Tensor = render_result["alphainv_cum"][..., -1].clamp(1e-6, 1 - 1e-6)
            entropy_last_loss = -(
                pout * torch.log(pout)
                + (1 - pout) * torch.log(1 - pout)
            ).mean()

            loss += weight_entropy_last * entropy_last_loss
        
        if weight_rgbper > 0:
            # Add pointwise rgb loss
            rgbper: torch.Tensor = render_result["raw_rgb"] - target.unsqueeze(-2)
            rgbper = rgbper.pow(2).sum(-1)
            rgbper_loss = (
                rgbper * render_result["weights"].detach()
            ).sum(-1).mean()
            
            loss += weight_rgbper * rgbper_loss
        
        # Add total variation loss for density and color
        if (
            weight_tv_density > 0
            and global_step > tv_from
            and global_step % tv_every == 0
        ):
            loss += weight_tv_density * model.density_total_variation()
        if (
            weight_tv_k0 > 0
            and global_step > tv_from
            and global_step & tv_every == 0
        ):
            loss += weight_tv_k0 * model.k0_total_variation()

        # Add L1 loss
        if weight_l1_loss > 0:
            loss += weight_l1_loss * torch.abs(model.density).mean()


        loss.backward()
        optimizer.step()

        # Update lr
        decay_steps = lrate_decay * 1000
        decay_factor = 0.1 ** (1 / decay_steps)

        for _, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = param_group["lr"] * decay_factor

        # Print update
        if global_step % print_every == 0:
            eps_time = time.time() - start_time
            eps_time_str = f"{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}"
            tqdm.write(
                f"[scene_rep_reconstruction] Iter {global_step:6d} / "
                f"Loss: {loss.item():.9f} / PSNR: {np.mean(psnr_list):5.2f} / "
                f"Eps: {eps_time_str}"
            )
            psnr_list = []

        # Save state
        if global_step % save_every == 0:
            path = os.path.join(
                basedir,
                expname,
                f"{global_step:06d}_{phase}.tar"
            )
            torch.save(dict(
                global_step=global_step,
                model_kwargs=model.get_kwargs(),
                MaskCache_kwargs=model.get_MaskCache_kwargs(),
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
            ), path)

            print(f"[scene_rep_reconstruction] Saved checkpoints at {path}")
        
    if global_step != -1:
        torch.save(dict(
            global_step=global_step,
            model_kwargs=model.get_kwargs(),
            MaskCache_kwargs=model.get_MaskCache_kwargs(),
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict()),
            last_ckpt_path,
        )
        print(f"[scene_rep_reconstruction]: Saved checkpoints at {last_ckpt_path}")

def train(
    data_dict: dict[str, Any],
    basedir: str,
    expname: str,
    num_voxels: int,
    batch_size: int = 8192,
    N_iters_coarse: int = 20_000,
    N_iters_fine: int = 20_000,
    pervoxel_lr_downrate: int = 1,
    tv_from: int = 0,
    tv_every: int = 1,
    mask_cache_thres: float = 1e-3,
    alpha_init: float = 1e-6,
    world_bound_scale: float = 1.0,
    fast_color_thres: float = 0.0,
    no_reload: bool = False,
    scale_epochs: "list[int]" = [],
    lrate_decay: float = 20, 
    lrate_density: float = 1e-1,  # lr decay by 0.1 after every lrate_decay*1000 steps
    lrate_k0: float = 1e-1,
    stepsize: float = 0.5,
    weight_main: float = 1.0,
    weight_entropy_last: float = 0.01,
    weight_rgbper: float = 0.1,
    weight_tv_density: float = 0.0,
    weight_tv_k0: float = 0.0,
    weight_l1_loss: float = 0.0,
    no_reload_optimizer: bool = False,
    white_bkgd: bool = True,
    inverse_y: bool = False,
    flip_x: bool = False,
    flip_y: bool = False,
    ray_sampling: Literal["in_maskcache", "flatten", "random"] = "random",
    load2gpu_on_the_fly: bool = False,
    print_every: int = 100,
    save_every: int = 1000,
    verbose: bool = True
):
    args = dict(data_dict=data_dict, basedir=basedir, expname=expname, num_voxels=num_voxels, batch_size=batch_size, N_iters_coarse=N_iters_coarse, N_iters_fine=N_iters_fine, pervoxel_lr_downrate=pervoxel_lr_downrate, tv_from=tv_from, tv_every=tv_every, mask_cache_thres=mask_cache_thres, alpha_init=alpha_init, world_bound_scale=world_bound_scale, fast_color_thres=fast_color_thres, no_reload=no_reload, scale_epochs=scale_epochs, lrate_decay=lrate_decay, lrate_density=lrate_density, lrate_k0=lrate_k0, stepsize=stepsize, weight_main=weight_main, weight_entropy_last=weight_entropy_last, weight_rgbper=weight_rgbper, weight_tv_density=weight_tv_density, weight_tv_k0=weight_tv_k0, weight_l1_loss=weight_l1_loss, no_reload_optimizer=no_reload_optimizer, white_bkgd=white_bkgd, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y, ray_sampling=ray_sampling, load2gpu_on_the_fly=load2gpu_on_the_fly, print_every=print_every, save_every=save_every, verbose=verbose)
    
    os.makedirs(
        os.path.join(basedir, expname),
        exist_ok=True
    )

    with open(os.path.join(basedir, expname, "args.json"), "w") as f:
        dargs = deepcopy(args)
        dargs.pop("data_dict")
        json.dump(dargs, f, indent=4)
        del dargs

    print("[train] Computing scene bbox from camera frustums")
    coarse_args = deepcopy(args)
    coarse_args.pop("N_iters_coarse")
    coarse_args.pop("N_iters_fine")
    coarse_args["scale_epochs"] = []  # don't scale the coarse grid
    coarse_args["N_iters"] = N_iters_coarse
    coarse_args["weight_tv_density"] = 0.0  # don't apply tv loss on coarse grid
    coarse_args["weight_tv_k0"] = 0.0
    coarse_args["num_voxels"] = int(
        num_voxels 
        / (2 ** len(scale_epochs))
    )
    coarse_args["weight_l1_loss"] = 0.0
    xyz_min, xyz_max = bbox_utils.compute_bbox_by_camera_frustum(
        data_dict["HW"],
        data_dict["Ks"],
        data_dict["poses"],
        data_dict["near"],
        data_dict["far"],
        inverse_y,
        flip_x,
        flip_y,
    )

    print("[train] Training coarse grid")
    scene_rep_reconstruction("coarse", xyz_min, xyz_max, **coarse_args)
    print("[train] Training finished")

    print("[train] Computing scene bbox from coarse geometry")
    fine_args = deepcopy(args)
    fine_args.pop("N_iters_fine")
    fine_args.pop("N_iters_coarse")
    fine_args["N_iters"] = N_iters_fine
    fine_args["ray_sampling"] = "in_maskcache"
    fine_args["alpha_init"] = 1e-2
    fine_args["weight_entropy_last"] *= 1e-1
    fine_args["weight_rgbper"] *= 1e-1
    xyz_min_coarse, xyz_max_coarse = bbox_utils.compute_bbox_by_coarse_geo(
        model_path=os.path.join(BASE_DIR, EXPNAME, "last_coarse.tar"),
        threshold=mask_cache_thres
    )

    print("[train] Training fine grid")
    fine_args["mask_cache_path"] = os.path.join(BASE_DIR, EXPNAME, "last_coarse.tar")
    scene_rep_reconstruction("fine", xyz_min_coarse, xyz_max_coarse, **fine_args)
    print("[train] Training finished")

def render_test(phase: str, basedir: str, expname: str, stepsize: float, white_bkgd: bool, inverse_y: bool, flip_x: bool, flip_y: bool, data_dict: dict[str, Any]):
    ckpt_path = os.path.join(basedir, expname, f"last_{phase}.tar")
    ckpt_name = os.path.split(ckpt_path)[1][:-4]
    model = checkpoint_utils.load_model(ckpt_path)
    render_kwargs=dict(
        near=data_dict["near"],
        far=data_dict["far"],
        bg=int(white_bkgd),
        stepsize=stepsize,
        inverse_y=inverse_y
    )

    testsavedir = os.path.join(basedir, expname, f"render_test_{ckpt_name}")
    os.makedirs(testsavedir, exist_ok=True)

    rgbs, disps = model.render_viewpoints(
        data_dict["poses"],
        HW=data_dict["HW"],
        Ks=data_dict["Ks"],
        render_kwargs=render_kwargs,
        flip_x=flip_x,
        flip_y=flip_y
    )

    imageio.mimwrite(
        os.path.join(testsavedir, "video_rgb.mp4"),
        (255 * np.clip(rgbs, 0, 1)).astype(np.uint8),
        fps=30,
        quality=8
    )

    imageio.mimwrite(
        os.path.join(testsavedir, "video_disp.mp4"),
        (255 * np.clip(disps/disps.max(), 0, 1)).astype(np.uint8),
        fps=30,
        quality=8
    )

def viz_cameras_3d(data_dict: dict[str, Any], inverse_y: bool, flip_x: bool, flip_y: bool):
    xyz_min, xyz_max = bbox_utils.compute_bbox_by_camera_frustum(
        data_dict["HW"],
        data_dict["Ks"],
        data_dict["poses"],
        data_dict["near"],
        data_dict["far"],
        inverse_y,
        flip_x,
        flip_y,
    )

    poses, HW, Ks = data_dict["poses"], data_dict["HW"], data_dict["Ks"]
    near, far = data_dict["near"], data_dict["far"]

    cam_list = []
    for c2w, (H, W), K in zip(poses, HW, Ks):
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
            H=H,
            W=W,
            K=K,
            c2w=c2w,
            inverse_y=inverse_y,
            flip_x=flip_x,
            flip_y=flip_y,
        )
    
        cam_o = rays_o[0, 0].cpu().numpy()
        cam_d = rays_d[[0, 0, -1, -1], [0, -1, 0, -1]].cpu().numpy()
        cam_list.append(np.array([cam_o, *(cam_o + cam_d * max(near, far * 0.05))]))

    xyz_min = xyz_min.cpu().numpy()
    xyz_max = xyz_max.cpu().numpy()
    cam_list = np.array(cam_list)

    # Outer aabb
    aabb_01 = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 1],
            [1, 1, 0],
        ]
    )

    out_bbox = o3d.geometry.LineSet()
    out_bbox.points = o3d.utility.Vector3dVector(xyz_min + aabb_01 * (xyz_max - xyz_min))
    out_bbox.colors = o3d.utility.Vector3dVector([[1, 0, 0] for i in range(12)])
    out_bbox.lines = o3d.utility.Vector2iVector(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]
    )

    # Cameras
    cam_frustrm_lst = []
    for cam in cam_list:
        cam_frustrm = o3d.geometry.LineSet()
        cam_frustrm.points = o3d.utility.Vector3dVector(cam)
    
        if len(cam) == 5:
            cam_frustrm.colors = o3d.utility.Vector3dVector([[0, 0, 0] for i in range(8)])
            cam_frustrm.lines = o3d.utility.Vector2iVector(
                [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 4], [4, 3], [3, 1]]
            )
        elif len(cam) == 8:
            cam_frustrm.colors = o3d.utility.Vector3dVector([[0, 0, 0] for i in range(12)])
            cam_frustrm.lines = o3d.utility.Vector2iVector(
                [
                    [0, 1],
                    [1, 3],
                    [3, 2],
                    [2, 0],
                    [4, 5],
                    [5, 7],
                    [7, 6],
                    [6, 4],
                    [0, 4],
                    [1, 5],
                    [3, 7],
                    [2, 6],
                ]
            )
        cam_frustrm_lst.append(cam_frustrm)

    o3d.visualization.draw_geometries([
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=xyz_min),
        out_bbox,
        *cam_frustrm_lst,
    ])
    return xyz_min, xyz_max, cam_list

@torch.no_grad()
def viz_geom_3d(phase: str, basedir: str, expname: str, cam_list: np.ndarray, thres: float = 1e-3):
    ckpt_path = os.path.join(basedir, expname, f"last_{phase}.tar")
    model = checkpoint_utils.load_model(ckpt_path)
    alpha = model.activate_density(model.density).squeeze().cpu().numpy()  # activate the raw density grid
    rgb = torch.sigmoid(model.k0).squeeze().permute(1, 2, 3, 0).cpu().numpy()  # activate the raw color grid

    xyz_min = model.xyz_min.cpu().numpy()
    xyz_max = model.xyz_max.cpu().numpy()

    if rgb.shape[0] < rgb.shape[-1]:
        alpha = np.transpose(alpha, (1, 2, 0))
        rgb = np.transpose(rgb, (1, 2, 3, 0))

    cam_frustum_list = []
    for cam in cam_list:
        cam_frustrm = o3d.geometry.LineSet()
        cam_frustrm.points = o3d.utility.Vector3dVector(cam)
        if len(cam) == 5:
            cam_frustrm.colors = o3d.utility.Vector3dVector(
                [[0.5, 0.5, 0.5] for i in range(8)]
            )
            cam_frustrm.lines = o3d.utility.Vector2iVector(
                [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 4], [4, 3], [3, 1]]
            )
        elif len(cam) == 8:
            cam_frustrm.colors = o3d.utility.Vector3dVector(
                [[0.5, 0.5, 0.5] for i in range(12)]
            )
            cam_frustrm.lines = o3d.utility.Vector2iVector(
                [
                    [0, 1],
                    [1, 3],
                    [3, 2],
                    [2, 0],
                    [4, 5],
                    [5, 7],
                    [7, 6],
                    [6, 4],
                    [0, 4],
                    [1, 5],
                    [3, 7],
                    [2, 6],
                ]
            )
        cam_frustum_list.append(cam_frustrm)
    
    aabb_01 = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 1],
            [1, 1, 0],
        ]
    )
    out_bbox = o3d.geometry.LineSet()
    out_bbox.points = o3d.utility.Vector3dVector(xyz_min + aabb_01 * (xyz_max - xyz_min))
    out_bbox.colors = o3d.utility.Vector3dVector([[1, 0, 0] for i in range(12)])
    out_bbox.lines = o3d.utility.Vector2iVector(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]
    )

    xyz = np.stack((alpha > thres).nonzero(), -1)
    color = rgb[xyz[:, 0], xyz[:, 1], xyz[:, 2]]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        xyz / alpha.shape * (xyz_max - xyz_min) + xyz_min
    )
    pcd.colors = o3d.utility.Vector3dVector(color[:, :3])
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        pcd, voxel_size=max((xyz_max - xyz_min) / alpha.shape)
    )

    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        return False


    o3d.visualization.draw_geometries_with_key_callbacks(
        [
            o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=(xyz_max - xyz_min).min() * 0.1, origin=xyz_min
            ),
            out_bbox,
            voxel_grid,
            *cam_frustum_list,
        ],
        {ord("K"): change_background_to_black},
    )

if __name__ == "__main__":

    DATA_BASE_DIR = "data/beam_equilibrium"
    NEAR_CLIP = 0.2
    FAR_CLIP = 3

    BASE_DIR = "results"
    EXPNAME = "equilibrium"

    STEPSIZE = 0.5
    WHITE_BKGD = True
    INVERSE_Y = False
    FLIP_X = False
    FLIP_Y = False

    data_dict = load_data(
        data_base_dir=DATA_BASE_DIR,
        near=NEAR_CLIP,
        far=FAR_CLIP,
    )

    xyz_min, xyz_max, cam_list = viz_cameras_3d(
        data_dict=data_dict,
        inverse_y=INVERSE_Y,
        flip_x=FLIP_X,
        flip_y=FLIP_Y
    )

    train(
        data_dict=data_dict,
        basedir=BASE_DIR,
        expname=EXPNAME,
        num_voxels=128**3,
        batch_size=8192,
        N_iters_coarse=3_000,
        N_iters_fine=4_500,
        pervoxel_lr_downrate=1,
        tv_from=0,
        tv_every=1,
        mask_cache_thres=1e-3,
        alpha_init=1e-6,
        world_bound_scale=1.0,
        fast_color_thres=0.0,
        no_reload=False,
        scale_epochs=[1_000, 2_000, 3_000],
        lrate_decay=20,  # lr decay by 0.1 after every lrate_decay*1000 steps
        lrate_density=1e-1,
        lrate_k0=1e-1,
        stepsize=STEPSIZE,
        weight_main=1.0,
        weight_entropy_last=0.01,
        weight_rgbper=0.1,
        weight_tv_density=0.1,
        weight_tv_k0=0.0,
        weight_l1_loss=0.0,
        no_reload_optimizer=False,
        white_bkgd=WHITE_BKGD,
        inverse_y=INVERSE_Y,
        flip_x=FLIP_X,
        flip_y=FLIP_Y,
        ray_sampling="random",
        load2gpu_on_the_fly=False,
        print_every=100,
        save_every=1000,
        verbose=True
    )

    render_test(
        phase="fine",
        basedir=BASE_DIR,
        expname=EXPNAME,
        stepsize=STEPSIZE,
        white_bkgd=WHITE_BKGD,
        inverse_y=INVERSE_Y,
        flip_x=FLIP_X,
        flip_y=FLIP_Y,
        data_dict=data_dict
    )

    viz_geom_3d(
        phase="coarse",
        basedir=BASE_DIR,
        expname=EXPNAME,
        cam_list=cam_list,
        thres=1e-3
    )

    viz_geom_3d(
        phase="fine",
        basedir=BASE_DIR,
        expname=EXPNAME,
        cam_list=cam_list,
        thres=1e-3
    )
