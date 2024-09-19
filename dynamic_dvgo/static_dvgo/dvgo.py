import torch.nn.functional as F
import torch.nn as nn
import torch

import numpy as np

from typing import Optional, Literal, Any
from tqdm import tqdm

from .maskcache import MaskCache
from .tv_loss import total_variation
from .ray_utils import get_ray_marching_ray, get_rays_of_a_view

class DVGO(nn.Module):
    def __init__(
        self,
        xyz_min: "tuple[float, float, float]",
        xyz_max: "tuple[float, float, float]",
        num_voxels: int,
        num_voxels_base: int,
        alpha_init: Optional[float],
        nearest: bool = False,
        mask_cache_path: Optional[bool] = None,
        mask_cache_thres: float = 1e-3,
        fast_color_thres: float = 0.0,
        verbose = True,
        **kwargs
    ):
        super().__init__()

        self.verbose = verbose

        # Register boundary parameters as buffers
        self.register_buffer("xyz_min", torch.Tensor(xyz_min))
        self.register_buffer("xyz_max", torch.Tensor(xyz_max))
        
        # Type defs for syntax highlighting
        self.xyz_min: torch.Tensor
        self.xyz_max: torch.Tensor

        # Store parameters related to voxel grid structure
        self.fast_color_thres = fast_color_thres
        self.nearest = nearest

        # Determine base grid resolution
        self.num_voxels_base = num_voxels_base
        
        range = self.xyz_max - self.xyz_min
        volume = range.prod()
        volume_per_vox = volume / self.num_voxels_base
        vox_side_len = volume_per_vox ** (1 / 3)
        self.voxel_size_base = vox_side_len

        # Determine density bias shift
        self.alpha_init = alpha_init
        self.act_shift = np.log(1 / (1 - alpha_init) - 1)
        if self.verbose: print(f"[DVGO.__init__] Set density bias shift to {self.act_shift}")

        # Determine initial grid resolution
        self._set_grid_resolution(num_voxels)

        # Initialize density voxel grid
        self.density = nn.Parameter(torch.zeros([1, 1, *self.world_size]))
        if self.verbose: print(f"[DVGO.__init__] Density voxel grid shape: {self.density.shape}")

        # Initialize color voxel grid
        self.k0_dim = 3
        self.k0 = nn.Parameter(torch.zeros([1, self.k0_dim, *self.world_size]))
        self.rgbnet = None
        if self.verbose: print(f"[DVGO.__init__] Color voxel grid shape: {self.k0.shape}")

        # Use the coarse geometry if provided
        self.mask_cache_path = mask_cache_path
        self.mask_cache_thres = mask_cache_thres
        if mask_cache_path is not None and mask_cache_path:
            self.mask_cache = MaskCache(
                path=mask_cache_path, mask_cache_thresh=mask_cache_thres
            ).to(self.xyz_min.device)
            self._set_nonempty_mask()
        else:
            self.mask_cache = None
            self.nonempty_mask = None

    def _set_grid_resolution(self, num_voxels):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1 / 3)  # side length of each cubic voxel
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()  # number of voxels in the domain
        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base  # ratio of new voxel size to base voxel size

        if self.verbose:
            print(f"[DVGO._set_grid_resolution] Grid resolution updated")
            print(f"> Voxel Size:       {self.voxel_size}")
            print(f"> World Size:       {self.world_size}")
            print(f"> Voxel Size Base:  {self.voxel_size_base}")
            print(f"> Voxel Size Ratio: {self.voxel_size_ratio}")
    
    @torch.no_grad()
    def _set_nonempty_mask(self):
        # Find grid points that are inside nonempty (occupied) space
        self_grid_xyz = torch.stack(
            torch.meshgrid(
                torch.linspace(self.xyz_min[0], self.xyz_max[0], self.density.shape[2]),
                torch.linspace(self.xyz_min[1], self.xyz_max[1], self.density.shape[3]),
                torch.linspace(self.xyz_min[2], self.xyz_max[2], self.density.shape[4]),
                indexing="ij"
            ),
            -1
        )

        # Get a boolean mask: True if space is occupied, False otherwise
        nonempty_mask: torch.BoolTensor = self.mask_cache(self_grid_xyz)[None, None].contiguous()

        # Create and set nonempty mask attribute if it doesn't already exist
        # Otherwise, just set it
        if hasattr(self, "nonempty_mask"):
            self.nonempty_mask = nonempty_mask
        else:
            self.register_buffer("nonempty_mask", nonempty_mask)
        
        # Type def for syntax highlighting
        self.nonempty_mask: torch.Tensor

        # Set density in free space to -100
        self.density[~self.nonempty_mask] = -100

    def get_kwargs(self) -> dict[str, Any]:
        return dict(
            xyz_min = self.xyz_min.cpu().numpy(),
            xyz_max = self.xyz_max.cpu().numpy(),
            num_voxels = self.num_voxels,
            num_voxels_base = self.num_voxels_base,
            alpha_init = self.alpha_init,
            nearest = self.nearest,
            mask_cache_path = self.mask_cache_path,
            mask_cache_thres = self.mask_cache_thres,
        )
    
    def get_MaskCache_kwargs(self) -> dict[str, Any]:
        return dict(
            xyz_min = self.xyz_min.cpu().numpy(),
            xyz_max = self.xyz_max.cpu().numpy(),
            act_shift = self.act_shift,
            voxel_size_ratio = self.voxel_size_ratio,
            nearest = self.nearest,
        )
    
    @torch.no_grad()
    def maskout_near_cam_vox(self, cam_o: torch.Tensor, near: float):
        # Get grid coords
        self_grid_xyz = torch.stack(
            torch.meshgrid(
                torch.linspace(self.xyz_min[0], self.xyz_max[0], self.density.shape[2]),
                torch.linspace(self.xyz_min[1], self.xyz_max[1], self.density.shape[3]),
                torch.linspace(self.xyz_min[2], self.xyz_max[2], self.density.shape[4]),
                indexing="ij"
            ),
            -1,
        )

        # Compute shortest distance between each voxel and a camera
        chunks = cam_o.split(100)
        shortest_dist = []
        for chunk in chunks:
            displacement_vector: torch.Tensor = self_grid_xyz.unsqueeze(-2) - chunk  # shape [num_voxels, 1, 3]
            distance = (displacement_vector ** 2).sum(-1).sqrt()
            min_distance = distance.amin(-1)
            shortest_dist.append(min_distance)

        shortest_dist = torch.stack(shortest_dist)
        shortest_dist = shortest_dist.amin(0)
        
        # Mask out points too close to camera
        self.density[shortest_dist[None, None] <= near] = -100

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels: int):
        original_world_size = self.world_size
        self._set_grid_resolution(num_voxels)

        if self.verbose: print(f"[DVGO.scale_volume_grid] Scaling voxel grid from {original_world_size} to {self.world_size}")

        self.density = torch.nn.Parameter(
            F.interpolate(
                self.density.data,
                size=tuple(self.world_size),
                mode="trilinear",
                align_corners=True,
            )
        )

        self.k0 = torch.nn.Parameter(
            F.interpolate(
                self.k0.data,
                size=tuple(self.world_size),
                mode="trilinear",
                align_corners=True,
            )
        )

        if self.mask_cache is not None:
            self._set_nonempty_mask()

        if self.verbose: print(f"[DVGO.scale_volume_grid] Done scaling voxel grid from {original_world_size} to {self.world_size}")
    
    def voxel_count_views(
            self,
            rays_o_tr: torch.Tensor,  # [N_rays, 3]
            rays_d_tr: torch.Tensor,  # [N_rays, 3]
            imsz: int,
            near: float,
            far: float,
            stepsize: float,
            downrate: int = 1,
            irregular_shape: bool = False,
        ):

        if self.verbose: print("[DVGO.voxel_count_views] Counting how many time each voxel is viewed")

        # Maximum number of samples for a ray
        N_samples = (
            int(
                np.linalg.norm(
                    np.array(self.density.shape[2:]) + 1
                ) # length of the voxel grid's diagonal in indices (NOT world units)
                / stepsize # number of voxels / sample
            ) + 1
        )

        # Vector of sampling distances in indices
        rng = torch.arange(N_samples)[None].float()
        device = rng.device

        # How many times each voxel is intersected by rays
        count = torch.zeros_like(self.density.detach()) 

        for rays_o_, rays_d_ in zip(rays_o_tr.split(imsz), rays_d_tr.split(imsz)):
            ones = torch.ones_like(self.density).requires_grad_(True)
            
            # Type hints for syntax highlighting
            rays_o_: torch.Tensor
            rays_d_: torch.Tensor

            if irregular_shape:
                # Split into chunks if shape is irregular
                rays_o_ = rays_o_.split(10_000)
                rays_d_ = rays_d_.split(10_000)
            else:
                # Otherwise just downsample
                rays_o_ = (
                    rays_o_[::downrate, ::downrate]
                    .to(device)
                    .flatten(0, -2)
                    .split(10_000)
                )
                rays_d_ = (
                    rays_d_[::downrate, ::downrate]
                    .to(device)
                    .flatten(0, -2)
                    .split(10_000)
                )
            
            for rays_o, rays_d in zip(rays_o_, rays_d_):
                # Ray direction with zeros replaced by 1 / 1e6 to avoid div by zero
                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
                
                # Distance between max / min bounds and each ray's origin, divided by the ray's direction
                # Represents how far the ray has to travel along each axis before hitting the voxel grid bounds
                rate_a = (self.xyz_max.to(device) - rays_o) / vec
                rate_b = (self.xyz_min.to(device) - rays_o) / vec

                # Earliest distance when the ray enters the voxel grid along any axis
                # Clamped between near and far plane
                t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(near, far)

                step = stepsize * self.voxel_size * rng  # step size along each ray in world units
                interpx = t_min[..., None] + step / rays_d.norm(dim=-1, keepdim=True)

                # 3D coordinates of the points along the ray that intersect with the voxel grid
                rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
                
                # Sample the grid at rays_pts, then sum, then backward
                # The gradients will be used to figure out how many rays have been hit
                self.grid_sampler(rays_pts.to(self.xyz_max), ones.to(self.xyz_max)).sum().backward()

            with torch.no_grad():
                count += ones.grad > 1
            
        if self.verbose: print("[DVGO.voxel_count_views] Done counting how many time each voxel is viewed")
        return count
    
    def grid_sampler(self, xyz: torch.Tensor, *grids: torch.Tensor, mode: Optional[Literal["nearest", "bilinear"]] = None, align_corners: bool = True):
        if mode is None:
            mode = "nearest" if self.nearest else "bilinear"

        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1, 1, 1, -1, 3)

        ind_norm = (
            (xyz - self.xyz_min) 
            / (self.xyz_max - self.xyz_min)
        ).flip((-1,)) * 2 - 1

        ret_lst = [
            F.grid_sample(grid, ind_norm, mode, align_corners=align_corners)
            .reshape(grid.shape[1], -1)
            .T
            .reshape(*shape, grid.shape[1])
            .squeeze()
            for grid in grids
        ]

        if len(ret_lst) == 1:
            return ret_lst[0]
        
        return ret_lst
    
    def density_total_variation(self) -> torch.Tensor:
        return total_variation(
            self.activate_density(self.density, 1),
            self.nonempty_mask
        )

    def activate_density(self, density: torch.Tensor, interval: Optional[float] = None) -> torch.Tensor:
        interval = interval if interval is not None else self.voxel_size_ratio
        return 1 - torch.exp(-F.softplus(density + self.act_shift) * interval)
    
    def k0_total_variation(self) -> torch.Tensor:
        v = torch.sigmoid(self.k0)
        return total_variation(v, self.nonempty_mask)
    
    def sample_ray(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        near: float,
        far: float,
        stepsize: float,
        is_train: bool = False,
        **render_kwargs
    ) -> "tuple[torch.Tensor, torch.Tensor]":

        # Determine the upper bound for the number of query points necessary to sample a ray through the whole volume
        # Better explanation in voxel_count_views
        N_samples = (
            int(np.linalg.norm(np.array(self.density.shape[2:]) + 1) / stepsize) + 1
        )

        # Determine the end points of the ray/bbox intersection (rate_a and rate_b)
        vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.xyz_max.to(rays_o) - rays_o) / vec
        rate_b = (self.xyz_min.to(rays_o) - rays_o) / vec

        # t_min: Distance along each ray at which the ray enters the bbox
        # t_max: Distance along each ray at which the ray exits the bbox
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)
        t_max = torch.maximum(rate_a, rate_b).amin(-1).clamp(min=near, max=far)

        # Check if each ray intersects the bbox at all
        # Finds when entire rays pass outside the bbox
        mask_outbbox = t_max < t_min

        # Sample points along each ray
        rng = torch.arange(N_samples, device=rays_o.device)[None].float()

        if is_train:
            # If training, sample stochastically
            rng = rng.repeat(rays_d.shape[-2], 1)
            rng += torch.rand_like(rng[:, [0]])
        
        # Better explanations of these in voxel_count_views
        step = stepsize * self.voxel_size.to(rays_o.device) * rng
        interpx = t_min[..., None] + step / rays_d.norm(dim=-1, keepdim=True)
        rays_pts: torch.Tensor = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]

        # Update mask_outbbox for query points outside the bbox
        # Finds the portion of a given ray that lies in the bbox
        mask_outbbox = (
            mask_outbbox[..., None] 
            | (
                (self.xyz_min.to(rays_pts) > rays_pts)
                | (self.xyz_max.to(rays_pts) < rays_pts)
            ).any(dim=-1)
        )

        return rays_pts, mask_outbbox
    
    def forward(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        global_step: Optional[int] = None,
        **render_kwargs,
    ):
        # Sample points along rays
        rays_pts, mask_outbbox = self.sample_ray(
            rays_o,
            rays_d,
            is_train=global_step is not None,
            **render_kwargs,
        )
        interval = render_kwargs["stepsize"] * self.voxel_size_ratio

        # Update mask for query poitns in known free space
        if self.mask_cache is not None:
            points_in_bbox = rays_pts[~mask_outbbox]
            points_in_bbox_but_also_in_free_space = ~self.mask_cache(points_in_bbox)
            mask_outbbox[~mask_outbbox] |= points_in_bbox_but_also_in_free_space

        # Get alpha values
        alpha = torch.zeros_like(rays_pts[..., 0])
        density = self.grid_sampler(rays_pts[~mask_outbbox], self.density)
        alpha[~mask_outbbox] = self.activate_density(density, interval)

        # Get accumulated transmittance at each point along each ray
        weights, alphainv_cum = get_ray_marching_ray(alpha)

        # Get color at each point along each ray
        mask = weights > self.fast_color_thres  # don't worry about points with low densities
        k0 = torch.zeros(*weights.shape, self.k0_dim).to(weights)  # allocate space for colors
        k0[mask] = self.grid_sampler(rays_pts[mask], self.k0)  # sample the color grid
        rgb = torch.sigmoid(k0)

        # Ray marching: Color
        rgb_marched: torch.Tensor = (
            (weights[..., None] * rgb).sum(-2)
            + alphainv_cum[..., [-1]] * render_kwargs["bg"]
        )
        rgb_marched = rgb_marched.clamp(0, 1)

        # Ray marching: Depth
        depth = (rays_o[..., None, :] - rays_pts).norm(-1)  # distance of each point from the camera origin
        depth = (
            (weights * depth).sum(-1)
            + alphainv_cum[..., -1] * render_kwargs["far"]
        )

        # Ray marching: disp (?)
        disp = 1 / depth

        return dict(
            alphainv_cum=alphainv_cum,
            weights=weights,
            rgb_marched=rgb_marched,
            raw_alpha=alpha,
            raw_rgb=rgb,
            depth=depth,
            disp=disp,
            mask=mask,
        )

    @torch.no_grad()
    def render_viewpoints(
        self,
        render_poses: torch.Tensor,
        HW: np.ndarray,
        Ks: np.ndarray,
        render_kwargs: dict[str, Any],
        flip_x: bool = False,
        flip_y: bool = False,
    ) -> "tuple[np.ndarray, np.ndarray]":
        # Initialize lists to hold image results
        rgbs = []
        disps = []

        # Get render results for each view
        for i, c2w in enumerate(tqdm(render_poses)):
            h, w = HW[i]
            K = Ks[i]

            # Get rays for this view
            rays_o, rays_d, viewdirs = get_rays_of_a_view(
                h, w, K, c2w, render_kwargs["inverse_y"], flip_x, flip_y
            )

            # Query the voxel grid for all rays in this view
            # Split the queries into chunks for more efficient computation
            chunksize = 16
            render_result_chunks: "list[dict[str, torch.Tensor]]" = []
            for ro, rd, vd in zip(rays_o.split(chunksize), rays_d.split(chunksize), viewdirs.split(chunksize)):
                res = self.forward(ro, rd, vd, **render_kwargs)
                res = {k: v for k, v in res.items() if k in ("rgb_marched", "disp")}
                render_result_chunks.append(res)

            # Concatenate the chunks back into images
            render_result = {
                k: torch.cat([ret[k] for ret in render_result_chunks])
                for k in render_result_chunks[0].keys()
            }

            # Extract rgb and disp tensors, then convert to numpy
            rgb = render_result["rgb_marched"].cpu().numpy()
            disp = render_result["disp"].cpu().numpy()

            # Add rgb and disp tensors to result lists
            rgbs.append(rgb)
            disps.append(disp)

        # Concatenate rgb and disp lists into np arrays
        rgbs = np.array(rgbs)
        disps = np.array(disps)

        return rgbs, disps