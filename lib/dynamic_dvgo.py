import numpy as np
from lib.dvgo import DirectVoxGO, get_ray_marching_ray
from typing import Callable

import torch.nn as nn
import torch


class BeamBender(nn.Module):
    def __init__(self, width, depth, activation, beam_length):
        super().__init__()

        # Phys Params
        self.beam_length = beam_length

        # MLP: (t) -> (v)
        self.mlp = []
        self.mlp += [nn.Linear(4, width), activation()]
        for _ in range(depth - 1):
            self.mlp += [nn.Linear(width, width), activation()]
        self.mlp += [nn.Linear(width, 3)]

        with torch.no_grad():
            for layer in self.mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.uniform_(layer.weight, -0.2, 0.2)
        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, xyzt: torch.Tensor) -> torch.Tensor:
        t = xyzt[..., -1]
        t = t * 10
        xyzt[..., -1] = t

        xyz = xyzt[..., :3]
        uvw = self.mlp(xyzt)
        xyz = xyz - uvw
        return xyz


class DynamicVoxelGrid(DirectVoxGO):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_ray_bender(self, ray_bender: Callable[[torch.Tensor], torch.Tensor]):
        self.ray_bender = ray_bender

    def sample_ray(
        self,
        rays_o,
        rays_d,
        times,
        near,
        far,
        stepsize,
        is_train=False,
        **render_kwargs
    ):
        """Sample query points on rays"""

        xyz_min_tmp = self.xyz_min.clone()
        xyz_max_tmp = self.xyz_max.clone()

        xyz_min_tmp[0] = -0.1
        xyz_min_tmp[1] = -0.5
        xyz_min_tmp[2] = -0.51
        xyz_max_tmp[0] = 0.1
        xyz_max_tmp[1] = 0.5
        xyz_max_tmp[2] = 0.51

        # 1. determine the maximum number of query points to cover all possible rays
        N_samples = (
            int(np.linalg.norm(np.array(self.density.shape[2:]) + 1) / stepsize) + 1
        )
        # 2. determine the two end-points of ray bbox intersection
        vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (xyz_max_tmp - rays_o) / vec
        rate_b = (xyz_min_tmp - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)
        # 4. sample points on each ray
        rng = torch.arange(N_samples, device=rays_o.device)[None].float()
        if is_train:
            rng = rng.repeat(rays_d.shape[-2], 1)
            rng += torch.rand_like(rng[:, [0]])
        step = stepsize * self.voxel_size.to(rays_o.device) * rng
        interpx = t_min[..., None] + step / rays_d.norm(dim=-1, keepdim=True)
        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]

        # 5. update mask for query points outside bbox
        mask_outbbox = ((xyz_min_tmp > rays_pts) | (rays_pts > xyz_max_tmp)).any(dim=-1)

        rays_pts_ = rays_pts.clone()
        times = times[:, None].expand(mask_outbbox.shape)[..., None]

        with torch.no_grad():
            rays_pts_[~mask_outbbox] = self.ray_bender(
                torch.cat((rays_pts_[~mask_outbbox], times[~mask_outbbox]), dim=-1)
            )

        mask_outbbox = ((self.xyz_min > rays_pts_) | (rays_pts_ > self.xyz_max)).any(
            dim=-1
        )
        rays_pts[~mask_outbbox] = self.ray_bender(
            torch.cat((rays_pts[~mask_outbbox], times[~mask_outbbox]), dim=-1)
        )

        return rays_pts, mask_outbbox

    def forward(
        self, rays_o, rays_d, viewdirs, times, global_step=None, **render_kwargs
    ):
        """Volume rendering"""

        ret_dict = {}

        # sample points on rays
        rays_pts, mask_outbbox = self.sample_ray(
            rays_o=rays_o,
            rays_d=rays_d,
            times=times,
            is_train=global_step is not None,
            **render_kwargs
        )
        interval = render_kwargs["stepsize"] * self.voxel_size_ratio

        # query for alpha
        alpha = torch.zeros_like(rays_pts[..., 0])

        # post-activation
        density = self.grid_sampler(rays_pts[~mask_outbbox], self.density)
        alpha[~mask_outbbox] = self.activate_density(density, interval)

        # compute accumulated transmittance
        weights, alphainv_cum = get_ray_marching_ray(alpha)

        # query for color
        mask = weights > self.fast_color_thres
        k0 = torch.zeros(*weights.shape, self.k0_dim).to(weights)
        if not self.rgbnet_full_implicit:
            k0[mask] = self.grid_sampler(rays_pts[mask], self.k0)

        if self.rgbnet is None:
            # no view-depend effect
            rgb = torch.sigmoid(k0)
        else:
            # view-dependent color emission
            if self.rgbnet_direct:
                k0_view = k0
            else:
                k0_view = k0[..., 3:]
                k0_diffuse = k0[..., :3]
            viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
            viewdirs_emb = torch.cat(
                [viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1
            )
            rays_xyz = (rays_pts[mask] - self.xyz_min) / (self.xyz_max - self.xyz_min)
            xyz_emb = (rays_xyz.unsqueeze(-1) * self.posfreq).flatten(-2)
            xyz_emb = torch.cat([rays_xyz, xyz_emb.sin(), xyz_emb.cos()], -1)
            rgb_feat = torch.cat(
                [
                    k0_view[mask],
                    xyz_emb,
                    # TODO: use `rearrange' to make it readable
                    viewdirs_emb.flatten(0, -2)
                    .unsqueeze(-2)
                    .repeat(1, weights.shape[-1], 1)[mask.flatten(0, -2)],
                ],
                -1,
            )
            rgb_logit = torch.zeros(*weights.shape, 3).to(weights)
            rgb_logit[mask] = self.rgbnet(rgb_feat)
            if self.rgbnet_direct:
                rgb = torch.sigmoid(rgb_logit)
            else:
                rgb_logit[mask] = rgb_logit[mask] + k0_diffuse
                rgb = torch.sigmoid(rgb_logit)

        # Ray marching
        rgb_marched = (weights[..., None] * rgb).sum(-2) + alphainv_cum[
            ..., [-1]
        ] * render_kwargs["bg"]
        rgb_marched = rgb_marched.clamp(0, 1)
        depth = (rays_o[..., None, :] - rays_pts).norm(dim=-1)
        depth = (weights * depth).sum(-1) + alphainv_cum[..., -1] * render_kwargs["far"]
        disp = 1 / depth
        ret_dict.update(
            {
                "alphainv_cum": alphainv_cum,
                "weights": weights,
                "rgb_marched": rgb_marched,
                "raw_alpha": alpha,
                "raw_rgb": rgb,
                "depth": depth,
                "disp": disp,
                "mask": mask,
            }
        )
        return ret_dict
