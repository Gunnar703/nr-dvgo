import torch.nn.functional as F
import torch.nn as nn
import torch

class MaskCache(nn.Module):
    """
    Module for the searched coarse geometry.
    It supports query for the known free space and unknown space.
    """
    def __init__(self, path: str, mask_cache_thresh: float, ks: int = 3):
        super().__init__()

        state = torch.load(path)
        self.mask_cache_thres = mask_cache_thresh

        # Register boundary parameters as buffers
        self.register_buffer("xyz_min", torch.FloatTensor(state["MaskCache_kwargs"]["xyz_min"]))
        self.register_buffer("xyz_max", torch.FloatTensor(state["MaskCache_kwargs"]["xyz_max"]))

        # Register density grid as buffer
        self.register_buffer(
            "density",
            F.max_pool3d(
                state["model_state_dict"]["density"],
                kernel_size=ks,
                padding = ks // 2,
                stride = 1,
            )
        )

        # Type defs for syntax highlighting
        self.xyz_min: torch.FloatTensor
        self.xyz_max: torch.FloatTensor
        self.density: torch.Tensor

        maskcache_kwargs: dict = state["MaskCache_kwargs"]

        # Activation shift
        self.act_shift = maskcache_kwargs["act_shift"]
        
        # Voxel size ratio
        self.voxel_size_ratio = maskcache_kwargs["voxel_size_ratio"].cpu()

        # Whether to use nearest-neighbor interpolation
        self.nearest = maskcache_kwargs.get("nearest", False)

    @torch.no_grad()
    def forward(self, xyz: torch.Tensor) -> torch.BoolTensor:
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1, 1, 1, -1, 3)
        ind_norm = (
            (xyz.to(self.xyz_min) - self.xyz_min) 
            / 
            (self.xyz_max - self.xyz_min)
        ).flip((-1,)) * 2 - 1

        if self.nearest:
            density = F.grid_sample(self.density, ind_norm, align_corners=True, mode="nearest")
        else:
            density = F.grid_sample(self.density, ind_norm, align_corners=True)
        
        alpha = 1 - torch.exp(-F.softplus(density + self.act_shift) * self.voxel_size_ratio)
        alpha = alpha.reshape(*shape)
        return alpha >= self.mask_cache_thres
