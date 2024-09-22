from typing import Callable
from .static_dvgo.dvgo import DVGO
from .deformation import grid2particles, particles2grid

import torch.nn.functional as F
import torch.nn as nn
import torch

class DynamicObserver(nn.Module):
    def __init__(self, base_grid: DVGO, scale_factor_xyz: "tuple[int, int, int]", n_particles: int):
        super().__init__()

        # Number of particles used to deform the grid via the Eulerian-to-Lagrangian-to-Eulerian method (see PACNeRF paper)
        self.n_particles = n_particles

        # Hacky way to keep base_grid from being populated in self.parameters()
        self.base_grid = [base_grid,]
        
        # Freeze all gradients of the base_grid
        for p in self.base_grid[0].parameters():
            p.requires_grad = False

        # Resize grids
        mc_xyz_min, mc_xyz_max, mask_cache_density = self._scale_grid(
            scale_factor_xyz=scale_factor_xyz, 
            original_grid=self.base_grid[0].mask_cache.density, 
            xyz_min=self.base_grid[0].mask_cache.xyz_min, 
            xyz_max=self.base_grid[0].mask_cache.xyz_max
        )
        xyz_min, xyz_max, density = self._scale_grid(
            scale_factor_xyz=scale_factor_xyz, 
            original_grid=self.base_grid[0].density, 
            xyz_min=self.base_grid[0].xyz_min, 
            xyz_max=self.base_grid[0].xyz_max
        )
        _, _, k0 = self._scale_grid(
            scale_factor_xyz=scale_factor_xyz, 
            original_grid=self.base_grid[0].k0, 
            xyz_min=self.base_grid[0].xyz_min, 
            xyz_max=self.base_grid[0].xyz_max
        )

        self.base_grid[0].mask_cache.density = nn.Parameter(mask_cache_density, requires_grad=False)
        self.base_grid[0].density = nn.Parameter(density, requires_grad=False)
        self.base_grid[0].k0 = nn.Parameter(k0, requires_grad=False)

        # Update xyz_min / xyz_max to preserve interpolation behavior
        # - mask_cache.xyz_min
        # - mask_cache.xyz_max
        # - xyz_min
        # - xyz_max

        self.base_grid[0].mask_cache.xyz_min = mc_xyz_min
        self.base_grid[0].mask_cache.xyz_max = mc_xyz_max
        self.base_grid[0].xyz_min = xyz_min
        self.base_grid[0].xyz_max = xyz_max

        # Make a deep copy of each new grid as an attribute of this DynamicObserver object
        # This will serve as the base_grid
        self.mask_cache_density = torch.clone(self.base_grid[0].mask_cache.density.detach())
        self.density = torch.clone(self.base_grid[0].density.detach())
        self.k0 = torch.clone(self.base_grid[0].k0.detach())

        # Seed the new grid with particles. We want to do this here in order
        # to keep a consistent set of particles across all calls to deform_grid
        self.mc_particle_coords = place_particles(
            grid_shape=self.base_grid[0].mask_cache.density.shape[-3:],
            xyz_min=self.base_grid[0].mask_cache.xyz_min,
            xyz_max=self.base_grid[0].mask_cache.xyz_max,
            particles_per_grid_cell=self.n_particles
        ).contiguous()

        self.particle_coords = place_particles(
            grid_shape=self.base_grid[0].density.shape[-3:],
            xyz_min=self.base_grid[0].xyz_min,
            xyz_max=self.base_grid[0].xyz_max,
            particles_per_grid_cell=self.n_particles
        ).contiguous()

        # Also compute the values associated with the particles (in each base grid) here.
        # This should help to avoid duplicate computation.
        self.particle_values_mc_density = grid2particles(
            grid_values=self.mask_cache_density,
            particle_coords=self.mc_particle_coords, 
            xyz_min=self.base_grid[0].mask_cache.xyz_min, 
            xyz_max=self.base_grid[0].mask_cache.xyz_max
        )
        self.particle_values_density = grid2particles(
            grid_values=self.density,
            particle_coords=self.particle_coords,
            xyz_min=self.base_grid[0].xyz_min,
            xyz_max=self.base_grid[0].xyz_max
        )
        self.particle_values_k0 = grid2particles(
            grid_values=self.k0,
            particle_coords=self.particle_coords,
            xyz_min=self.base_grid[0].xyz_min,
            xyz_max=self.base_grid[0].xyz_max
        )
    
    def _scale_grid(self, scale_factor_xyz: "tuple[float, float, float]", original_grid: torch.Tensor, xyz_min: torch.Tensor, xyz_max: torch.Tensor):
        """Scales an input grid by expanding its bbox.

        Args:
            scale_factor_xyz (tuple[float, float, float]): Scale factors for the scene bounding box in x, y, and z.
            original_grid (torch.Tensor): Grid to scale.
            xyz_min (torch.Tensor): Tensor of shape [3,] representing the minimum coordinates of the bbox.
            xyz_max (torch.Tensor): Tensor of shape [3,] representing the maximum coordinates of the bbox.

        Returns:
            new_xyz_min (torch.Tensor): The new minimum coordinates of the bbox.
            new_xyz_max (torch.Tensor): The new maximum coordinates of the bbox.
            new_grid (torch.Tensor): The new grid.
        """

        # Get shape of the original grid
        _, num_channels, nvox_x, nvox_y, nvox_z = original_grid.shape
        
        # Compute size of the new grid
        nvox_x_new = int(nvox_x * scale_factor_xyz[0])
        nvox_y_new = int(nvox_y * scale_factor_xyz[1])
        nvox_z_new = int(nvox_z * scale_factor_xyz[2])

        pad_x = (nvox_x_new - nvox_x) // 2
        pad_y = (nvox_y_new - nvox_y) // 2
        pad_z = (nvox_z_new - nvox_z) // 2

        # Initialize the new grid
        new_grid = torch.zeros(
            1, num_channels,
            nvox_x + 2 * pad_x,
            nvox_y + 2 * pad_y,
            nvox_z + 2 * pad_z,
        ).to(original_grid) - 1000 * (num_channels == 1)

        # Place the old grid within the new grid
        new_grid[0, :,
            pad_x : nvox_x + pad_x,
            pad_y : nvox_y + pad_y,
            pad_z : nvox_z + pad_z,
        ] = original_grid[0, :]

        # Compute the new xyz_min and xyz_max
        voxel_len_x = (xyz_max[0] - xyz_min[0]) / nvox_x
        voxel_len_y = (xyz_max[1] - xyz_min[1]) / nvox_y
        voxel_len_z = (xyz_max[2] - xyz_min[2]) / nvox_z

        xyz_shift = torch.tensor([
            pad_x * voxel_len_x,
            pad_y * voxel_len_y,
            pad_z * voxel_len_z,
        ]).to(xyz_min)

        new_xyz_min = xyz_min - xyz_shift
        new_xyz_max = xyz_max + xyz_shift

        return new_xyz_min, new_xyz_max, new_grid

    def deform_grid(self, deformation_function: Callable[[torch.Tensor], torch.Tensor]):
        # deformation function: (x, y, z) -> (x', y', z')

        # This function (deform_grid) should:
        # 1) Compute the new positions of each particle using deformation_function
        # 2) Use the particles2grid function to create deformed grids
        # 3) Assign the deformed grids back to the appropriate attributes of self.base_grid[0]
        # 4) Manually delete unneeded tensors to ensure they are erased from memory

        # 1) Compute the new positions of each particle using deformation_function
        deformed_mc_particle_coords = deformation_function(self.mc_particle_coords)
        deformed_particle_coords = deformation_function(self.particle_coords)

        # 2) Use the particles2grid function to create deformed grids
        deformed_mc_density_grid = particles2grid(
            particle_coords=deformed_mc_particle_coords,
            particle_values=self.particle_values_mc_density,
            xyz_min=self.base_grid[0].xyz_min,
            xyz_max=self.base_grid[0].xyz_max,
            grid_shape=self.base_grid[0].density.shape[-3:]
        )
        deformed_density_grid = particles2grid(
            particle_coords=deformed_particle_coords,
            particle_values=self.particle_values_density,
            xyz_min=self.base_grid[0].xyz_min,
            xyz_max=self.base_grid[0].xyz_max,
            grid_shape=self.base_grid[0].density.shape[-3:]
        )
        deformed_k0_grid = particles2grid(
            particle_coords=deformed_particle_coords,
            particle_values=self.particle_values_k0,
            xyz_min=self.base_grid[0].xyz_min,
            xyz_max=self.base_grid[0].xyz_max,
            grid_shape=self.base_grid[0].density.shape[-3:]
        )

        # 3) Assign the deformed grids back to the appropriate attributes of self.base_grid[0]
        self.base_grid[0].mask_cache.density = nn.Parameter(deformed_mc_density_grid, requires_grad=False)
        self.base_grid[0].density = nn.Parameter(deformed_density_grid, requires_grad=False)
        self.base_grid[0].k0 = nn.Parameter(deformed_k0_grid, requires_grad=False)

        # 4) Manually delete unneeded tensors to ensure they are erased from memory
        del deformed_particle_coords, deformed_mc_density_grid, deformed_density_grid, deformed_k0_grid

    def forward(self, rays_o: torch.Tensor, rays_d: torch.Tensor, **render_kwargs):
        # Wrapper of self.base_grid[0].forward
        # Should pass rays through the new (deformed) volume
        
        return self.base_grid[0](rays_o, rays_d, None, **render_kwargs)
    
def place_particles(grid_shape: "tuple[int, int, int]", xyz_min: torch.Tensor, xyz_max: torch.Tensor, particles_per_grid_cell: float) -> torch.Tensor:
    """Generate an [N, 3] tensor representing the coordinates of particles in the grid.

    Args:
        grid_shape (tuple[int, int, int]): A tuple of integers representing the size of the last 3 dimensions of the grid.
        xyz_min (torch.Tensor): A tensor of shape [1, 3] representing the lower bound of the bbox in each dimension.
        xyz_max (torch.Tensor): A tensor of shape [1, 3] representing the upper bound of the bbox in each dimension.
        particles_per_grid_cell (float): Number of particles to place in each grid cell.

    Returns:
        torch.Tensor: Coordinates transformed to index coords.
    """
    depth, height, width = grid_shape

    xyz = torch.meshgrid(
        torch.linspace(xyz_min[0], xyz_max[0], int(depth * particles_per_grid_cell ** (1/3))),
        torch.linspace(xyz_min[1], xyz_max[1], int(height * particles_per_grid_cell ** (1/3))),
        torch.linspace(xyz_min[2], xyz_max[2], int(width * particles_per_grid_cell ** (1/3))),
        indexing="ij"
    )
    xyz = torch.stack(xyz, -1).flatten(0, 2)
    return xyz