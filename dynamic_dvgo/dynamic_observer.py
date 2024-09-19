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
        self.mask_cache_density = torch.clone(mask_cache_density.detach())
        self.density = torch.clone(density.detach())
        self.k0 = torch.clone(k0.detach())

        # Seed the new grid with particles. We want to do this here in order
        # to keep a consistent set of particles across all calls to deform_grid
        self.mc_particle_coords = place_particles(
            grid_shape=mask_cache_density.shape[-3:],
            xyz_min=mc_xyz_min,
            xyz_max=mc_xyz_max,
            particles_per_grid_cell=self.n_particles
        ).contiguous()

        self.particle_coords = place_particles(
            grid_shape=density.shape[-3:],
            xyz_min=xyz_min,
            xyz_max=xyz_max,
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
    
    def _scale_grid(self, scale_factor_xyz: float, original_grid: torch.Tensor, xyz_min: torch.Tensor, xyz_max: torch.Tensor):
        # # Compute the side lengths of a voxel
        # x_max, y_max, z_max = xyz_max
        # x_min, y_min, z_min = xyz_min
        # nchannels, nvox_z, nvox_y, nvox_x = original_grid.shape[1:]

        # vox_len_x = (x_max - x_min) / nvox_x
        # vox_len_y = (y_max - y_min) / nvox_y
        # vox_len_z = (z_max - z_min) / nvox_z

        # # Compute the new grid dimensions
        # nvox_x_new = int(nvox_x * scale_factor_xyz[0])
        # nvox_y_new = int(nvox_y * scale_factor_xyz[1])
        # nvox_z_new = int(nvox_z * scale_factor_xyz[2])
        
        # # Create the new grid
        # new_grid = torch.zeros(1, nchannels, nvox_z_new, nvox_y_new, nvox_x_new).to(original_grid)

        # # Place the old grid in the center of the new one
        # old_grid_start_i = (nvox_z_new - nvox_z) // 2
        # old_grid_start_j = (nvox_y_new - nvox_y) // 2
        # old_grid_start_k = (nvox_x_new - nvox_x) // 2

        # old_grid_end_i = old_grid_start_i + nvox_z
        # old_grid_end_j = old_grid_start_j + nvox_y
        # old_grid_end_k = old_grid_start_k + nvox_x

        # new_grid[
        #     0, :,
        #     old_grid_start_i : old_grid_end_i,
        #     old_grid_start_j : old_grid_end_j,
        #     old_grid_start_k : old_grid_end_k
        # ] = original_grid[0, :]

        # # Compute new xyz_min and xyz_max
        # start_padding = torch.tensor([
        #     old_grid_start_k * vox_len_x,
        #     old_grid_start_j * vox_len_y,
        #     old_grid_start_i * vox_len_z,
        # ]).to(xyz_max)

        # end_padding = torch.tensor([
        #     (nvox_x_new - old_grid_end_k) * vox_len_x,
        #     (nvox_y_new - old_grid_end_j) * vox_len_y,
        #     (nvox_z_new - old_grid_end_i) * vox_len_z,
        # ]).to(xyz_max)

        # new_xyz_min = xyz_min - start_padding
        # new_xyz_max = xyz_max + end_padding

        # print(f"{vox_len_x=}, {vox_len_y=}, {vox_len_z=}")

        # return new_xyz_min, new_xyz_max, new_grid
        return xyz_min, xyz_max, original_grid

    def deform_grid(self, deformation_function: Callable[[torch.Tensor], torch.Tensor]):
        # deformation function: (x, y, z) -> (x', y', z')

        # This function (deform_grid) should:
        # 1) Compute the new positions of each particle using deformation_function
        # 2) Use the particles2grid function to create deformed grids
        # 3) Assign the deformed grids back to the appropriate attributes of self.base_grid[0]
        # 4) Manually delete unneeded tensors to ensure they are erased from memory

        # 1) Compute the new positions of each particle using deformation_function
        deformed_particle_coords = deformation_function(self.particle_coords)

        # 2) Use the particles2grid function to create deformed grids
        deformed_mc_density_grid = particles2grid(
            particle_coords=deformed_particle_coords,
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