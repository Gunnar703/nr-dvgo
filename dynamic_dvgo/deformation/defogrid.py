from typing import Callable, Optional

import taichi as ti

import torch.nn as nn
import torch

@ti.data_oriented
class DeformableGrid:
    def __init__(self, grid: torch.Tensor, xyz_min: torch.Tensor, xyz_max: torch.Tensor):
        """
        Initializes a DeformableGrid module. This module supports deformation via
            1. Seeding a grid with particles at various (x, y, z) positions and assigning each particle a value based on its location
            2. Moving the particles according to some deformation function
            3. Re-assigning values to the grid based on the new distribution of particles

        Args:
            grid (torch.Tensor): A tensor of shape [1, C, D, H, W]. Represents a C-dimensional vector field in 3D space.
            xyz_min (torch.Tensor): A tensor of shape [3,]. Represents the lowest x, y, and z coordinate represented by the grid.
            xyz_max (torch.Tensor): A tensor of shape [3,]. Represents the highest x, y, and z coordinate represented by the grid.
        """
        self.xyz_min = xyz_min
        self.xyz_max = xyz_max
        self.base_grid = grid

    def coords_xyz2ijk(self, coords: torch.Tensor) -> torch.Tensor:
        """Normalize input coordinates to [0, MAX]. Also flips them so they represent (x, y, z) coordinates.

        Args:
            coords (torch.Tensor): A tensor of shape [N, 3] representing N 3-tuples of (z, y, x) coordinates.

        Returns:
            torch.Tensor: Coordinates transformed to index coords.
        """

        coords = (
            (coords.to(self.xyz_min) - self.xyz_min)
            / (self.xyz_max - self.xyz_min)
        )
        coords = coords.flip((-1,))

        x_mul, y_mul, z_mul = self.base_grid.shape[2:]
        x, y, z = coords.split(1, 1)
        x, y, z = x * (x_mul - 1), y * (y_mul - 1), z * (z_mul - 1)
        coords = torch.cat((x.view(-1, 1), y.view(-1, 1), z.view(-1, 1)), 1)
        return coords
    
    def place_particles(self, particles_per_grid_cell: float) -> torch.Tensor:
        depth, height, width = self.base_grid.shape[-3:]

        xyz = torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], int(depth * particles_per_grid_cell ** (1/3))),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], int(height * particles_per_grid_cell ** (1/3))),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], int(width * particles_per_grid_cell ** (1/3))),
            indexing="ij"
        )
        xyz = torch.stack(xyz, -1).flatten(0, 2)
        return xyz
    
    def grid2particles(self, particle_coords: torch.Tensor) -> torch.Tensor:
        """Get values associated with each particle by interpolating in the grid.

        Args:
            particle_coords (torch.Tensor): A tensor of shape [..., 3]. Represents the coordinates of each particle.
        
        Returns:
            particle_values (torch.Tensor): A tensor of shape [..., C]. Represents the value of the vector field at each input position.
        """

        # Reshape particle coords for grid_sample
        shape = particle_coords.shape[:-1]
        particle_coords = particle_coords.reshape(-1, 3)

        # Interpolate in the grid
        particle_coords = self.coords_xyz2ijk(particle_coords)

        # Prepare tensors for Taichi kernel
        grid_values = self.base_grid.contiguous()
        particle_coords = particle_coords.contiguous()
        particle_values_ret = torch.zeros((particle_coords.shape[0], grid_values.shape[1]), dtype=torch.float32, device=self.base_grid.device).contiguous()

        self._grid2particles_kernel(
            grid_values=grid_values,
            particle_coords=particle_coords,
            particle_values_ret=particle_values_ret
        )

        # Reshape back to original shape of particle_coords
        particle_values = particle_values_ret.reshape(*shape, -1)
        
        return particle_values

    def particles2grid(self, particle_coords: torch.Tensor, particle_values: torch.Tensor):
        """Wrapper for Lagrangian to Eulerian transfer. Prepares tensors, then passes them to the Taichi kernel.

        Args:
            particle_coords (torch.Tensor): A tensor of shape [..., 3] representing the positions of all particles
            particle_values (torch.Tensor): A tensor of shape [..., C] representing the C-dimensional property vector associated with each particle
        """
        # Convert both tensors to N x C and N x 3 respectively
        particle_coords = particle_coords.reshape(-1, 3)
        particle_values = particle_values.reshape(particle_coords.shape[0], -1)

        # Normalize coordinates to [0, MAX] along each axis
        particle_coords = self.coords_xyz2ijk(particle_coords)

        # Prepare tensors to be passed to taichi kernel
        particle_coords = particle_coords.contiguous()
        particle_values = particle_values.contiguous()
        grid_values_ret = torch.zeros_like(self.base_grid, dtype=torch.float32, device=self.base_grid.device).contiguous()
        numerator       = torch.zeros_like(self.base_grid, dtype=torch.float32, device=self.base_grid.device).contiguous()
        denominator     = torch.zeros(1, 1, *self.base_grid.shape[-3:], dtype=torch.float32, device=self.base_grid.device).contiguous()

        # Call taichi kernel
        self._particles2grid_kernel(
            particle_coords=particle_coords,
            particle_values=particle_values,
            grid_values_ret=grid_values_ret,  # these three variables are mutated in-place by taichi
            numerator=numerator,              # these three variables are mutated in-place by taichi
            denominator=denominator,          # these three variables are mutated in-place by taichi
        )

        return grid_values_ret

    def deform_grid(self, deformation_function: Callable[[torch.Tensor], torch.Tensor], particles_per_grid_cell: int):
        """Return a deformed version of self.base_grid

        Args:
            deformation_function (Callable[[torch.Tensor], torch.Tensor]): Function that maps an N x 3 tensor to another N x 3 tensor: (x, y, z) -> (x', y', z') where (x', y', z') is the new position of a particle originally at (x, y, z)

        Returns:
            torch.Tensor: 1 x C x D x H x W tensor representing the deformed grid/volume
        """

        # Seed the grid with particles
        particle_coords = self.place_particles(particles_per_grid_cell)

        # Transfer values from the grid to the particles
        particle_values = self.grid2particles(particle_coords)

        # Move the particles
        particle_coords = deformation_function(particle_coords)

        # Transfer values from the particles back to the grid
        deformed_grid = self.particles2grid(particle_coords, particle_values)

        return deformed_grid

    @ti.kernel
    def _particles2grid_kernel(
        self,
        particle_coords : ti.types.ndarray(),  # type: ignore
        particle_values : ti.types.ndarray(),  # type: ignore
        grid_values_ret : ti.types.ndarray(),  # type: ignore
        numerator       : ti.types.ndarray(),  # type: ignore
        denominator     : ti.types.ndarray(),  # type: ignore
    ):
        """Taichi kernel to perform the lagrangian to eulerian conversion.

        Args:
            particle_coords (ti.types.ndarray): N x 3 tensor of (x, y, z) coordinates of particles
            particle_values (ti.types.ndarray): N x C tensor of the C-dimensional property vectors associated with each particle
            grid_values_ret (ti.types.ndarray): 1 x C x D x H x W tensor to hold return values
            numerator (ti.types.ndarray): 1 x C x D x H x W tensor used in calculations (numerator of PACNeRF Eq. 5)
            denominator (ti.types.ndarray): 1 x C x D x H x W tensor used in calculations (denominator of PACNeRF Eq. 5)
        
        Returns:
            None: return value is directly written to tensor passed to grid_values_ret argument
        """

        # Determine number of particles/number of channels in the grid
        num_particles = particle_coords.shape[0]
        num_grid_channels = grid_values_ret.shape[1]

        # Loop over all particles in the list
        for i in ti.ndrange(num_particles):

            # Get the coordinates of each particle
            (
                particle_i, 
                particle_j, 
                particle_k
            ) = (
                particle_coords[i, 0],
                particle_coords[i, 1],
                particle_coords[i, 2]
            )

            # Check if i, j, k is in bounds
            # Skip particles that are out of bounds
            if (
                (0 <= particle_i) and (particle_i < grid_values_ret.shape[2] - 1)
                and (0 <= particle_j) and (particle_j < grid_values_ret.shape[3] - 1)
                and (0 <= particle_k) and (particle_k < grid_values_ret.shape[4] - 1)
            ): 
                # Coordinate is in bounds
                # Check if any of the particle coordinates are integers. 
                # This corresponds to cases where they lie exactly on a grid line.
                if ti.abs(particle_i - ti.floor(particle_i)) < 1e-12:
                    particle_i -= 0.5  # move the particle to left/back/down by half a cell
                    if particle_i < 0: particle_i += 1  # if this moves it out of bounds, put it back in bounds
                if ti.abs(particle_j - ti.floor(particle_j)) < 1e-12:
                    particle_j -= 0.5  # move the particle to left/back/down by half a cell
                    if particle_j < 0: particle_j += 1  # if this moves it out of bounds, put it back in bounds
                if ti.abs( particle_j- ti.floor(particle_j)) < 1e-12:
                    particle_j -= 0.5  # move the particle to left/back/down by half a cell
                    if particle_j < 0: particle_j += 1  # if this moves it out of bounds, put it back in bounds

                # Get the bounding grid indices
                ## Upper
                ub_i: ti.int32 = ti.ceil(particle_i)  # type: ignore
                ub_j: ti.int32 = ti.ceil(particle_j)  # type: ignore
                ub_k: ti.int32 = ti.ceil(particle_k)  # type: ignore

                ## Lower
                lb_i: ti.int32 = ti.floor(particle_i)  # type: ignore
                lb_j: ti.int32 = ti.floor(particle_j)  # type: ignore
                lb_k: ti.int32 = ti.floor(particle_k)  # type: ignore

                # Bilinear interpolation weights
                # naming = w_(lower/upper in x)(lower/upper in y)(lower/upper in z)
                w_lll = (particle_i - lb_i) * (particle_j - lb_j) * (particle_k - lb_k)
                w_llu = (particle_i - lb_i) * (particle_j - lb_j) * (ub_k - particle_k)
                w_luu = (particle_i - lb_i) * (ub_j - particle_j) * (ub_k - particle_k)
                w_lul = (particle_i - lb_i) * (ub_j - particle_j) * (particle_k - lb_k)
                w_ull = (ub_i - particle_i) * (particle_j - lb_j) * (particle_k - lb_k)
                w_ulu = (ub_i - particle_i) * (particle_j - lb_j) * (ub_k - particle_k)
                w_uuu = (ub_i - particle_i) * (ub_j - particle_j) * (ub_k - particle_k)
                w_uul = (ub_i - particle_i) * (ub_j - particle_j) * (particle_k - lb_k)

                # Eulerian to Lagrangian conversion
                ## Assemble the numerator of the lagrangian-to-eulerian interpolation function
                for c in ti.ndrange(num_grid_channels):
                    numerator[0, c, lb_i, lb_j, lb_k] += w_lll * particle_values[i, c]
                    numerator[0, c, lb_i, lb_j, ub_k] += w_llu * particle_values[i, c]
                    numerator[0, c, lb_i, ub_j, ub_k] += w_luu * particle_values[i, c]
                    numerator[0, c, lb_i, ub_j, lb_k] += w_lul * particle_values[i, c]
                    numerator[0, c, ub_i, lb_j, lb_k] += w_ull * particle_values[i, c]
                    numerator[0, c, ub_i, lb_j, ub_k] += w_ulu * particle_values[i, c]
                    numerator[0, c, ub_i, ub_j, ub_k] += w_uuu * particle_values[i, c]
                    numerator[0, c, ub_i, ub_j, lb_k] += w_uul * particle_values[i, c]
                
                ## Assemble the denominator of the lagrangian-to-eulerian interpolation function
                denominator[0, 0, lb_i, lb_j, lb_k] += w_lll
                denominator[0, 0, lb_i, lb_j, ub_k] += w_llu
                denominator[0, 0, lb_i, ub_j, ub_k] += w_luu
                denominator[0, 0, lb_i, ub_j, lb_k] += w_lul
                denominator[0, 0, ub_i, lb_j, lb_k] += w_ull
                denominator[0, 0, ub_i, lb_j, ub_k] += w_ulu
                denominator[0, 0, ub_i, ub_j, ub_k] += w_uuu
                denominator[0, 0, ub_i, ub_j, lb_k] += w_uul
        
        depth, height, width = grid_values_ret.shape[-3:]
        for c, i, j, k in ti.ndrange(num_grid_channels, depth, height, width):
            if denominator[0, 0, i, j, k] > 1e-10:
                # If denominator is zero, grid index i, j, k is underdetermined
                # (i.e. there are no particles in its neighboring cells)
                # For these cases, let the grid stay zero
                grid_values_ret[0, c, i, j, k] = numerator[0, c, i, j, k] / denominator[0, 0, i, j, k]

    @ti.kernel
    def _grid2particles_kernel(
        self,
        grid_values         : ti.types.ndarray(),  # type: ignore
        particle_coords     : ti.types.ndarray(),  # type: ignore
        particle_values_ret : ti.types.ndarray(),  # type: ignore
    ):
        
        num_particles = particle_coords.shape[0]
        num_channels = grid_values.shape[1]

        # Loop over all particles
        for i in range(num_particles):
            # Get the coordinates of each particle
            (
                particle_i, 
                particle_j, 
                particle_k
            ) = (
                particle_coords[i, 0],
                particle_coords[i, 1],
                particle_coords[i, 2]
            )
            
            # Check if i, j, k is out of bounds (notice the 'not')
            if not (
                (0 <= particle_i) and (particle_i < grid_values.shape[2] - 1)
                and (0 <= particle_j) and (particle_j < grid_values.shape[3] - 1)
                and (0 <= particle_k) and (particle_k < grid_values.shape[4] - 1)
            ): 
                # Coordinate is out of bounds
                # Set particle value to zero
                for c in range(num_channels):
                    particle_values_ret[i, c] = 0
            else:
                # Coordinate is in bounds
                # Check if any of the particle coordinates are integers. 
                # This corresponds to cases where they lie exactly on a grid line.
                if ti.abs(particle_i - ti.floor(particle_i)) < 1e-12:
                    particle_i -= 0.1  # move the particle to left/back/down
                    if particle_i < 0: particle_i += 1  # if this moves it out of bounds, put it back in bounds
                if ti.abs(particle_j - ti.floor(particle_j)) < 1e-12:
                    particle_j -= 0.1  
                    if particle_j < 0: particle_j += 1
                if ti.abs( particle_j- ti.floor(particle_j)) < 1e-12:
                    particle_j -= 0.1 
                    if particle_j < 0: particle_j += 1

                # Get the bounding grid indices
                ## Upper
                ub_i: ti.int32 = ti.ceil(particle_i)  # type: ignore
                ub_j: ti.int32 = ti.ceil(particle_j)  # type: ignore
                ub_k: ti.int32 = ti.ceil(particle_k)  # type: ignore

                ## Lower
                lb_i: ti.int32 = ti.floor(particle_i)  # type: ignore
                lb_j: ti.int32 = ti.floor(particle_j)  # type: ignore
                lb_k: ti.int32 = ti.floor(particle_k)  # type: ignore

                # Bilinear interpolation weights
                # naming = w_(lower/upper in x)(lower/upper in y)(lower/upper in z)
                w_lll = (particle_i - lb_i) * (particle_j - lb_j) * (particle_k - lb_k)
                w_llu = (particle_i - lb_i) * (particle_j - lb_j) * (ub_k - particle_k)
                w_luu = (particle_i - lb_i) * (ub_j - particle_j) * (ub_k - particle_k)
                w_lul = (particle_i - lb_i) * (ub_j - particle_j) * (particle_k - lb_k)
                w_ull = (ub_i - particle_i) * (particle_j - lb_j) * (particle_k - lb_k)
                w_ulu = (ub_i - particle_i) * (particle_j - lb_j) * (ub_k - particle_k)
                w_uuu = (ub_i - particle_i) * (ub_j - particle_j) * (ub_k - particle_k)
                w_uul = (ub_i - particle_i) * (ub_j - particle_j) * (particle_k - lb_k)

                # Interpolate
                for c in range(num_channels):
                    val = (
                        w_lll * grid_values[0, c, lb_i, lb_j, lb_k]
                        + w_llu * grid_values[0, c, lb_i, lb_j, ub_k]
                        + w_luu * grid_values[0, c, lb_i, ub_j, ub_k]
                        + w_lul * grid_values[0, c, lb_i, ub_j, lb_k]
                        + w_ull * grid_values[0, c, ub_i, lb_j, lb_k]
                        + w_ulu * grid_values[0, c, ub_i, lb_j, ub_k]
                        + w_uuu * grid_values[0, c, ub_i, ub_j, ub_k]
                        + w_uul * grid_values[0, c, ub_i, ub_j, ub_k]
                    )
                    particle_values_ret[i, c] = val
        

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ti.init()
    
    grid = torch.zeros(1, 3, 80, 40, 40)
    grid[0, :, 10:-10, 15:-15, 15:-15] = 1

    xyz_min = torch.tensor([-2, -2, -2], dtype=torch.float32)
    xyz_max = -xyz_min

    deformable_grid = DeformableGrid(grid, xyz_min, xyz_max)

    def deformation_function(original_coords: torch.Tensor):
        x, y, z = original_coords.split(1, -1)

        new_x = x
        new_y = y + torch.sin(input=2*z) / 2
        new_z = z

        new_coords = torch.stack((new_x, new_y, new_z), -1)
        return new_coords
    
    deformed_grid = deformable_grid.deform_grid(deformation_function, particles_per_grid_cell=4)
    plt.figure()
    plt.subplot(121)
    plt.imshow(grid[0, :, :, :, 15].permute(1, 2, 0).cpu().numpy())

    plt.subplot(122)
    plt.imshow(deformed_grid[0, :, :, :, 15].permute(1, 2, 0).cpu().numpy())
    plt.show()