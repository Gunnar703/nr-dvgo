import taichi as ti
import torch

from .xyz2ijk import coords_xyz2ijk

@ti.kernel
def _g2p_kernel(
    grid_values         : ti.types.ndarray(),  # type: ignore
    particle_coords     : ti.types.ndarray(),  # type: ignore
    particle_values_ret : ti.types.ndarray(),  # type: ignore
):
    # Loop over all particles
    for i in ti.ndrange(particle_coords.shape[0]):
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
            for c in range(grid_values.shape[1]):
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
            for c in ti.ndrange(grid_values.shape[1]):
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


class _Grid2Particles(torch.autograd.Function):
    """
    Autograd function to interpolate in a volume grid using Taichi.
    """

    @staticmethod
    def forward(ctx, grid_values: torch.Tensor, particle_coords: torch.Tensor):
        """Get values associated with each particle by interpolating in the grid. Input coordinates must be in grid indical coordinates.

        *Grid Indical Coordinates*: A coordinate system where all grid values lie at integer coordinates. `coords_xyz2ijk` converts from world coordinates to grid indical coordinates.

        Args:
            grid_values (torch.Tensor): A tensor of shape [1, C, D, H, W]. Represents A C-dimensional vector field discretized at D x H x W points. Must be contiguous.
            particle_coords (torch.Tensor): A tensor of shape [N, 3]. Represents the coordinates of each particle. Must be contiguous.
        
        Returns:
            particle_values (torch.Tensor): A tensor of shape [N, C]. Represents the value of the vector field at each input position.
        """
        assert particle_coords.is_contiguous(), "particle_coords MUST be contiguous (call particle_coords.contiguous() before passing to this function)"
        assert grid_values.is_contiguous(), "grid_values MUST be contiguous (call grid_values.contiguous() before passing to this function)"

        # Create tensor to hold taichi kernel output
        num_channels = grid_values.shape[1]
        particle_values_ret = torch.zeros(
            (particle_coords.shape[0], num_channels),
            dtype=grid_values.dtype,
            device=grid_values.device,
            requires_grad=True
        )

        # Call taichi kernel; Modifies particle_values_ret in-place
        _g2p_kernel(
            grid_values=grid_values,
            particle_coords=particle_coords,
            particle_values_ret=particle_values_ret
        )

        # Save information for the backward pass
        ctx.save_for_backward(
            grid_values,
            particle_coords,
            particle_values_ret
        )
        return particle_values_ret
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Retrieve saved tensors
        grid_values, particle_coords, particle_values_ret = ctx.saved_tensors
        
        # grad_output is d(.)/d(output) and particle_values_ret is 
        # the output of this function, so grad_output is really
        # the derivative w.r.t particle_values_ret
        particle_values_ret.grad = grad_output.contiguous()

        # Accumulate gradients into grid_values and particle_coords
        _g2p_kernel.grad(grid_values, particle_coords, particle_values_ret)
        
        return grid_values.grad, particle_coords.grad

# Expose an easier way to call the g2p function 
def grid2particles(
    grid_values: torch.Tensor, 
    particle_coords: torch.Tensor, 
    xyz_min: torch.Tensor, 
    xyz_max: torch.Tensor
) -> torch.Tensor:
    """Get values associated with each particle by interpolating in the grid.

    Args:
        grid_values (torch.Tensor): A tensor of shape [1, C, D, H, W]. Represents A C-dimensional vector field discretized at D x H x W points. Must be contiguous.
        particle_coords (torch.Tensor): A tensor of shape [N, 3]. Represents the coordinates of each particle.
        xyz_min (torch.Tensor): A tensor of shape [1, 3] representing the lower bound of the bbox in each dimension.
        xyz_max (torch.Tensor): A tensor of shape [1, 3] representing the upper bound of the bbox in each dimension.
    
    Returns:
        particle_values (torch.Tensor): A tensor of shape [N, C]. Represents the value of the vector field at each input position.
    """
    
    # Convert from world coords to indical coords
    particle_coords = coords_xyz2ijk(particle_coords, grid_values.shape[-3:], xyz_min, xyz_max)

    # Prepare tensors for the taichi kernel/autograd function
    grid_values = grid_values.contiguous()
    particle_coords = particle_coords.contiguous()

    return _Grid2Particles.apply(grid_values, particle_coords)

