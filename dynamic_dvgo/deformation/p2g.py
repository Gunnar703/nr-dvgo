import taichi as ti
import torch

from .xyz2ijk import coords_xyz2ijk

@ti.kernel
def _p2g_kernel_part_1(
    particle_coords : ti.types.ndarray(),  # type: ignore
    particle_values : ti.types.ndarray(),  # type: ignore
    numerator       : ti.types.ndarray(),  # type: ignore
    denominator     : ti.types.ndarray(),  # type: ignore
):
    """First part of Taichi kernel to perform the lagrangian to eulerian conversion. Computes the numerator and denominator PACNeRF Eq. 5

    Args:
        particle_coords (ti.types.ndarray): N x 3 tensor of (x, y, z) coordinates of particles
        particle_values (ti.types.ndarray): N x C tensor of the C-dimensional property vectors associated with each particle
        numerator (ti.types.ndarray): 1 x C x D x H x W tensor used in calculations (numerator of PACNeRF Eq. 5)
        denominator (ti.types.ndarray): 1 x C x D x H x W tensor used in calculations (denominator of PACNeRF Eq. 5)
    
    Returns:
        None: return valus are directly written to tensors passed to numerator and denominator arguments
    """

    # Loop over all particles in the list
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

        # Check if i, j, k is in bounds
        # Skip particles that are out of bounds
        if (
            (0 <= particle_i) and (particle_i < numerator.shape[2] - 1)
            and (0 <= particle_j) and (particle_j < numerator.shape[3] - 1)
            and (0 <= particle_k) and (particle_k < numerator.shape[4] - 1)
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
            for c in ti.ndrange(numerator.shape[1]):
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
    
def _p2g_kernel_part_2(
    numerator       : ti.types.ndarray(),  # type: ignore
    denominator     : ti.types.ndarray(),  # type: ignore
    grid_values_ret : ti.types.ndarray(),  # type: ignore
):
    """First part of Taichi kernel to perform the lagrangian to eulerian conversion. Computes the numerator and denominator PACNeRF Eq. 5

    Args:
        numerator (ti.types.ndarray): 1 x C x D x H x W tensor used in calculations (numerator of PACNeRF Eq. 5)
        denominator (ti.types.ndarray): 1 x C x D x H x W tensor used in calculations (denominator of PACNeRF Eq. 5)
    
    Returns:
        None: return value is directly written to tensor passed to grid_values_ret argument
    """
    for c, i, j, k in ti.ndrange(numerator.shape[1], *grid_values_ret.shape[-3:]):
        if denominator[0, 0, i, j, k] > 1e-10:
            # If denominator is zero, grid index i, j, k is underdetermined
            # (i.e. there are no particles in its neighboring cells)
            # For these cases, let the grid stay zero
            grid_values_ret[0, c, i, j, k] = numerator[0, c, i, j, k] / denominator[0, 0, i, j, k]


class _Particles2Grid(torch.autograd.Function):

    @staticmethod
    def forward(ctx, particle_coords: torch.Tensor, particle_values: torch.Tensor, grid_shape: "tuple[int, int, int]") -> torch.Tensor:
        """Wrapper for Lagrangian to Eulerian transfer. Particle coordinates must be given in grid indical coordinates.

        *Grid Indical Coordinates*: A coordinate system where all grid values lie at integer coordinates. `coords_xyz2ijk` converts from world coordinates to grid indical coordinates.

        Args:
            particle_coords (torch.Tensor): A tensor of shape [N, 3] representing the positions of all particles. Must be contiguous.
            particle_values (torch.Tensor): A tensor of shape [N, C] representing the C-dimensional property vector associated with each particle. Must be contiguous.
            grid_shape (tuple[int, int, int]): A tuple of integers representing the size of the last 3 dimensions of the grid.

        Returns:
            torch.Tensor: A tensor of shape [1, C, D, H, W]. Represents A C-dimensional vector field discretized at D x H x W points.
        """
        assert particle_coords.is_contiguous(), "particle_coords MUST be contiguous (call particle_coords.contiguous() before passing to this function)"
        assert particle_values.is_contiguous(), "particle_values MUST be contiguous (call particle_values.contiguous() before passing to this function)"

        # Create tensor to hold taichi kernel output
        num_channels = particle_values.shape[1]
        grid_values_ret = torch.zeros(
            (1, num_channels, *grid_shape),
            dtype=particle_values.dtype,
            device=particle_values.device,
            requires_grad=True
        )

        # Create numerator and denominator tensors. Used as a scratchpad by the kernel.
        numerator = torch.zeros(
            size=(1, num_channels, *grid_shape),
            dtype=particle_values.dtype,
            device=particle_values.device,
        )
        denominator = torch.zeros(
            size=(1, 1, *grid_shape),
            dtype=particle_values.dtype,
            device=particle_values.device
        )

        # Call taichi kernels; Modifies grid_values_ret in-place
        _p2g_kernel_part_1(
            particle_coords=particle_coords,
            particle_values=particle_values,
            numerator=numerator,
            denominator=denominator,
        )

        _p2g_kernel_part_2(
            numerator=numerator,
            denominator=denominator,
            grid_values_ret=grid_values_ret
        )


        # Save information for the backward pass
        ctx.save_for_backward(
            particle_coords,
            particle_values,
            grid_values_ret,
            numerator,
            denominator
        )
        return grid_values_ret
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Retrieve saved tensors
        (
            particle_coords,
            particle_values,
            grid_values_ret,
            numerator,
            denominator
        ) = ctx.saved_tensors

        # grad_output is d(.)/d(output) and grid_values_ret is 
        # the output of this function, so grad_output is really
        # the derivative w.r.t grid_values_ret
        grid_values_ret.grad = grad_output.contiguous()

        # Accumulate gradients into numerator and denominator
        _p2g_kernel_part_2.grad(numerator, denominator, grid_values_ret)

        # Accumulate gradients into particle_coords and particle_values
        _p2g_kernel_part_1.grad(particle_coords, particle_values, numerator, denominator)

        return particle_coords.grad, particle_values.grad, None

# Expose an easier way to call the p2g function
def particles2grid(
    particle_coords: torch.Tensor,
    particle_values: torch.Tensor,
    xyz_min: torch.Tensor,
    xyz_max: torch.Tensor,
    grid_shape: "tuple[int, int, int]",
) -> torch.Tensor:
    """Gets values at each grid point from the position and values of particles within the grid.

    Args:
        particle_coords (torch.Tensor): A tensor of shape [N, 3]. Represents the coordinates of each particle in world coordinates.
        particle_values (torch.Tensor): A tensor of shape [N, C]. Represents the value of the vector field at each input position.
        xyz_min (torch.Tensor): A tensor of shape [1, 3] representing the lower bound of the bbox in each dimension.
        xyz_max (torch.Tensor): A tensor of shape [1, 3] representing the upper bound of the bbox in each dimension.
        grid_shape (tuple[int, int, int]): A tuple of integers representing the size of the last 3 dimensions of the grid.

    Returns:
        torch.Tensor: A tensor of shape [1, C, D, H, W]. Represents A C-dimensional vector field discretized at D x H x W points.
    """

    # Convert from world coords to indical coords
    particle_coords = coords_xyz2ijk(particle_coords, grid_shape, xyz_min, xyz_max)

    # Prepare tensors for the taichi kernel/autograd function
    particle_coords = particle_coords.contiguous()
    particle_values = particle_values.contiguous()

    return _Particles2Grid.apply(particle_coords, particle_values, grid_shape)

