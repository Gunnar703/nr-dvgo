import taichi as ti
import torch

if __name__ == "__main__":
    from xyz2ijk import coords_xyz2ijk
else:
    from .xyz2ijk import coords_xyz2ijk

@ti.kernel
def _p2g_kernel_part_1(
    particle_coords       : ti.types.ndarray(),  # type: ignore
    particle_values       : ti.types.ndarray(),  # type: ignore
    numerator_denominator : ti.types.ndarray(),  # type: ignore
):
    """First part of Taichi kernel to perform the lagrangian to eulerian conversion. Computes the numerator and denominator PACNeRF Eq. 5

    Args:
        particle_coords (ti.types.ndarray): N x 3 tensor of (x, y, z) coordinates of particles
        particle_values (ti.types.ndarray): N x C tensor of the C-dimensional property vectors associated with each particle
        numerator_denominator (ti.types.ndarray): 1 x (C + 1) x D x H x W tensor used in calculations (numerator/denominator of PACNeRF Eq. 5)
    
    Returns:
        None: return valus are directly written to tensors passed to numerator_denominator argument
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
            (0 <= particle_i) and (particle_i < numerator_denominator.shape[2] - 1)
            and (0 <= particle_j) and (particle_j < numerator_denominator.shape[3] - 1)
            and (0 <= particle_k) and (particle_k < numerator_denominator.shape[4] - 1)
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
            num_channels = numerator_denominator.shape[1] - 1
            for c in ti.ndrange(num_channels):
                numerator_denominator[0, c, lb_i, lb_j, lb_k] += w_lll * particle_values[i, c]
                numerator_denominator[0, c, lb_i, lb_j, ub_k] += w_llu * particle_values[i, c]
                numerator_denominator[0, c, lb_i, ub_j, ub_k] += w_luu * particle_values[i, c]
                numerator_denominator[0, c, lb_i, ub_j, lb_k] += w_lul * particle_values[i, c]
                numerator_denominator[0, c, ub_i, lb_j, lb_k] += w_ull * particle_values[i, c]
                numerator_denominator[0, c, ub_i, lb_j, ub_k] += w_ulu * particle_values[i, c]
                numerator_denominator[0, c, ub_i, ub_j, ub_k] += w_uuu * particle_values[i, c]
                numerator_denominator[0, c, ub_i, ub_j, lb_k] += w_uul * particle_values[i, c]
            
            ## Assemble the denominator of the lagrangian-to-eulerian interpolation function
            numerator_denominator[0, num_channels, lb_i, lb_j, lb_k] += w_lll
            numerator_denominator[0, num_channels, lb_i, lb_j, ub_k] += w_llu
            numerator_denominator[0, num_channels, lb_i, ub_j, ub_k] += w_luu
            numerator_denominator[0, num_channels, lb_i, ub_j, lb_k] += w_lul
            numerator_denominator[0, num_channels, ub_i, lb_j, lb_k] += w_ull
            numerator_denominator[0, num_channels, ub_i, lb_j, ub_k] += w_ulu
            numerator_denominator[0, num_channels, ub_i, ub_j, ub_k] += w_uuu
            numerator_denominator[0, num_channels, ub_i, ub_j, lb_k] += w_uul

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
        assert particle_values.shape[0] == particle_coords.shape[0], "particle_coords and particle_values MUST have the same shape in dimension 0"

        # Create numerator and denominator tensor. Used as a scratchpad by the kernel.
        num_channels = particle_values.shape[1]
        numerator_denominator = torch.zeros(
            size=(1, num_channels + 1, *grid_shape),
            dtype=particle_values.dtype,
            device=particle_values.device,
        )

        # Call taichi kernel; Modifies numerator_denominator in-place
        _p2g_kernel_part_1(
            particle_coords=particle_coords,
            particle_values=particle_values,
            numerator_denominator=numerator_denominator,
        )

        # Save information for the backward pass
        ctx.save_for_backward(
            particle_coords,
            particle_values,
            numerator_denominator,
        )
        return numerator_denominator
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Retrieve saved tensors
        (
            particle_coords,
            particle_values,
            numerator_denominator,
        ) = ctx.saved_tensors

        # grad_output is d(.)/d(output) and numerator_denominator is 
        # the output of this function, so grad_output is really
        # the derivative w.r.t numerator_denominator
        numerator_denominator.grad = grad_output.contiguous()

        # Accumulate gradients into particle_coords and particle_values
        _p2g_kernel_part_1.grad(particle_coords, particle_values, numerator_denominator)

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

    # Get numerator and denominator of the interpolation function
    numerator_denominator = _Particles2Grid.apply(particle_coords, particle_values, grid_shape)
    numerator = numerator_denominator[0:1, :-1].view(1, numerator_denominator.shape[1] - 1, *numerator_denominator.shape[2:])
    denominator = numerator_denominator[0:1, -1].view(1, 1, *numerator_denominator.shape[2:])

    # Divide only at locations where denominator != 0
    indices = denominator[0, 0] > 0
    result = torch.zeros_like(numerator).to(numerator)
    result[0, :, indices] = numerator[0, :, indices] / denominator[0, 0, indices]

    return result


if __name__ == "__main__":
    # Test to verify that the module works in the forward pass and is differentiable

    ti.init()

    particle_coords = torch.rand(1000, 3) * 2 - 1
    particle_coords.requires_grad = True
    particle_values = torch.sin(particle_coords)
    

    xyz_max = torch.tensor([-1, -1, -1], dtype=torch.float32).view(1, 3)
    xyz_min = -xyz_max

    grid_shape = (100, 100, 100)

    grid_values = particles2grid(particle_coords, particle_values, xyz_min, xyz_max, grid_shape)
    print(f"{grid_values.shape=}", f"{grid_values.min()=}", f"{grid_values.max()=}")

    test_objective = grid_values.max()

    test_objective.backward()

    print(f"{particle_coords.grad.max()=}")

