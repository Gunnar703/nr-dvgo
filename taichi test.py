import matplotlib.pyplot as plt
import taichi as ti

import torch
from time import time

ti.init()

init_image = r"C:\Projects\dvgo\data\beam\equilibrium\Camera_4_equilibrium.png"
init_image = plt.imread(init_image)
init_image = torch.tensor(init_image)

# Create particles
particles_i, particles_j = torch.meshgrid(torch.linspace(-1, 1, init_image.shape[0] * 2),
                                                          torch.linspace(-1, 1, init_image.shape[1] * 2),
                                                          indexing="ij")
particle_coords: torch.Tensor  = torch.stack((particles_i.ravel(), particles_j.ravel()), axis=-1)
eulerian_grid_coords = torch.stack((particles_i, particles_j), axis=-1)
eulerian_grid_vals = init_image.clone()

def G2P(eulerian_grid: torch.Tensor, particle_coords: torch.Tensor):
    """
    eulerian_grid: H x W x C
    particle_coords: N x 2 (normalized so that [-1, 1] bounds the scene)
    """

    particle_colors = torch.nn.functional.grid_sample(eulerian_grid.permute(2, 0, 1)[None], particle_coords.flip(1)[None, None], align_corners=False)[0, :, 0].T
    return particle_colors

@ti.kernel
def P2G(
    particle_values: ti.types.ndarray(),  # type: ignore
    particle_coords: ti.types.ndarray(),  # type: ignore
    grid_values_ret: ti.types.ndarray(),  # type: ignore
    numerator: ti.types.ndarray(),        # type: ignore
    denominator: ti.types.ndarray(),      # type: ignore
):
    
    for i, j in ti.ndrange(particle_coords.shape[0], 2):
        particle_coords[i, j] += 1
        particle_coords[i, j] /= 2
    
    for i in range(particle_coords.shape[0]):
        particle_coords[i, 0] *= grid_values_ret.shape[0] - 1
        particle_coords[i, 1] *= grid_values_ret.shape[1] - 1

    for i in range(particle_coords.shape[0]):
        pi, pj = particle_coords[i, 0], particle_coords[i, 1]

        lb_i: ti.int32 = ti.floor(pi) # type: ignore
        ub_i: ti.int32 = ti.ceil(pi)  # type: ignore
        lb_j: ti.int32 = ti.floor(pj) # type: ignore
        ub_j: ti.int32 = ti.ceil(pj)  # type: ignore

        if (lb_i == ub_i) and (lb_i <  grid_values_ret.shape[0] - 1): ub_i = lb_i + 1
        if (lb_i == ub_i) and (lb_i == grid_values_ret.shape[0] - 1): lb_i = ub_i - 1
        if (lb_j == ub_j) and (lb_j <  grid_values_ret.shape[0] - 1): ub_j = lb_j + 1
        if (lb_j == ub_j) and (lb_j == grid_values_ret.shape[0] - 1): lb_j = ub_j - 1

        w_ll = (pi - lb_i) * (pj - lb_j)
        w_lu = (pi - lb_i) * (ub_j - pj)
        w_ul = (ub_i - pi) * (pj - lb_j)
        w_uu = (ub_i - pi) * (ub_j - pj)

        numerator[lb_i, lb_j, 0] += w_ll * particle_values[i, 0]
        numerator[lb_i, lb_j, 1] += w_ll * particle_values[i, 1]
        numerator[lb_i, lb_j, 2] += w_ll * particle_values[i, 2]
        numerator[lb_i, lb_j, 3] += w_ll * particle_values[i, 3]
        numerator[lb_i, ub_j, 0] += w_lu * particle_values[i, 0]
        numerator[lb_i, ub_j, 1] += w_lu * particle_values[i, 1]
        numerator[lb_i, ub_j, 2] += w_lu * particle_values[i, 2]
        numerator[lb_i, ub_j, 3] += w_lu * particle_values[i, 3]
        numerator[ub_i, lb_j, 0] += w_ul * particle_values[i, 0]
        numerator[ub_i, lb_j, 1] += w_ul * particle_values[i, 1]
        numerator[ub_i, lb_j, 2] += w_ul * particle_values[i, 2]
        numerator[ub_i, lb_j, 3] += w_ul * particle_values[i, 3]
        numerator[ub_i, ub_j, 0] += w_uu * particle_values[i, 0]
        numerator[ub_i, ub_j, 1] += w_uu * particle_values[i, 1]
        numerator[ub_i, ub_j, 2] += w_uu * particle_values[i, 2]
        numerator[ub_i, ub_j, 3] += w_uu * particle_values[i, 3]

        denominator[lb_i, lb_j] += w_ll
        denominator[lb_i, ub_j] += w_lu
        denominator[ub_i, lb_j] += w_ul
        denominator[ub_i, ub_j] += w_uu

    for i, j in ti.ndrange(*denominator.shape):
        grid_values_ret[i, j, 0] = numerator[i, j, 0] / denominator[i, j]
        grid_values_ret[i, j, 1] = numerator[i, j, 1] / denominator[i, j]
        grid_values_ret[i, j, 2] = numerator[i, j, 2] / denominator[i, j]
        grid_values_ret[i, j, 3] = numerator[i, j, 3] / denominator[i, j]

grid_values = torch.zeros_like(init_image, dtype=torch.float32)
numerator = torch.zeros(init_image.shape, dtype=torch.float32)
denominator = torch.zeros(init_image.shape[:-1], dtype=torch.float32)
particle_values = G2P(init_image, particle_coords)

# move particles
particle_coords[:, 1] += torch.sin(particle_coords[:, 0] * 2 * torch.pi) / 2

start = time()
P2G(
    particle_values.contiguous(), 
    particle_coords.contiguous(), 
    grid_values.contiguous(),
    numerator.contiguous(),
    denominator.contiguous(),
)
end = time()

print(f"Total Time: {end - start} [s]")

plt.figure()
plt.imshow(grid_values)
plt.show()