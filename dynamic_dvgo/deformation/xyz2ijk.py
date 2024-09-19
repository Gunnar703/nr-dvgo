import torch

def coords_xyz2ijk(coords: torch.Tensor, grid_shape: "tuple[int, int, int]", xyz_min: torch.Tensor, xyz_max: torch.Tensor) -> torch.Tensor:
    """Normalize input coordinates to [0, MAX]. Also flips them so they represent (x, y, z) coordinates.

    Args:
        coords (torch.Tensor): A tensor of shape [N, 3] representing N 3-tuples of (z, y, x) coordinates.
        grid_shape (tuple[int, int, int]): A tuple of integers representing the size of the last 3 dimensions of the grid.
        xyz_min (torch.Tensor): A tensor of shape [1, 3] representing the lower bound of the bbox in each dimension.
        xyz_max (torch.Tensor): A tensor of shape [1, 3] representing the upper bound of the bbox in each dimension.

    Returns:
        torch.Tensor: Coordinates transformed to index coords.
    """

    coords = (
        (coords.to(xyz_min) - xyz_min)
        / (xyz_max - xyz_min)
    )
    coords = coords.flip((-1,))

    x_mul, y_mul, z_mul = grid_shape
    x, y, z = coords[:, 0:1], coords[:, 1:2], coords[:, 2:3]
    x, y, z = x * (x_mul - 1), y * (y_mul - 1), z * (z_mul - 1)
    coords = torch.cat((x, y, z), 1)
    return coords