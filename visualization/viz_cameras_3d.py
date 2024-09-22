import torch
import numpy as np
import open3d as o3d

from typing import Any, Callable

def viz_cameras_3d(xyz_min: torch.Tensor, xyz_max: torch.Tensor, data_dict: dict[str, Any], inverse_y: bool, flip_x: bool, flip_y: bool, get_rays_of_a_view: Callable):

    poses, HW, Ks = data_dict["poses"], data_dict["HW"], data_dict["Ks"]
    near, far = data_dict["near"], data_dict["far"]

    cam_list = []
    for c2w, (H, W), K in zip(poses, HW, Ks):
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
            H=H,
            W=W,
            K=K,
            c2w=c2w,
            inverse_y=inverse_y,
            flip_x=flip_x,
            flip_y=flip_y,
        )
    
        cam_o = rays_o[0, 0].cpu().numpy()
        cam_d = rays_d[[0, 0, -1, -1], [0, -1, 0, -1]].cpu().numpy()
        cam_list.append(np.array([cam_o, *(cam_o + cam_d * max(near, far * 0.05))]))

    xyz_min = xyz_min.cpu().numpy()
    xyz_max = xyz_max.cpu().numpy()
    cam_list = np.array(cam_list)

    # Outer aabb
    aabb_01 = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 1],
            [1, 1, 0],
        ]
    )

    out_bbox = o3d.geometry.LineSet()
    out_bbox.points = o3d.utility.Vector3dVector(xyz_min + aabb_01 * (xyz_max - xyz_min))
    out_bbox.colors = o3d.utility.Vector3dVector([[1, 0, 0] for i in range(12)])
    out_bbox.lines = o3d.utility.Vector2iVector(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]
    )

    # Cameras
    cam_frustrm_lst = []
    for cam in cam_list:
        cam_frustrm = o3d.geometry.LineSet()
        cam_frustrm.points = o3d.utility.Vector3dVector(cam)
    
        if len(cam) == 5:
            cam_frustrm.colors = o3d.utility.Vector3dVector([[0, 0, 0] for i in range(8)])
            cam_frustrm.lines = o3d.utility.Vector2iVector(
                [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 4], [4, 3], [3, 1]]
            )
        elif len(cam) == 8:
            cam_frustrm.colors = o3d.utility.Vector3dVector([[0, 0, 0] for i in range(12)])
            cam_frustrm.lines = o3d.utility.Vector2iVector(
                [
                    [0, 1],
                    [1, 3],
                    [3, 2],
                    [2, 0],
                    [4, 5],
                    [5, 7],
                    [7, 6],
                    [6, 4],
                    [0, 4],
                    [1, 5],
                    [3, 7],
                    [2, 6],
                ]
            )
        cam_frustrm_lst.append(cam_frustrm)

    o3d.visualization.draw_geometries([
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=xyz_min),
        out_bbox,
        *cam_frustrm_lst,
    ])
    return cam_list