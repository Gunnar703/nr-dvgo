from dynamic_dvgo.static_dvgo import checkpoint_utils, bbox_utils, dvgo
from dynamic_dvgo.dynamic_observer import DynamicObserver
from visualization.viz_cameras_3d import viz_cameras_3d
from visualization.viz_volume_3d import viz_geom_3d
from dataset.load_blender import load_data

import taichi as ti
import torch
import time
import os


if __name__ == "__main__":
    ti.init(ti.gpu)

    BASE_DIR = "results"
    EXPNAME = "equilibrium"

    DATA_BASE_DIR = "data/beam_equilibrium"
    NEAR_CLIP = 0.2
    FAR_CLIP = 3

    STEPSIZE = 0.5
    WHITE_BKGD = True
    INVERSE_Y = False
    FLIP_X = False
    FLIP_Y = False

    PHASE = "fine"

    data_dict = load_data(
        data_base_dir=DATA_BASE_DIR,
        near=NEAR_CLIP,
        far=FAR_CLIP,
    )

    ckpt_path = os.path.join(BASE_DIR, EXPNAME, f"last_{PHASE}.tar")
    model = checkpoint_utils.load_model(ckpt_path)
    model.cuda()

    render_kwargs=dict(
        near=data_dict["near"],
        far=data_dict["far"],
        bg=int(WHITE_BKGD),
        stepsize=STEPSIZE,
        inverse_y=INVERSE_Y
    )

    print("[main] Initializing dynamic observer")
    dynamic_observer = DynamicObserver(
        base_grid=model,
        scale_factor_xyz=(2, 2, 1),
        n_particles=8
    )
    print("[main] Done initializing dynamic observer")

    def deformation_func(zyx: torch.Tensor):
        z, y, x = zyx[:, 0:1], zyx[:, 1:2], zyx[:, 2:3]

        x = x
        y = y + torch.sin(z) / 2
        z = z
        return torch.cat((z, y, x), 1)

    print("[main] Starting grid deformation.")
    start = time.time()
    dynamic_observer.deform_grid(deformation_function=deformation_func)
    print(f"[main] Grid deformation finished in {time.time() - start} [s].")

    cam_list = viz_cameras_3d(
        dynamic_observer.base_grid[0].xyz_min,
        dynamic_observer.base_grid[0].xyz_max,
        data_dict,
        INVERSE_Y,
        FLIP_X,
        FLIP_Y,
        dvgo.get_rays_of_a_view
    )

    viz_geom_3d(
        model=dynamic_observer.base_grid[0],
        cam_list=cam_list,
    )