from dynamic_dvgo.dynamic_observer import DynamicObserver
from dynamic_dvgo.static_dvgo import checkpoint_utils
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
        scale_factor_xyz=(3, 3, 1),
        n_particles=1
    )
    print("[main] Done initializing dynamic observer")

    def deformation_func(xyz: torch.Tensor):
        x, y, z = xyz[:, 0:1], xyz[:, 1:2], xyz[:, 2:3]

        x = x
        y = y + torch.sin(z)
        z = z
        return torch.cat((x, y, z), 1)

    print("[main] Starting grid deformation.")
    start = time.time()
    dynamic_observer.deform_grid(deformation_function=deformation_func)
    print(f"[main] Grid deformation finished in {time.time() - start} [s].")
    
