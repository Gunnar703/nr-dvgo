from typing import Any, Callable
import numpy as np
import imageio
import os

def render_test(phase: str, basedir: str, expname: str, stepsize: float, white_bkgd: bool, inverse_y: bool, flip_x: bool, flip_y: bool, data_dict: dict[str, Any], load_model: Callable):
    ckpt_path = os.path.join(basedir, expname, f"last_{phase}.tar")
    ckpt_name = os.path.split(ckpt_path)[1][:-4]
    model = load_model(ckpt_path)
    render_kwargs=dict(
        near=data_dict["near"],
        far=data_dict["far"],
        bg=int(white_bkgd),
        stepsize=stepsize,
        inverse_y=inverse_y
    )

    testsavedir = os.path.join(basedir, expname, f"render_test_{ckpt_name}")
    os.makedirs(testsavedir, exist_ok=True)

    rgbs, disps = model.render_viewpoints(
        data_dict["poses"],
        HW=data_dict["HW"],
        Ks=data_dict["Ks"],
        render_kwargs=render_kwargs,
        flip_x=flip_x,
        flip_y=flip_y
    )

    imageio.mimwrite(
        os.path.join(testsavedir, "video_rgb.mp4"),
        (255 * np.clip(rgbs, 0, 1)).astype(np.uint8),
        fps=30,
        quality=8
    )

    imageio.mimwrite(
        os.path.join(testsavedir, "video_disp.mp4"),
        (255 * np.clip(disps/disps.max(), 0, 1)).astype(np.uint8),
        fps=30,
        quality=8
    )