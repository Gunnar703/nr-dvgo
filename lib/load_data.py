import numpy as np

from .load_blender import load_blender_data

def load_data(args):
    K, depths = None, None

    dynamic = False
    if args.dataset_type == "blender":
        blender_ret = load_blender_data(args.datadir, args.half_res, args.testskip)
        if len(blender_ret) == 6:
            dynamic = True
            images, poses, render_poses, hwf, i_split, timestep_nums = blender_ret
        else:
            images, poses, render_poses, hwf, i_split = blender_ret
        print("Loaded blender", images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near, far = 0.2, 2.5

        if images.shape[-1] == 4:
            if args.white_bkgd:
                images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
            else:
                images = images[..., :3] * images[..., -1:]

    else:
        raise NotImplementedError(f"Unknown dataset type {args.dataset_type} exiting")

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    HW = np.array([im.shape[:2] for im in images])
    irregular_shape = images.dtype is np.dtype("object")

    if K is None:
        K = np.array([[focal, 0, 0.5 * W], [0, focal, 0.5 * H], [0, 0, 1]])

    if len(K.shape) == 2:
        Ks = K[None].repeat(len(poses), axis=0)
    else:
        Ks = K

    render_poses = render_poses[..., :4]

    data_dict = dict(
        hwf=hwf,
        HW=HW,
        Ks=Ks,
        near=near,
        far=far,
        i_train=i_train,
        i_val=i_val,
        i_test=i_test,
        poses=poses,
        render_poses=render_poses,
        images=images,
        depths=depths,
        irregular_shape=irregular_shape,
    )

    if dynamic:
        data_dict.update({"timestep_nums": timestep_nums})

    return data_dict


def inward_nearfar_heuristic(cam_o, ratio=0.05):
    dist = np.linalg.norm(cam_o[:, None] - cam_o, axis=-1)
    far = dist.max()
    near = far * ratio
    return near, far
