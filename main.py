import imageio
from lib.load_data import load_data
from lib import dvgo, dynamic_dvgo, utils
from mmengine import Config
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import warnings
import torch
import os

warnings.filterwarnings("ignore")


def load_everything(cfg, dynamic: bool = False):
    """Load images / poses / camera settings / data split."""
    data_dict = load_data(cfg.data)

    # remove useless field
    kept_keys = {
        "hwf",
        "HW",
        "Ks",
        "near",
        "far",
        "i_train",
        "i_val",
        "i_test",
        "irregular_shape",
        "poses",
        "render_poses",
        "images",
    }
    if dynamic:
        kept_keys.add("timestep_nums")
    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)

    # construct data tensor
    if data_dict["irregular_shape"]:
        data_dict["images"] = [
            torch.FloatTensor(im, device="cpu") for im in data_dict["images"]
        ]
    else:
        data_dict["images"] = torch.FloatTensor(data_dict["images"], device="cpu")
    data_dict["poses"] = torch.Tensor(data_dict["poses"])
    return data_dict


if __name__ == "__main__":
    cfg_file = "configs/custom/beam_dynamic.py"
    ckpt_path = "logs/beam/equilibrium/fine_last.tar"

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    ## IMPORT DATA
    cfg = Config.fromfile(cfg_file)

    data_dict = load_everything(cfg, True)
    data_dict["near"] = 0.2
    data_dict["far"] = 3

    idx_list = np.arange(len(data_dict["poses"]))

    fps = 120
    images = data_dict["images"]
    i_train = data_dict["i_train"]
    poses = data_dict["poses"]
    HW = data_dict["HW"]
    Ks = data_dict["Ks"]
    gt_imgs = [data_dict["images"][i] for i in idx_list]
    gt_imgs = torch.tensor(np.stack(gt_imgs, axis=0), device=device)
    times = data_dict["timestep_nums"] / fps
    times = torch.tensor(times, device=device)

    ## CREATE MODEL
    ray_bender = dynamic_dvgo.BeamBender(128, 5, torch.nn.Tanh, 1.0)
    ckpt = torch.load(ckpt_path)
    model = dynamic_dvgo.DynamicVoxelGrid(**ckpt["model_kwargs"])
    model.load_state_dict(ckpt["model_state_dict"])
    for p in model.parameters():
        p.requires_grad = False
    model.set_ray_bender(ray_bender)
    model.cuda()

    render_kwargs = {
        "near": data_dict["near"],
        "far": data_dict["far"],
        "bg": 1 if cfg.data.white_bkgd else 0,
        "stepsize": 0.5,
        "inverse_y": cfg.data.inverse_y,
        "flip_x": cfg.data.flip_x,
        "flip_y": cfg.data.flip_y,
    }

    ## GET TRAINING RAYS
    def gather_training_rays():
        rgb_tr_ori = images[i_train].to(
            "cpu" if cfg.data.load2gpu_on_the_fly else device
        )

        rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays(
            rgb_tr=rgb_tr_ori,
            train_poses=poses[i_train],
            HW=HW[i_train],
            Ks=Ks[i_train],
            ndc=cfg.data.ndc,
            inverse_y=cfg.data.inverse_y,
            flip_x=cfg.data.flip_x,
            flip_y=cfg.data.flip_y,
        )

        index_generator = dvgo.batch_indices_generator(
            len(rgb_tr), cfg.fine_train.N_rand
        )
        batch_index_sampler = lambda: next(index_generator)

        times_tr = times[i_train, None, None]
        times_tr = times_tr.expand(rays_o_tr.shape[:-1])

        return (
            rgb_tr,
            rays_o_tr,
            times_tr,
            rays_d_tr,
            viewdirs_tr,
            imsz,
            batch_index_sampler,
        )

    (
        rgb_tr,
        rays_o_tr,
        times_tr,
        rays_d_tr,
        viewdirs_tr,
        imsz,
        batch_index_sampler,
    ) = gather_training_rays()

    ## PERFORM TRAINING
    torch.cuda.empty_cache()

    optimizer = torch.optim.Adam(ray_bender.parameters(), lr=1e-5)

    num_epochs = 10_000
    progress_bar = tqdm(range(num_epochs), miniters=50)

    save_folder = os.path.join("logs", "beam", "dynamic")
    os.makedirs(save_folder, exist_ok=True)

    for step in progress_bar:
        # Sample rays
        sel_b = torch.randint(rgb_tr.shape[0], [cfg.fine_train.N_rand], device="cpu")
        sel_r = torch.randint(rgb_tr.shape[1], [cfg.fine_train.N_rand], device="cpu")
        sel_c = torch.randint(rgb_tr.shape[2], [cfg.fine_train.N_rand], device="cpu")
        target = rgb_tr[sel_b, sel_r, sel_c]
        rays_o = rays_o_tr[sel_b, sel_r, sel_c]
        rays_d = rays_d_tr[sel_b, sel_r, sel_c]
        viewdirs = viewdirs_tr[sel_b, sel_r, sel_c]
        times = times_tr[sel_b, sel_r, sel_c]

        if cfg.data.load2gpu_on_the_fly:
            target = target.to(device)
            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)
            viewdirs = viewdirs.to(device)
            times = times.to(device)

        # Render
        render_result = model(
            rays_o, rays_d, viewdirs, times, global_step=step, **render_kwargs
        )

        # Compute Loss / Backprop
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(render_result["rgb_marched"], target)
        psnr = utils.mse2psnr(loss.detach()).item()

        loss.backward()
        optimizer.step()

        if not step % 2:
            progress_bar.set_description(
                f"Epoch {step + 1}/{num_epochs} | Loss {loss: .4g} | PSNR {psnr}"
            )

    torch.save(model.state_dict(), os.path.join(save_folder, "ray_bender.pt"))

    @torch.no_grad()
    def render_viewpoints(
        model,
        render_poses,
        HW,
        Ks,
        ndc,
        render_kwargs,
        gt_imgs=None,
        savedir=None,
        render_factor=0,
        eval_ssim=False,
        eval_lpips_alex=False,
        eval_lpips_vgg=False,
    ):
        """Render images for the given viewpoints; run evaluation if gt given."""
        assert len(render_poses) == len(HW) and len(HW) == len(Ks)

        if render_factor != 0:
            HW = np.copy(HW)
            Ks = np.copy(Ks)
            HW //= render_factor
            Ks[:, :2, :3] //= render_factor

        rgbs = []
        disps = []
        psnrs = []
        ssims = []
        lpips_alex = []
        lpips_vgg = []

        for i, c2w in enumerate(tqdm(render_poses)):
            H, W = HW[i]
            K = Ks[i]
            t = times[i]
            rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H,
                W,
                K,
                c2w,
                ndc,
                inverse_y=render_kwargs["inverse_y"],
                flip_x=cfg.data.flip_x,
                flip_y=cfg.data.flip_y,
            )
            keys = ["rgb_marched", "disp"]
            render_result_chunks = [
                {k: v for k, v in model(ro, rd, vd, **render_kwargs).items() if k in keys}
                for ro, rd, vd in zip(
                    rays_o.split(16, 0), rays_d.split(16, 0), viewdirs.split(16, 0)
                )
            ]
            render_result = {
                k: torch.cat([ret[k] for ret in render_result_chunks])
                for k in render_result_chunks[0].keys()
            }
            rgb = render_result["rgb_marched"].cpu().numpy()
            disp = render_result["disp"].cpu().numpy()

            rgbs.append(rgb)
            disps.append(disp)
            if i == 0:
                print("Testing", rgb.shape, disp.shape)

            if gt_imgs is not None and render_factor == 0:
                p = -10.0 * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
                psnrs.append(p)
                if eval_ssim:
                    ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
                if eval_lpips_alex:
                    lpips_alex.append(
                        utils.rgb_lpips(rgb, gt_imgs[i], net_name="alex", device=c2w.device)
                    )
                if eval_lpips_vgg:
                    lpips_vgg.append(
                        utils.rgb_lpips(rgb, gt_imgs[i], net_name="vgg", device=c2w.device)
                    )

            if savedir is not None:
                rgb8 = utils.to8b(rgbs[-1])
                filename = os.path.join(savedir, "{:03d}.png".format(i))
                imageio.imwrite(filename, rgb8)

        rgbs = np.array(rgbs)
        disps = np.array(disps)
        if len(psnrs):
            """
            print('Testing psnr', [f'{p:.3f}' for p in psnrs])
            if eval_ssim: print('Testing ssim', [f'{p:.3f}' for p in ssims])
            if eval_lpips_vgg: print('Testing lpips (vgg)', [f'{p:.3f}' for p in lpips_vgg])
            if eval_lpips_alex: print('Testing lpips (alex)', [f'{p:.3f}' for p in lpips_alex])
            """
            print("Testing psnr", np.mean(psnrs), "(avg)")
            if eval_ssim:
                print("Testing ssim", np.mean(ssims), "(avg)")
            if eval_lpips_vgg:
                print("Testing lpips (vgg)", np.mean(lpips_vgg), "(avg)")
            if eval_lpips_alex:
                print("Testing lpips (alex)", np.mean(lpips_alex), "(avg)")

        return rgbs, disps
    
