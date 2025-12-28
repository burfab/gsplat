import argparse
import math
import time

import os
import json
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import viser
from pathlib import Path
from gsplat.distributed import cli
from gsplat.rendering import rasterization

from nerfview import CameraState, RenderTabState, apply_float_colormap
from gsplat_viewer_2dgs import GsplatViewer, GsplatRenderTabState

from datasets.colmap import Dataset, Parser

from tqdm import tqdm

def main(local_rank: int, world_rank, world_size: int, args):
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(42)
    device = torch.device("cuda")

    means, quats, scales, opacities, sh0, shN = [], [], [], [], [], []
    for ckpt_path in args.ckpt:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)[
            "splats"
        ]
        means.append(ckpt["means"])
        quats.append(F.normalize(ckpt["quats"], p=2, dim=-1))
        scales.append(torch.exp(ckpt["scales"]))
        opacities.append(torch.sigmoid(ckpt["opacities"]))
        sh0.append(ckpt["sh0"])
        shN.append(ckpt["shN"])
    means = torch.cat(means, dim=0)
    quats = torch.cat(quats, dim=0)
    scales = torch.cat(scales, dim=0)
    opacities = torch.cat(opacities, dim=0)
    sh0 = torch.cat(sh0, dim=0)
    shN = torch.cat(shN, dim=0)
    colors = torch.cat([sh0, shN], dim=-2)
    sh_degree = int(math.sqrt(colors.shape[-2]) - 1)
    print("Number of Gaussians:", len(means))

    # register and open viewer
    @torch.no_grad()
    def render_fn(c2w, K, width, height, near_plane = 0.01, far_plane = 1_000):
        viewmat = c2w.inverse()

        render_colors, render_alphas, info = rasterization(
            means,  # [N, 3]
            quats,  # [N, 4]
            scales,  # [N, 3]
            opacities,  # [N]
            colors,  # [N, S, 3]
            viewmat[None],  # [1, 4, 4]
            K[None],  # [1, 3, 3]
            width,
            height,
            sh_degree=sh_degree,
            near_plane=near_plane,
            far_plane=far_plane,
            radius_clip=args.radius_clip,
            eps2d=args.eps2d,
            render_mode="RGB+D",
            rasterize_mode="antialiased" if args.antialiased else "classic",
            camera_model="pinhole",
            packed=False,
            with_ut=args.with_3dgut,
            with_eval3d=args.with_3dgut,
        )
        render_median = render_colors[...,3].contiguous()
        return render_colors[...,:3].squeeze(0), render_alphas.squeeze(0), render_median.squeeze(0)

    def maskedPSNR(x,y,mask, max_ = 1.0):
        x = x[...,:3]
        y = y[...,:3]
        if mask is None:
            mask = torch.ones_like(x).float()
        else:
            if mask.dim() == 2: mask = mask.unsqueeze(-1)
            mask = mask.expand(-1,-1,x.shape[-1])
        l2 = ((x-y)*mask)**2
        mse = l2.sum()/(mask.sum())
        psnr = 10 * torch.log10((max_**2)/mse)
        return psnr
    
    
    def write_image(img, path):
        if img.dim() == 2: img = img.unsqueeze(-1)
        assert img.dim() == 3
        if img.dtype != torch.uint8: img = (img.detach() * 255).to(torch.uint8)
        img_np = img.detach().cpu().numpy()
        if img_np.shape[-1] == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, img_np)

    # Load data: Training data should contain initial points and colors.
    parser = Parser(
        data_dir=args.data_dir,
        normalize=args.normalize_world_space,
        test_every=args.test_every,
    )

    dataset = Dataset(
        parser,
        split=args.dataset_split,
        load_depths=False,
    )
    scene_scale = parser.scene_scale * 1.1 * args.global_scale
    print("Scene scale:", scene_scale)
    baselines = [s * scene_scale for s in args.baselines_relative]

    def get_augmented_c2w(c2w, baseline):
        c2w = c2w.clone()
        R = c2w[:3,:3]
        tx = torch.tensor([baseline, 0,0]).to(c2w.dtype).to(c2w.device)
        tx_cam = R @ tx
        c2w[:3,3] = c2w[:3,3] + tx_cam
        return c2w
    
    def get_moved_back_c2w(c2w, t):
        c2w = c2w.clone()
        R = c2w[:3,:3]
        tz = torch.tensor([0, 0, t]).to(c2w.dtype).to(c2w.device)
        tz_cam = R @ tz
        c2w[:3,3] = c2w[:3,3] + tz_cam
        return c2w
    

    render_path = os.path.join(args.output_dir, args.dataset_split, "renders")
    gts_path = os.path.join(args.output_dir, args.dataset_split, "gt")
    info_file = os.path.join(args.output_dir, args.dataset_split, "info.json")

    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)
    cam_infos = []
    for i in tqdm(range(0,len(dataset), args.render_step)):
        image_id = dataset.indices[i]
        data = dataset[i]

        camtoworld = data["camtoworld"].to(device)  # [4, 4]
        K = data["K"].to(device)  # [3, 3]
        gt_pixels = data["image"].to(device).squeeze() / 255.0  # [H, W, 3]
        fgmask = data["fgmask"].to(device).squeeze()
        if "mask" in data:
            mask = fgmask * data["mask"][...,None].to(device).bool()
        else: mask = fgmask
        height, width = gt_pixels.shape[0:2]

        #camtoworld = get_moved_back_c2w(camtoworld, -scene_scale * 0.3)
        #assert len(baselines) == 1, "Want to move a bit right, a bit left"
        #camtoworld = get_augmented_c2w(camtoworld, -baselines[0]*0.5)
        w2c = camtoworld.inverse()
        cam_info = {
            "uid":f"{image_id:07d}",
            "K": K.cpu().numpy().tolist(),
            "R": w2c.cpu().numpy()[:3,:3].tolist(),
            "T": w2c.cpu().numpy()[:3,3].tolist(),
            "baselines": baselines
        }
        name = cam_info["uid"]


        write_image(gt_pixels, os.path.join(gts_path, f"{name}.png"))
        write_image(mask, os.path.join(gts_path, f'{name}_mask' + ".png"))


        render_colors, render_alphas, render_median = render_fn(camtoworld, K, width, height)
        cam_info["psnr"] = maskedPSNR(render_colors, gt_pixels, mask).item()
        write_image(render_colors[...,:3], os.path.join(render_path, f'{name}_left' + ".png"))

        for idx_baseline, baseline in enumerate(baselines):
            c2w_right = get_augmented_c2w(camtoworld, baseline)
            render_colors, render_alphas, render_median = render_fn(c2w_right, K, width, height)
            write_image(render_colors[...,:3], os.path.join(render_path, f"{name}_right_b{idx_baseline:02d}.png"))
        
        cam_infos.append(cam_info)

    with open(info_file, "w+") as f:
        json.dump({"views": cam_infos, "scene_scale": scene_scale}, f, indent=4)




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--normalize_world_space", type=bool, default=True, help="Normalize the world space"
    )
    parser.add_argument(
        "--test_every", type=int, default=8, help="Test every nth frame"
    )
    parser.add_argument(
        "--dataset_split", type=str, default="train", help="Which dataset split to use. train/val"
    )
    parser.add_argument(
        "--global_scale", type=float, default=1.0, help="Scene scale modifier"
    )
    parser.add_argument(
        "--radius_clip", type=float, default=0.0, help="Radius clip"
    )
    parser.add_argument(
        "--baselines_relative", type=list, default=[0.07], help="Baseline relative to scene scale"
    )
    parser.add_argument(
        "--data_dir", type=str, default=".", help="dataset input directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default="stereo/", help="where to dump outputs"
    )
    parser.add_argument(
        "--render_step", type=int, default=1, help="Step when going through render set"
    )
    parser.add_argument(
        "--with_3dgut", type=bool, default=False, help="Use 3d gut"
    )
    parser.add_argument(
        "--eps2d", type=float, default=0.3, help="Eps 2d to add to covar"
    )
    parser.add_argument(
        "--antialiased", type=bool, default=0.3, help="Antialiased"
    )
    parser.add_argument(
        "--scene_grid", type=int, default=1, help="repeat the scene into a grid of NxN"
    )
    parser.add_argument(
        "--ckpt", type=str, nargs="+", default=None, help="path to the .pt file"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="port for the viewer server"
    )

    args = parser.parse_args()
    assert args.scene_grid % 2 == 1, "scene_grid must be odd"
    

    cli(main, args, verbose=True)
