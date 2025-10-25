
import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import ctypes

import fused_ssim
from torch.amp import autocast
import fused_ssim_cuda

from multiprocessing import Value as SMValue
import cv2

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from datasets.colmap import Dataset, Parser
from datasets.traj import (
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
)
from fused_ssim import fused_ssim
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed

import splats

from gsplat import export_splats
from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.optimizers import SelectiveAdam
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat_viewer import GsplatViewer, GsplatRenderTabState
from nerfview import CameraState, RenderTabState, apply_float_colormap


@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path
    render_traj_path: str = "interp"

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # increase data factor at these iterations until we are at 1
    data_factor_up_steps = [5000, 15_000]
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 32_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Whether to save ply file (storage size can be large)
    save_ply: bool = True
    # Steps to save the model as ply
    ply_steps: List[int] = field(default_factory=lambda: [7000, 10_000, 14_000, 21_000, 30_000])
    # Whether to disable video generation during training and evaluation
    disable_video: bool = True

    # Initialization strategy
    init_type: str = "sfm"
    #Mesh path for mesh initialization
    mesh_name: str = None
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = True

    # Use random background for training to discourage transparency
    random_bkgd: bool = True

    # LR for 3D point positions
    means_lr: float = 1.6e-4
    # LR for Gaussian scale factors
    scales_lr: float = 5e-3
    # LR for alpha blending weights
    opacities_lr: float = 5e-2
    # LR for orientation (quaternions)
    quats_lr: float = 1e-3
    # LR for SH band 0 (brightness)
    sh0_lr: float = 2.5e-3
    # LR for higher-order SH (detail)
    shN_lr: float = 2.5e-3 / 20

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Enable camera optimization.
    pose_opt: bool = True
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable bilateral grid. (experimental)
    use_bilateral_grid: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"

    # 3DGUT (uncented transform + eval 3D)
    with_ut: bool = False
    with_eval3d: bool = False

    # Whether use fused-bilateral grid
    use_fused_bilagrid: bool = False

    def adjust_steps(self, factor: float, strategies):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        for strategy in strategies:
            if strategy is None: continue
            if isinstance(strategy, DefaultStrategy):
                strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
                strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
                strategy.reset_every = int(strategy.reset_every * factor)
                strategy.refine_every = int(strategy.refine_every * factor)
            elif isinstance(strategy, MCMCStrategy):
                strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
                strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
                strategy.refine_every = int(strategy.refine_every * factor)
            else:
                assert_never(strategy)

import open3d as o3d

def create_splats(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
    mesh_path = None
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "mesh":
        if mesh_path is None or len(mesh_path) == 0: assert False, "Mesh path may not be None or empty"
        mesh_o3d = o3d.io.read_triangle_mesh(mesh_path)
        return splats.create_surface_splats_from_mesh(mesh_o3d, 2, sh_degree, init_opacity, init_scale, device)
    elif init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]
    

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    return splats.create_free_splats_from_points(points, rgbs, sh_degree, init_opacity, init_scale, feature_dim, device)


class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.ply_dir = f"{cfg.result_dir}/ply"
        os.makedirs(self.ply_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        self.shared_factor = SMValue(ctypes.c_int, cfg.data_factor)


        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=self.shared_factor.value,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
        )
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
            shared_factor=self.shared_factor
        )
        self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Model
        feature_dim = 32 if cfg.app_opt else None
        self.splats_dict = {}
        self.optimizers_dict = {}
        """
        self.splats_dict["free"] = create_splats(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
        )
        """
        
        self.splats_dict["mesh"] = create_splats(
            self.parser,
            init_type="mesh",
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
            mesh_path=None if cfg.mesh_name is None else os.path.join(cfg.data_dir,cfg.mesh_name)
        )
        
        
        if "mesh" in self.splats_dict:
            self.optimizers_dict["mesh"] = self.splats_dict["mesh"].create_optimizers(means_lr=cfg.means_lr,
                                                                            scales_lr=cfg.scales_lr, opacities_lr=cfg.opacities_lr,
                                                                            quats_lr=cfg.quats_lr, sh0_lr=cfg.sh0_lr, shN_lr=cfg.shN_lr,
                                                                            batch_size= cfg.batch_size, world_size= world_size, 
                                                                            sparse_grad=cfg.sparse_grad, visible_adam=cfg.visible_adam)
        
        
        
        if "free" in self.splats_dict:
            self.optimizers_dict["free"] = self.splats_dict["free"].create_optimizers(means_lr=cfg.means_lr * self.scene_scale, 
                                                                            scales_lr=cfg.scales_lr, opacities_lr=cfg.opacities_lr,
                                                                            quats_lr=cfg.quats_lr, sh0_lr=cfg.sh0_lr, shN_lr=cfg.shN_lr,
                                                                            batch_size= cfg.batch_size, world_size= world_size, 
                                                                            sparse_grad=cfg.sparse_grad, visible_adam=cfg.visible_adam)
        
        self.strategies_dict = {
            "free": {"strategy": MCMCStrategy(), "state": None},
            "mesh": {"strategy": None, "state": None}
        }
        
        
        print("Model initialized. Number of splats: ", len(self.splats_dict))
        for k in self.splats_dict:
            print(f"\tSplat: {k}  |  Number Of GS: ", self.splats_dict[k].npoints())
            assert k in self.optimizers_dict, "Not all splats have optimizers"
            assert k in self.strategies_dict, "Not all splats have strategies"
            
            # Densification Strategy
            strategy = self.strategies_dict[k]["strategy"]
            strategy_state = None
            if not (strategy is None):
                strategy.check_sanity(self.splats_dict[k].params_dict, self.optimizers_dict[k])
                
                if isinstance(strategy, DefaultStrategy):
                    strategy_state = strategy.initialize_state(
                        scene_scale=self.scene_scale
                    )
                elif isinstance(strategy, MCMCStrategy):
                    strategy.cap_max = 100_000
                    strategy_state = strategy.initialize_state()
                else:
                    assert_never(strategy)
                
            self.strategies_dict[k]["state"] = strategy_state
            
        
        cfg.adjust_steps(cfg.steps_scaler ,[s["strategy"] for s in self.strategies_dict.values()])
        
        

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        self.bil_grid_optimizers = []
        if cfg.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
            self.bil_grid_optimizers = [
                torch.optim.Adam(
                    self.bil_grids.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]

        # Losses & Metrics.
        def maskedPSNR(x,y,mask, max_ = 1.0):
            if mask is None:
                mask = torch.ones_like(x).float()
            l2 = ((x-y)*mask.permute(0,3,1,2).repeat(1,3,1,1))**2
            mse = l2.mean()
            psnr = 10 * torch.log10((max_**2)/mse)
            return psnr

        # Losses & Metrics.
        self.psnr = maskedPSNR

        def sobel_filter(img):
            # Convert to (B, C, H, W)
            img = img.permute(0, 3, 1, 2)  # NHWC -> NCHW
            B, C, H, W = img.shape
            device = img.device
            dtype = img.dtype

            # Define Sobel kernels
            sobel_x = torch.tensor([[-1, 0, 1],
                                    [-2, 0, 2],
                                    [-1, 0, 1]], dtype=dtype, device=device) / 8.0

            sobel_y = torch.tensor([[-1, -2, -1],
                                    [ 0,  0,  0],
                                    [ 1,  2,  1]], dtype=dtype, device=device) / 8.0

            # Reshape to (1, 1, 3, 3)
            sobel_x = sobel_x.view(1, 1, 3, 3)
            sobel_y = sobel_y.view(1, 1, 3, 3)

            # Expand to (C, 1, 3, 3) for depthwise conv
            sobel_x = sobel_x.expand(C, 1, 3, 3)
            sobel_y = sobel_y.expand(C, 1, 3, 3)

            # Depthwise convolution
            grad_x = F.conv2d(img, sobel_x, padding=1, groups=C)
            grad_y = F.conv2d(img, sobel_y, padding=1, groups=C)

            # Return in original NHWC format
            return grad_x.permute(0, 2, 3, 1), grad_y.permute(0, 2, 3, 1)


        def tv_loss_sobel(img, norm='l1',epsilon=1e-3, reduction='mean'):
            dx, dy = sobel_filter(img)
            if norm == "l1": tv = torch.abs(dx) + torch.abs(dy)
            elif norm == "l2": tv = (dx * dx + dy * dy)
            else: assert False, "Unknown norm"

            if reduction == 'mean':
                return tv.mean()
            elif reduction == 'sum':
                return tv.sum()
            else:
                return tv
            
        self.tvloss = tv_loss_sobel


        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = GsplatViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                output_dir=Path(cfg.result_dir),
                mode="training",
            )

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        rasterize_mode: Optional[Literal["classic", "antialiased"]] = None,
        camera_model: Optional[Literal["pinhole", "ortho", "fisheye"]] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = None
        quats = None
        scales = None
        opacities = None
        colors = None
        features = None
        
        dicts = [self.splats_dict[k].as_renderer_dict() for k in self.splats_dict]
        means = torch.cat([d["means"] for d in dicts])
        quats = torch.cat([d["quats"] for d in dicts])
        scales = torch.cat([d["scales"] for d in dicts])
        opacities = torch.cat([d["opacities"] for d in dicts])
            
        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = torch.cat([d["colors"] for d in dicts])
            features = torch.cat(d["features"] for d in dicts)
            colors = self.app_module(
                features=features,
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + colors
            colors = torch.sigmoid(colors)
        else:
            sh0 = torch.cat([d["sh0"] for d in dicts])
            shN = torch.cat([d["shN"] for d in dicts])
            colors = torch.cat([sh0, shN], 1)  # [N, K, 3]

        if rasterize_mode is None:
            rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        if camera_model is None:
            camera_model = self.cfg.camera_model
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats = torch.linalg.inv(camtoworlds.to(dtype=torch.float32)).to(dtype=means.dtype),
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=False,
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            with_ut=self.cfg.with_ut,
            with_eval3d=self.cfg.with_eval3d,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ) for optimizers in self.optimizers_dict.values()
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        if cfg.use_bilateral_grid:
            # bilateral grid has a learning rate schedule. Linear warmup for 1000 steps.
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.bil_grid_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.bil_grid_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                        ),
                    ]
                )
            )

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            while True:
                expected_factor = self.shared_factor.value
                try:
                    data = next(trainloader_iter)
                except StopIteration:
                    trainloader_iter = iter(trainloader)
                    data = next(trainloader_iter)
                all_factors_good = True
                for f in (data["factor"]): 
                    if (f != expected_factor):all_factors_good = False 
                if all_factors_good: break
            

            camtoworlds_gt = data["camtoworld"].to(device)
            camtoworlds    = camtoworlds_gt.clone()

            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            fgmask = data["fgmask"].to(device).float()
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device).float() if "mask" in data else None  # [1, H, W]

            total_mask = (masks * fgmask) if masks is not None else fgmask

            
            valid_px = total_mask.sum()
            all_px   = total_mask.numel()
            loss_scale_factor = all_px / (valid_px + 1e-8)


            if cfg.depth_loss:
                points = data["points"].to(device)  # [1, M, 2]
                depths_gt = data["depths"].to(device)  # [1, M]

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # forward
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                masks=masks,
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            if cfg.use_bilateral_grid:
                grid_y, grid_x = torch.meshgrid(
                    (torch.arange(height, device=self.device) + 0.5) / height,
                    (torch.arange(width, device=self.device) + 0.5) / width,
                    indexing="ij",
                )
                grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                colors = slice(
                    self.bil_grids,
                    grid_xy.expand(colors.shape[0], -1, -1, -1),
                    colors,
                    image_ids.unsqueeze(-1),
                )["rgb"]

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            for splats_key in self.splats_dict:
                if self.strategies_dict[splats_key]["strategy"] is None: continue
                self.strategies_dict[splats_key]["strategy"].step_pre_backward(
                    params=self.splats_dict[splats_key].params_dict,
                    optimizers=self.optimizers_dict[splats_key],
                    state=self.strategies_dict[splats_key]["state"],
                    step=step,
                    info=info,
                )
            
            #pixels = pixels * total_mask
            #colors = colors * total_mask

            # compute photometric loss only on foreground
            # === photometric on foreground only ===
            l1loss   = F.l1_loss(colors[total_mask.bool().expand_as(colors)], 
                                pixels[total_mask.bool().expand_as(pixels)])
            ssimloss = (1.0 - fused_ssim(
                (colors*total_mask).permute(0,3,1,2), 
                (pixels*total_mask).permute(0,3,1,2)
            )) * loss_scale_factor



            def lamda_tv_t(lamda0, lamda1, step,max_steps, first_step, last_step = None):
                if last_step is not None and last_step < step: return 0
                if first_step > step: return 0
                exp_scale_factor = (math.log(lamda1)-math.log(lamda0))/(max_steps-first_step)
                step = min(step, max_steps)
                return lamda0 * math.exp(exp_scale_factor * (step-first_step))

            bg = 1.0 - fgmask
            dfloss_alpha = ((alphas - fgmask)**2 * bg).sum()
            tvloss_alpha = self.tvloss(alphas * bg, "l2", reduction=None).sum(dim=-1, keepdim=True).mean()
            den_bg = (bg.numel() - bg.sum()).clamp(min=1)
            alpha_reg = lamda_tv_t(1, 1e-1, step, max_steps, 500) * (dfloss_alpha + 100*tvloss_alpha) / den_bg

            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda + alpha_reg


            #loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda + lamda_tv_t(1, 1e-1, step, max_steps, 500) * (dfloss_alpha + 100 * tvloss_alpha) / (np.prod(fgmask.shape) - fgmask.sum())
            if cfg.depth_loss:
                # sample depths at points
                pts = torch.stack([points[:,:,0]/(width-1)*2-1, points[:,:,1]/(height-1)*2-1], dim=-1)  # (B,M,2)
                grid = pts.unsqueeze(2)   # (B,M,1,2)
                sampled = F.grid_sample(depths.permute(0,3,1,2), grid, align_corners=True)  # (B,1,M,1)
                depths_s = sampled.squeeze(3).squeeze(1)  # (B,M)
                valid_d  = depths_s > 0
                disp     = torch.zeros_like(depths_s)
                disp[valid_d] = 1.0 / depths_s[valid_d]
                disp_gt  = 1.0 / depths_gt
                depthloss = F.l1_loss(disp[valid_d], disp_gt[valid_d]) * self.scene_scale
                loss += depthloss * cfg.depth_lambda
            if cfg.use_bilateral_grid:
                tvloss = 10 * total_variation_loss(self.bil_grids.grids)
                loss += tvloss

            # regularizations
            if cfg.opacity_reg > 0.0:
                for k in self.splats_dict:
                    loss = (
                        loss
                        + cfg.opacity_reg
                        * torch.abs(self.splats_dict[k].world_opacities()).mean()
                    )
            if cfg.scale_reg > 0.0:
                for k in self.splats_dict:
                    loss = (
                        loss
                        + cfg.scale_reg * torch.abs(self.splats_dict[k].world_scales()).mean()
                    )

            loss.backward()

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            # write images (gt and render)
            # if world_rank == 0 and step % 800 == 0:
            #     canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
            #     canvas = canvas.reshape(-1, *canvas.shape[2:])
            #     imageio.imwrite(
            #         f"{self.render_dir}/train_rank{self.world_rank}.png",
            #         (canvas * 255).astype(np.uint8),
            #     )

            num_GS=int(np.sum([self.splats_dict[k].npoints() for k in self.splats_dict]))
            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar(f"train/num_GS", num_GS, step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.use_bilateral_grid:
                    self.writer.add_scalar("train/tvloss", tvloss.item(), step)
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            # save checkpoint before updating the model
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": num_GS,
                }
                    
                print("Step: ", step, stats)
                with open(
                    f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                data = {"step": step, "splats": {}}
                for k in self.splats_dict:
                    data["splats"] = {k: {"state_dict": self.splats_dict[k].state_dict(), "type":self.splats_dict[k].type_str()}}
                if cfg.pose_opt:
                    if world_size > 1:
                        data["pose_adjust"] = self.pose_adjust.module.state_dict()
                    else:
                        data["pose_adjust"] = self.pose_adjust.state_dict()
                if cfg.app_opt:
                    if world_size > 1:
                        data["app_module"] = self.app_module.module.state_dict()
                    else:
                        data["app_module"] = self.app_module.state_dict()
                torch.save(
                    data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )
            if (
                step in [i - 1 for i in cfg.ply_steps] or step == max_steps - 1
            ) and cfg.save_ply:
                
                means = None
                quats = None
                scales = None
                opacities = None
                colors = None
                features = None
                
                dicts = [self.splats_dict[k].as_renderer_dict() for k in self.splats_dict]
                means = torch.cat([d["means"] for d in dicts])
                quats = torch.cat([d["quats"] for d in dicts])
                scales = torch.cat([d["scales"] for d in dicts])
                opacities = torch.cat([d["opacities"] for d in dicts])
                    
                if self.cfg.app_opt:
                    colors = torch.cat([d["colors"] for d in dicts])
                    features = torch.cat(d["features"] for d in dicts)
                    rgb = self.app_module(
                        features=features,
                        embed_ids=image_ids,
                        dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                        sh_degree=sh_degree_to_use
                    )
                    rgb = rgb + colors
                    rgb = torch.sigmoid(rgb).squeeze(0).unsqueeze(1)
                    sh0 = rgb_to_sh(rgb)
                    shN = torch.empty([sh0.shape[0], 0, 3], device=sh0.device)
                else:
                    sh0 = torch.cat([d["sh0"] for d in dicts])
                    shN = torch.cat([d["shN"] for d in dicts])
                
                scales = torch.log(scales) - torch.log(1.0-scales)
                opacities = torch.log(opacities)
                export_splats(
                    means=means,
                    scales=scales,
                    quats=quats,
                    opacities=opacities,
                    sh0=sh0,
                    shN=shN,
                    format="ply",
                    save_to=f"{self.ply_dir}/point_cloud_{step}.ply",
                )

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                raise NotImplementedError("NOT IMPL")
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            if cfg.visible_adam:
                raise NotImplementedError("NOT IMPL")
                gaussian_cnt = self.splats.means.shape[0]
                if cfg.packed:
                    visibility_mask = torch.zeros_like(
                        self.splats["opacities"], dtype=bool
                    )
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).all(-1).any(0)

            # optimize
            for optimizers in self.optimizers_dict.values():
                for optimizer in optimizers.values():
                    if cfg.visible_adam:
                        raise NotImplementedError("NOT IMPL")
                        optimizer.step(visibility_mask)
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                for optimizer in self.pose_optimizers:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                for optimizer in self.app_optimizers:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                for optimizer in self.bil_grid_optimizers:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # Run post-backward steps after backward and optimizer
            for splats_key in self.splats_dict:
                optimizers = self.optimizers_dict[splats_key]
                splats = self.splats_dict[splats_key]
                strategy = self.strategies_dict[splats_key]["strategy"]
                if not (strategy is None): 
                    strategy_state= self.strategies_dict[splats_key]["state"]
                    if isinstance(strategy, DefaultStrategy):
                        strategy.step_post_backward(
                            params=splats.params_dict,
                            optimizers=optimizers,
                            state=strategy_state,
                            step=step,
                            info=info,
                            packed=cfg.packed,
                        )
                    elif isinstance(strategy, MCMCStrategy):
                        strategy.step_post_backward(
                            params=splats.params_dict,
                            optimizers=optimizers,
                            state=strategy_state,
                            step=step,
                            info=info,
                            lr=schedulers[0].get_last_lr()[0],
                        )
                    else:
                        assert_never(strategy)

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)
                self.render_traj(step)
            
            # run compression
            if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
                self.run_compression(step=step)
            
            if step in [i-1 for i in cfg.data_factor_up_steps]:
                self.shared_factor.value = max(self.shared_factor.value-1, 1)


            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.render_tab_state.num_train_rays_per_sec = (
                    num_train_rays_per_sec
                )
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = defaultdict(list)
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]

            fgmask = data["fgmask"].to(device)
            if "mask" in data:
                mask = fgmask * data["mask"][...,None].to(device).bool()
            else: mask = fgmask
            loss_scale_factor = 1/((mask.sum()) / np.prod(mask.shape) + 1e-8)


            torch.cuda.synchronize()
            tic = time.time()
            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=masks,
            )  # [1, H, W, 3]
            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-10)

            colors = torch.clamp(colors, 0.0, 1.0)
            canvas_list = [pixels, colors]

            if world_rank == 0:
                # write images
                canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
                canvas = (canvas * 255).astype(np.uint8)
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}.png",
                    canvas,
                )
                
                colors = colors * mask
                pixels = pixels * mask

                pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                metrics["psnr"].append(self.psnr(colors_p, pixels_p, fgmask))
                metrics["ssim"].append(fused_ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))
                if cfg.use_bilateral_grid:
                    cc_colors = color_correct(colors, pixels)
                    cc_colors_p = cc_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))
                    metrics["cc_ssim"].append(fused_ssim(cc_colors_p, pixels_p))
                    metrics["cc_lpips"].append(self.lpips(cc_colors_p, pixels_p))

        if world_rank == 0:
            ellipse_time /= len(valloader)

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            
            num_GS=int(np.sum([self.splats_dict[k].npoints() for k in self.splats_dict]))
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": num_GS,
                }
            )
            if cfg.use_bilateral_grid:
                print(
                    f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                    f"CC_PSNR: {stats['cc_psnr']:.3f}, CC_SSIM: {stats['cc_ssim']:.4f}, CC_LPIPS: {stats['cc_lpips']:.3f} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS: {stats['num_GS']}"
                )
            else:
                print(
                    f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS: {stats['num_GS']}"
                )
            # save stats as json
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            # save stats to tensorboard
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        if self.cfg.disable_video:
            return
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds_all = self.parser.camtoworlds[5:-5]
        if cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 1
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(
                camtoworlds_all, height=height
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
            )
        else:
            raise ValueError(
                f"Render trajectory type not supported: {cfg.render_traj_path}"
            )

        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
            depths = renders[..., 3:4]  # [1, H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())
            canvas_list = [colors, depths.repeat(1, 1, 1, 3)]

            # write images
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def run_compression(self, step: int):
        raise NotImplementedError("Not Implemented")
        """Entry for running compression."""
        print("Running compression...")
        world_rank = self.world_rank

        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.splats)

        # evaluate compression
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: CameraState, render_tab_state: RenderTabState
    ):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }

        render_colors, render_alphas, info = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=min(render_tab_state.max_sh_degree, self.cfg.sh_degree),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=self.device)
            / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
        )  # [1, H, W, 3]
        render_tab_state.total_gs_count = len(info["radii"])
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "rgb":
            # colors represented with sh are not guranteed to be in [0, 1]
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            # normalize depth to [0, 1]
            depth = render_colors[0, ..., 0:1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            if render_tab_state.inverse:
                alpha = 1 - alpha
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        return renders


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        step = ckpts[0]["step"]
        runner.eval(step=step)
        runner.render_traj(step=step)
        if cfg.compression is not None:
            runner.run_compression(step=step)
    else:
        runner.train()

    if not cfg.disable_viewer:
        runner.viewer.complete()
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=9 python -m examples.simple_trainer default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25

    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = { 
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
            ),)
               }
    cfg = tyro.extras.overridable_config_cli(configs)

    # Import BilateralGrid and related functions based on configuration
    if cfg.use_bilateral_grid or cfg.use_fused_bilagrid:
        if cfg.use_fused_bilagrid:
            cfg.use_bilateral_grid = True
            from fused_bilagrid import (
                BilateralGrid,
                color_correct,
                slice,
                total_variation_loss,
            )
        else:
            cfg.use_bilateral_grid = True
            from lib_bilagrid import (
                BilateralGrid,
                color_correct,
                slice,
                total_variation_loss,
            )

    # try import extra dependencies
    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )

    if cfg.with_ut:
        assert cfg.with_eval3d, "Training with UT requires setting `with_eval3d` flag."

    cli(main, cfg, verbose=True)
