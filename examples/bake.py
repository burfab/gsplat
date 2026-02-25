import tqdm
import numpy as np
import cv2
import torch
import open3d as o3d
import torch.nn.functional as F
import nvdiffrast.torch as dr
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Tuple, Optional
import copy
import os
import struct
import pyatlas

import open3d.visualization.rendering as headless_rendering


def srgb_to_linear(x):
    x = x.clamp(0,1)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(x):
    x = x.clamp(0,1)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * (x ** (1/2.4)) - 0.055) 

def _grid_from_uv(u: torch.Tensor, v: torch.Tensor, W: int, H: int) -> torch.Tensor:
    """u,v in pixel coords. returns grid (1,P,1,2) for grid_sample."""
    gx = (u / (W - 1)) * 2.0 - 1.0
    gy = (v / (H - 1)) * 2.0 - 1.0
    return torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)

def _sample_rgb_depth(im: torch.Tensor, depth: torch.Tensor, grid: torch.Tensor, depth_mode: str):
    """
    im: (H,W,3), depth: (H,W), grid: (1,P,1,2)
    returns col (P,3), zbuf (P,)
    """
    im_t = im.permute(2, 0, 1).unsqueeze(0)           # (1,3,H,W)
    depth_t = depth.unsqueeze(0).unsqueeze(0)         # (1,1,H,W)

    col = F.grid_sample(im_t, grid, mode="bilinear", align_corners=True)
    col = col.squeeze(0).squeeze(-1).T                # (P,3)

    zbuf = F.grid_sample(depth_t, grid, mode=depth_mode, align_corners=True)
    zbuf = zbuf.squeeze(0).squeeze(0).squeeze(-1)     # (P,)
    return col, zbuf

def fit_affine_rgb_wls(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, eps: float = 1e-8):
    """
    Weighted least squares per channel:
        y ≈ a*x + b   (RGB independently)
    x,y: (P,3), w: (P,1) or (P,)
    returns a,b: (3,), (3,)
    """
    if w.ndim == 1:
        w = w[:, None]
    w = w.clamp_min(0)

    S   = w.sum(dim=0)                               # (1,)
    Sx  = (w * x).sum(dim=0)                         # (3,)
    Sy  = (w * y).sum(dim=0)                         # (3,)
    Sxx = (w * x * x).sum(dim=0)                     # (3,)
    Sxy = (w * x * y).sum(dim=0)                     # (3,)

    den = (S * Sxx - Sx * Sx).clamp_min(eps)         # (3,)
    a = (S * Sxy - Sx * Sy) / den
    b = (Sy - a * Sx) / S.clamp_min(eps)
    return a, b

@torch.no_grad()
def compute_exposure_to_anchor_affine(
    Xw_flat: torch.Tensor,          # (P,3) world points for covered texels (already filtered by UV mask)
    Nw_flat: torch.Tensor,          # (P,3) normals at those points
    views,                          # list of dicts with K, c2w, im, depth_im
    cfg,
    dev: torch.device,
    min_corr_points: int = 4096,
    clamp_a=(0.5, 2.0),
    clamp_b=(-0.5, 0.5),
    anchor = 0
):
    """
    Returns a_view, b_view with shape (V,3).
    Applies vi -> anchor(0) mapping:  col_corr = col * a_view[vi] + b_view[vi]
    Anchor is identity (a=[1,1,1], b=[0,0,0]).
    """
    Vn = len(views)
    a_view = torch.ones((Vn, 3), device=dev, dtype=torch.float32)
    b_view = torch.zeros((Vn, 3), device=dev, dtype=torch.float32)

    # ---- anchor ----
    v0 = views[anchor]
    K0 = torch.as_tensor(v0["K"], device=dev, dtype=torch.float32)
    c2w0 = torch.as_tensor(v0["c2w"], device=dev, dtype=torch.float32)
    w2c0 = torch.linalg.inv(c2w0)
    Cw0 = c2w0[:3, 3]

    im0 = torch.as_tensor(v0["im"], device=dev, dtype=torch.float32)
    im0 = srgb_to_linear(im0)
    
    
    dep0 = torch.as_tensor(v0["depth_im"], device=dev, dtype=torch.float32)
    H0, W0 = im0.shape[0], im0.shape[1]

    # project all points into anchor
    Xc0 = (torch.cat([Xw_flat, torch.ones((Xw_flat.shape[0], 1), device=dev)], dim=1) @ w2c0.T)[:, :3]
    z0 = Xc0[:, 2]
    u0 = K0[0, 0] * (Xc0[:, 0] / z0) + K0[0, 2]
    v0p = K0[1, 1] * (Xc0[:, 1] / z0) + K0[1, 2]

    ok0 = z0 > 1e-6
    b = int(cfg.clamp_border)
    ok0 = ok0 & (u0 >= b) & (u0 <= (W0 - 1 - b)) & (v0p >= b) & (v0p <= (H0 - 1 - b))
    if ok0.sum().item() < min_corr_points:
        return a_view, b_view  # not enough overlap, just return identity

    grid0 = _grid_from_uv(u0[ok0], v0p[ok0], W0, H0)
    col0, zbuf0 = _sample_rgb_depth(im0, dep0, grid0, cfg.depth_sample_mode)

    Xw0 = Xw_flat[ok0]
    Nw0 = Nw_flat[ok0]
    z0_ok = z0[ok0]

    if cfg.enable_depth_test:
        eps0 = cfg.depth_eps_abs + cfg.depth_eps_rel * z0_ok
        okd0 = torch.isfinite(zbuf0) & (torch.abs(zbuf0 - z0_ok) <= eps0)
        if okd0.sum().item() < min_corr_points:
            return a_view, b_view
        col0 = col0[okd0]; Xw0 = Xw0[okd0]; Nw0 = Nw0[okd0]; z0_ok = z0_ok[okd0]

    if cfg.reject_backfaces:
        Vdir0 = (Cw0.view(1, 3) - Xw0)
        Vdir0 = Vdir0 / (torch.norm(Vdir0, dim=1, keepdim=True) + 1e-12)
        ndv0 = torch.sum(Nw0 * Vdir0, dim=1)
        okf0 = ndv0 > 1e-6
        if okf0.sum().item() < min_corr_points:
            return a_view, b_view
        col0 = col0[okf0]; Xw0 = Xw0[okf0]; Nw0 = Nw0[okf0]; z0_ok = z0_ok[okf0]
        ndv0 = ndv0[okf0]
    else:
        ndv0 = torch.ones((col0.shape[0],), device=dev, dtype=torch.float32)

    # weights for anchor samples (used for fitting too)
    w0 = (torch.clamp(ndv0, 0.0, 1.0) ** float(cfg.weight_power)).view(-1, 1)
    if cfg.use_distance_weight:
        w0 = w0 / (z0_ok.view(-1, 1) * z0_ok.view(-1, 1) + 1e-12)

    # ---- fit each view to anchor ----
    for vi in range(1, Vn):
        v = views[vi]
        K = torch.as_tensor(v["K"], device=dev, dtype=torch.float32)
        c2w = torch.as_tensor(v["c2w"], device=dev, dtype=torch.float32)
        w2c = torch.linalg.inv(c2w)
        Cw = c2w[:3, 3]

        im = torch.as_tensor(v["im"], device=dev, dtype=torch.float32)
        im = srgb_to_linear(im)
        dep = torch.as_tensor(v["depth_im"], device=dev, dtype=torch.float32)
        H, W = im.shape[0], im.shape[1]

        Xc = (torch.cat([Xw0, torch.ones((Xw0.shape[0], 1), device=dev)], dim=1) @ w2c.T)[:, :3]
        z = Xc[:, 2]
        u = K[0, 0] * (Xc[:, 0] / z) + K[0, 2]
        vp = K[1, 1] * (Xc[:, 1] / z) + K[1, 2]

        ok = z > 1e-6
        ok = ok & (u >= b) & (u <= (W - 1 - b)) & (vp >= b) & (vp <= (H - 1 - b))
        if ok.sum().item() < min_corr_points:
            continue

        grid = _grid_from_uv(u[ok], vp[ok], W, H)
        col, zbuf = _sample_rgb_depth(im, dep, grid, cfg.depth_sample_mode)

        col0_ok = col0[ok]
        N_ok = Nw0[ok]
        X_ok = Xw0[ok]
        z_ok = z[ok]
        w_ok = w0[ok]

        if cfg.enable_depth_test:
            epsv = cfg.depth_eps_abs + cfg.depth_eps_rel * z_ok
            okd = torch.isfinite(zbuf) & (torch.abs(zbuf - z_ok) <= epsv)
            if okd.sum().item() < min_corr_points:
                continue
            col = col[okd]; col0_ok = col0_ok[okd]; N_ok = N_ok[okd]; X_ok = X_ok[okd]; z_ok = z_ok[okd]; w_ok = w_ok[okd]

        if cfg.reject_backfaces:
            Vdir = (Cw.view(1, 3) - X_ok)
            Vdir = Vdir / (torch.norm(Vdir, dim=1, keepdim=True) + 1e-12)
            ndv = torch.sum(N_ok * Vdir, dim=1)
            okf = ndv > 1e-6
            if okf.sum().item() < min_corr_points:
                continue
            col = col[okf]; col0_ok = col0_ok[okf]; z_ok = z_ok[okf]; ndv = ndv[okf]
            w_ok = w_ok[okf]
            w_fit = (torch.clamp(ndv, 0.0, 1.0) ** float(cfg.weight_power)).view(-1, 1)
            if cfg.use_distance_weight:
                w_fit = w_fit / (z_ok.view(-1, 1) * z_ok.view(-1, 1) + 1e-12)
        else:
            w_fit = w_ok

        a, bb = fit_affine_rgb_wls(col, col0_ok, w_fit)
        a = a.clamp(clamp_a[0], clamp_a[1])
        bb = bb.clamp(clamp_b[0], clamp_b[1])

        a_view[vi] = a
        b_view[vi] = bb

    # anchor fixed
    a_view[0] = torch.ones((3,), device=dev)
    b_view[0] = torch.zeros((3,), device=dev)
    return a_view, b_view


def compute_island_masks_from_atlas_coverage(atlas_mask_u8: np.ndarray):
    """
    atlas_mask_u8: (R,R) uint8 0/255 where mesh covers texel.
    returns list of boolean masks, one per connected component (UV island-ish).
    """
    mask = (atlas_mask_u8 > 0).astype(np.uint8)
    num, labels = cv2.connectedComponents(mask, connectivity=8)
    islands = []
    for lab in range(1, num):
        islands.append(labels == lab)
    return islands


def build_clip_from_K_w2c_opencv(
    Vw: torch.Tensor,
    K: torch.Tensor,
    w2c: torch.Tensor,
    H: int,
    W: int,
    near: float,
    far: float,
    flip_ndc_y: bool = True,
):
    """
    Option A:
      - stay entirely in OpenCV camera space
      - use K exactly as calibrated
      - produce clip coords directly (no CV→GL camera conversion)

    Row-vector convention:
      Xc_h = Vh @ w2c.T
      clip  = [x_clip, y_clip, z_clip, w_clip]

    Returns:
      pos_clip: (1,N,4) for nvdiffrast
      Xc:       (N,3) OpenCV camera-space coords (z is view-space depth)
    """
    device = Vw.device
    dtype = Vw.dtype

    N = Vw.shape[0]
    ones = torch.ones((N, 1), device=device, dtype=dtype)
    Vh = torch.cat([Vw, ones], dim=1)              # (N,4)

    # OpenCV camera space
    Xc_h = Vh @ w2c.T                               # (N,4)
    Xc = Xc_h[:, :3]
    x, y, z = Xc[:, 0], Xc[:, 1], Xc[:, 2]

    # Avoid division by zero / behind-camera points
    z = z.clamp_min(1e-6)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Pixel-center correction (critical)
    cx = cx #- 0.5
    cy = cy #- 0.5

    # ---- X / Y clip (derive directly from OpenCV projection) ----
    #
    # u = fx * x/z + cx
    # ndc_x = 2*u/(W-1) - 1
    #
    A = 2.0 * fx / (W - 1)
    B = 2.0 * cx / (W - 1) - 1.0
    clip_x = A * x + B * z

    if flip_ndc_y:
        # v = fy * y/z + cy   (v down)
        # ndc_y = 1 - 2*v/(H-1)
        C = 2.0 * fy / (H - 1)
        D = 1.0 - 2.0 * cy / (H - 1)
        clip_y = (-C) * y + D * z
    else:
        C = 2.0 * fy / (H - 1)
        D = 2.0 * cy / (H - 1) - 1.0
        clip_y = C * y + D * z

    # ---- Depth (monotonic, perspective-correct) ----
    #
    # z_ndc = (a*z + b) / z = a + b/z
    # such that:
    #   z=near -> -1
    #   z=far  -> +1
    #
    a = (far + near) / (far - near)
    b = (-2.0 * far * near) / (far - near)
    clip_z = a * z + b

    # Perspective-correct interpolation
    clip_w = z

    pos_clip = torch.stack(
        [clip_x, clip_y, clip_z, clip_w],
        dim=1
    )                                               # (N,4)

    return pos_clip[None, ...], Xc

@dataclass
class TexOptCfg:
    atlas_res: int = 2048
    lr: float = 0.05
    iters: int = 300
    views_per_iter: int = 4          # stochastic minibatch of views
    depth_eps_abs: float = 2e-4
    depth_eps_rel: float = 1e-3
    robust_delta: float = 0.05       # Charbonnier-like
    clamp_border: int = 4
    tv_weight: float = 1e-2          # total variation regularization
    use_depth_test: bool = True,
    near:float= 0.01
    far:float= 2.0
    clamp_border:int= 0
    lambda_dssim:float= 0.2
    debug:bool= False
    debug_every:int= 50

def charbonnier(x, eps=1e-3):
    return torch.sqrt(x*x + eps*eps)

def total_variation(img):
    # img: (R,R,3)
    dx = img[:, 1:, 0] - img[:, :-1, 0]
    dy = img[1:, :, 0] - img[:-1, :, 0]
    return dx.abs().mean() + dy.abs().mean()


import torch
import torch.nn.functional as F

def _gaussian_window(window_size: int, sigma: float, device):
    coords = torch.arange(window_size, device=device) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)  # (1,1,ws,ws)
    return w

def ssim(img1, img2, window_size=11, sigma=1.5, C1=0.01**2, C2=0.03**2):
    """
    img1,img2: (N,H,W,C) in [0,1], float32
    returns: scalar SSIM averaged over N and channels
    """
    assert img1.shape == img2.shape and img1.ndim == 4
    N,H,W,C = img1.shape
    device = img1.device

    # Convert to NCHW
    x = img1.permute(0,3,1,2)
    y = img2.permute(0,3,1,2)

    w = _gaussian_window(window_size, sigma, device)
    w = w.expand(C, 1, window_size, window_size)  # depthwise

    mu_x = F.conv2d(x, w, padding=window_size//2, groups=C)
    mu_y = F.conv2d(y, w, padding=window_size//2, groups=C)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x*x, w, padding=window_size//2, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y*y, w, padding=window_size//2, groups=C) - mu_y2
    sigma_xy = F.conv2d(x*y, w, padding=window_size//2, groups=C) - mu_xy

    num = (2*mu_xy + C1) * (2*sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = num / (den + 1e-8)
    return ssim_map.mean()

def dssim(img1, img2, **kwargs):
    return 0.5 * (1.0 - ssim(img1, img2, **kwargs))

def _logit(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x) - torch.log1p(-x)


def sobel_grad(img_nhwc: torch.Tensor) -> torch.Tensor:
    """
    img_nhwc: (N,H,W,3) float in [0,1] (or linear)
    returns:  (N,H,W,3,2) gradients (dx,dy) per channel
    """
    assert img_nhwc.ndim == 4 and img_nhwc.shape[-1] == 3
    x = img_nhwc.permute(0,3,1,2)  # NCHW
    N,C,H,W = x.shape

    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=x.device, dtype=x.dtype) / 8.0
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], device=x.device, dtype=x.dtype) / 8.0
    kx = kx.view(1,1,3,3).repeat(C,1,1,1)  # depthwise
    ky = ky.view(1,1,3,3).repeat(C,1,1,1)

    dx = F.conv2d(x, kx, padding=1, groups=C)
    dy = F.conv2d(x, ky, padding=1, groups=C)

    # back to NHWC with extra grad dim
    dx = dx.permute(0,2,3,1)
    dy = dy.permute(0,2,3,1)
    return torch.stack([dx, dy], dim=-1)  # (N,H,W,3,2)

def optimize_texture_atlas_logits_rgb(
    V2_np, F2_np, UV2_np, views, island_mask, cfg,
    init_atlas=None, device="cuda",
):
    dev = torch.device(device)
    R = cfg.atlas_res

    # Mesh tensors
    Vw  = torch.from_numpy(V2_np.astype(np.float32)).to(dev)
    tri = torch.from_numpy(F2_np.astype(np.int32)).to(dev)
    UV  = torch.from_numpy(UV2_np.astype(np.float32)).to(dev).clone()
    UV[..., 1] = 1.0 - UV[..., 1]

    ctx = dr.RasterizeCudaContext()
    topology_hash = dr.antialias_construct_topology_hash(tri)

    # --- Atlas params: logits ---
    if init_atlas is None:
        init_atlas = np.full((R, R, 3), 0.5, dtype=np.float32)
    init_atlas_t = torch.from_numpy(init_atlas.astype(np.float32)).to(dev).clamp(0, 1)

    atlas_logits = _logit(init_atlas_t).requires_grad_(True)

    # Island update mask
    island3 = None
    if island_mask is not None:
        island = torch.from_numpy(island_mask.astype(np.bool_)).to(dev)
        island3 = island[..., None].float()


    # Pack views
    packed = []
    flip_y_gt_images = True
    for v in views:
        K   = torch.from_numpy(np.asarray(v["K"],   np.float32)).to(dev)
        c2w = torch.from_numpy(np.asarray(v["c2w"], np.float32)).to(dev)
        w2c = torch.linalg.inv(c2w)

        im = v["im"]
        depth_im = v["depth_im"]
        if flip_y_gt_images:
            im = cv2.flip(im, 0)
            depth_im = cv2.flip(depth_im, 0)

        assert im.dtype == np.float32 and depth_im.dtype == np.float32
        im_t = torch.from_numpy(im).to(dev)            # (H,W,3) in [0,1]
        z_t  = torch.from_numpy(depth_im).to(dev)      # (H,W)
        packed.append((K, w2c, im_t, z_t))

    rng = np.random.default_rng(0)

    # Debug windows
    if getattr(cfg, "debug", False):
        cv2.namedWindow("PRED", cv2.WINDOW_KEEPRATIO); cv2.resizeWindow("PRED", 512, 512)
        cv2.namedWindow("GT",   cv2.WINDOW_KEEPRATIO); cv2.resizeWindow("GT",   512, 512)

    lambda_grad = getattr(cfg, "lambda_grad", 0.30)  # start small
    lambda_dssim = float(getattr(cfg, "lambda_dssim", 0.15))
    use_amp = bool(getattr(cfg, "use_amp", False))
    scaler = torch.amp.GradScaler(device=device,enabled=use_amp)

    num_views = len(packed)
    log_gt_scale = torch.nn.Parameter(torch.zeros((num_views, 3), device=dev))  # RGB log-scale
    gt_rgb_bias     = torch.nn.Parameter(torch.zeros((num_views, 3), device=dev))  # RGB bias

    opt = torch.optim.AdamW([atlas_logits, log_gt_scale, gt_rgb_bias], lr=cfg.lr, weight_decay=1e-4)
    anchor = 0
    
    
    for it in tqdm.tqdm(range(cfg.iters)):
        opt.zero_grad(set_to_none=True)

        bsz = min(cfg.views_per_iter, len(packed))
        view_ids = rng.choice(len(packed), size=bsz, replace=False)
        #if bsz > 1 and not anchor in view_ids: view_ids[0] = anchor
        

        with torch.amp.autocast(device_type=device,enabled=use_amp):
            # decode atlas each iter
            atlas = torch.sigmoid(atlas_logits)  # (R,R,3) in [0,1]
            loss_photo = torch.zeros((), device=dev, dtype=torch.float32)

            for vid in view_ids:
                K, w2c, im_gt, depth_gt = packed[vid]
                H, W = im_gt.shape[0], im_gt.shape[1]

                pos_clip, Xc = build_clip_from_K_w2c_opencv(Vw, K, w2c, H, W, cfg.near, cfg.far)
                rast, rast_db = dr.rasterize(ctx, pos_clip, tri, resolution=(H, W))
                valid = rast[..., 3] > 0

                if cfg.clamp_border > 0:
                    b = cfg.clamp_border
                    vb = torch.zeros((1, H, W), device=dev, dtype=torch.bool)
                    vb[:, b:H-b, b:W-b] = True
                    valid &= vb

                if not torch.any(valid):
                    continue

                # depth (camera space)
                Xc_pix, _ = dr.interpolate(Xc[None, ...].contiguous(), rast.contiguous(), tri.contiguous())
                z_pred = Xc_pix[..., 2]

                # UV + derivatives
                uv_samp, uv_da = dr.interpolate(
                    UV[None, ...].contiguous(),
                    rast.contiguous(),
                    tri.contiguous(),
                    rast_db=rast_db,
                    diff_attrs="all"
                )

                tex = atlas[None, ...]                  # (1,R,R,3)
                mip = dr.texture_construct_mip(tex)
                col_aliased = dr.texture(tex, uv_samp, uv_da=uv_da, mip=mip,
                                 filter_mode="linear-mipmap-linear",
                                 boundary_mode="clamp")
                col_pred = dr.antialias(col_aliased, rast, pos_clip, tri,
                                   topology_hash=topology_hash,
                                   pos_gradient_boost=1.0)
                
                col_pred = srgb_to_linear(col_pred)
                im_gt = srgb_to_linear(im_gt)
                

                # visibility mask (RE-ENABLED)
                vis = valid
                if cfg.use_depth_test:
                    zbuf = depth_gt[None, ...]
                    finite = torch.isfinite(zbuf) & (zbuf > 0)
                    eps = cfg.depth_eps_abs + cfg.depth_eps_rel * zbuf.abs()
                    vis = vis & finite & (torch.abs(zbuf - z_pred) <= eps)

                if not torch.any(vis):
                    continue

                # L1 on visible pixels
                
                s = torch.exp(log_gt_scale[vid])[None, None, None, :]   # (1,1,1,3)
                bb = gt_rgb_bias[vid][None, None, None, :]                 # (1,1,1,3)

                gt_corr = (im_gt[None, ...] * s + bb).clamp(0, 1)
                
                col_pred_visible = col_pred[vis[..., None].expand_as(col_pred)].view(-1, 3)
                gt_visible   = gt_corr[vis[..., None].expand_as(col_pred)].view(-1, 3)
                l1 = (col_pred_visible- gt_visible).abs().mean()
                
                

                # DSSIM
                if lambda_dssim > 0:
                    pred_full = col_pred.clone()
                    gt_full   = gt_corr.clone()
                    pred_full[~vis] = 0
                    gt_full[~vis]   = 0
                    d = dssim(pred_full, gt_full)

                    photo = (1 - lambda_dssim) * l1 + lambda_dssim * d 
                else:
                    photo = l1
                    
                if lambda_grad > 0:
                    # full-image gradients
                    g_pred = sobel_grad(col_pred)         # (1,H,W,3,2)
                    g_gt   = sobel_grad(gt_corr)          # (1,H,W,3,2)

                    # mask: same vis but needs broadcast to (..,3,2)
                    mask = vis[..., None, None].float()   # (1,H,W,1,1)
                    # L1 on gradients, normalized by visible area
                    grad_l1 = ( (g_pred - g_gt).abs() * mask ).sum() / (mask.sum() * 3 * 2 + 1e-8)

                    photo = photo + lambda_grad * grad_l1 
                    
                    

                loss_photo = loss_photo + photo

                if getattr(cfg, "debug", False) and (it % getattr(cfg, "debug_every", 50) == 0):
                    im_disp = (linear_to_srgb(col_pred).squeeze().detach().clamp(0,1).cpu().numpy() * 255).astype(np.uint8)
                    gt_disp = (linear_to_srgb(gt_corr).squeeze().detach().clamp(0,1).cpu().numpy() * 255).astype(np.uint8)
                    im_disp = cv2.cvtColor(im_disp, cv2.COLOR_RGB2BGR)
                    gt_disp = cv2.cvtColor(gt_disp, cv2.COLOR_RGB2BGR)
                    vis_np = vis.detach().cpu().numpy().squeeze()
                    im_disp[~vis_np, :] = 0
                    cv2.imshow("PRED", im_disp)
                    cv2.imshow("GT", gt_disp)

            loss = loss_photo / max(1, bsz)
            loss += 1e-2*(log_gt_scale[vid]**2).mean() / max(1, bsz)
            loss += 1e-2*(gt_rgb_bias[vid]**2).mean() / max(1, bsz)

            # TV regularization (in RGB, on decoded atlas)
            if cfg.tv_weight > 0:
                if island3 is not None:
                    loss = loss + cfg.tv_weight * total_variation(atlas * island3)
                else:
                    loss = loss + cfg.tv_weight * total_variation(atlas)
        

        scaler.scale(loss).backward()
        # after loss.backward()
        with torch.no_grad():
            if log_gt_scale.grad is not None: log_gt_scale.grad[anchor].zero_()
            if gt_rgb_bias.grad is not None:     gt_rgb_bias.grad[anchor].zero_()

        # also keep them exactly fixed
        with torch.no_grad():
            log_gt_scale[anchor].zero_()
            gt_rgb_bias[anchor].zero_() 

        # Mask gradients by island in *logit space* (still valid)
        if island3 is not None and atlas_logits.grad is not None:
            with torch.no_grad():
                atlas_logits.grad.mul_(island3)

        scaler.step(opt)
        scaler.update()

        if (it % 25) == 0:
            print(f"iter {it:04d} loss {float(loss):.6f}")
        #responsiveness
        if getattr(cfg, "debug", False):
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    atlas_out = torch.sigmoid(atlas_logits).detach().cpu().numpy().astype(np.float32)
    return atlas_out


class HeadlessRenderer:
    def __init__(self, width, height):
        # Create renderer
        self.width = width
        self.height = height
        self.renderer = None

    def create(self):
        self.renderer = headless_rendering.OffscreenRenderer(self.width, self.height)
        self.renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])
        self.renderer.scene.view.set_antialiasing(True,False)
        self.renderer.scene.view.set_sample_count(8)
        self.renderer.scene.view.set_post_processing(True)
        self.renderer.scene.view.set_ambient_occlusion(False,False)
        

    def cleanup(self):
        self.renderer.scene.clear_geometry()

    def add_geometry(self, geom, mat="default"):
        # Material:
        # - "defaultUnlit" shows vertex colors / textures without lighting changes.
        # - "defaultLit" uses lights (can be nicer, but changes appearance).
        if mat == "default":
            mat_ = headless_rendering.MaterialRecord()
            mat_.shader = "defaultUnlit"
        else: 
            mat_ = mat
        self.renderer.scene.add_geometry("mesh", geom, mat_)
        

    def render(self, K: np.ndarray,          # 3x3 intrinsics
        T_wc: np.ndarray,       # 4x4 camera-to-world OR world-to-camera (see below)
        z_in_view_space: bool = True,
        out_color_path=None,
        out_depth_u16_path=None):
        self.renderer.setup_camera(K, T_wc, self.width, self.height)

        color_o3d = self.renderer.render_to_image()
        
        if not out_color_path is None and len(out_color_path) > 0:
            o3d.io.write_image(out_color_path, color_o3d)

        depth_o3d = self.renderer.render_to_depth_image(z_in_view_space=z_in_view_space)
        depth = np.asarray(depth_o3d).astype(np.float32)
        depth[~np.isfinite(depth)] = 0
        
        if not out_depth_u16_path is None and len(out_depth_u16_path) > 0:
            valid_depth = (depth > 0)
            if valid_depth.any():
                depth_min = depth[depth > 0].min()
                depth_max = depth[depth > 0].max()
            else:
                depth_min = 0
                depth_max = 1e-6
            depth_disp = np.clip(np.clip(depth-depth_min,0, depth_max)/(depth_max-depth_min)*65535, 0, 65535).astype(np.uint16)
            depth_disp_o3d = o3d.geometry.Image(depth_disp)
            o3d.io.write_image(out_depth_u16_path, depth_disp_o3d)

        return np.asarray(color_o3d).astype(np.float32)/255.0, depth
    
    def destroy(self):
        self.renderer = None
    
    def __enter__(self):
        self.create()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.destroy()

# ============================================================
# Config
# ============================================================

BakeMode = Literal["blend", "best_view"]

@dataclass
class BakeCfg:
    atlas_res: int = 2048
    mode: BakeMode = "blend"     # "best_view" crisp, "blend" smoother

    # Visibility (depth is view-space Z)
    enable_depth_test: bool = True
    depth_eps_abs: float = 0
    depth_eps_rel: float = 2e-2

    # Backface + weights
    reject_backfaces: bool = True
    weight_power: float = 6.0
    use_distance_weight: bool = True

    # Depth sampling for z-buffer comparison
    depth_sample_mode: str = "nearest"  # "nearest" recommended

    # Image bounds
    clamp_border: int = 1
    exposure_aware: bool = False


# ============================================================
# Open3D UV atlas (robust resolver)
# ============================================================


def compute_open3d_uv_atlas(mesh_legacy: o3d.geometry.TriangleMesh,
                            res, gutter=2.0) -> o3d.t.geometry.TriangleMesh:
    
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
    max_stretch, num_charts, num_partitions = mesh_t.compute_uvatlas(
        res, gutter=gutter,max_stretch=0.16
    )

    if "texture_uvs" not in mesh_t.triangle:
        raise RuntimeError(
            f"UV atlas ran but no triangle['texture_uvs'] present. "
            f"Triangle attrs: {list(mesh_t.triangle.keys())}"
        )
        
    V = mesh_t.vertex["positions"].numpy().astype(np.float32)     # (N,3)
    Fidx = mesh_t.triangle["indices"].numpy().astype(np.int32)    # (M,3)

    UV_tri = mesh_t.triangle["texture_uvs"].numpy().astype(np.float32)
    if UV_tri.ndim == 2:
        UV_tri = UV_tri.reshape(-1, 3, 2)

    print("UV Split vertices for uvs")
    # Make seam-safe buffers for baking
    V2, F2, UV2 = split_vertices_for_uvs(V, Fidx, UV_tri)
    print("vertices split")
        
    return V2, F2, UV2

def compute_uv_atlas(mesh_legacy, res, gutter=2.0, max_stretch=0.16667, max_charts = 0):
    V = np.asarray(mesh_legacy.vertices)
    F = np.asarray(mesh_legacy.triangles)
    vmapping, indices, uvs = pyatlas.atlas(V, F,
                                        maxCharts = max_charts,
                                        maxStretch = max_stretch,
                                        gutter = gutter,
                                        width = res,
                                        height= res)
    return V[vmapping], indices, uvs

def split_vertices_for_uvs(V: np.ndarray, Fidx: np.ndarray, UV_tri: np.ndarray):
    """
    Convert Open3D UVs (per triangle-vertex) to seam-safe split-vertex buffers.

    Inputs:
      V: (N,3)
      Fidx: (M,3)
      UV_tri: (M,3,2)

    Outputs:
      V2: (M*3,3)
      F2: (M,3)
      UV2: (M*3,2)
    """
    M = Fidx.shape[0]
    corners = Fidx.reshape(-1)                  # (M*3,)
    V2 = V[corners].astype(np.float32)          # (M*3,3)
    UV2 = UV_tri.reshape(-1, 2).astype(np.float32)
    F2 = np.arange(M * 3, dtype=np.int32).reshape(M, 3)
    return V2, F2, UV2


# ============================================================
# nvdiffrast bake (Option C)
# ============================================================

def _to_torch(x, device, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.from_numpy(np.asarray(x)).to(device=device, dtype=dtype)

def _make_hom(x):
    ones = torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype)
    return torch.cat([x, ones], dim=1)

def pad_atlas_gutter(atlas_rgb: torch.Tensor,
                     valid: torch.Tensor,
                     gutter: int = 16) -> torch.Tensor:
    """
    atlas_rgb: (R,R,3) float32 in [0,1]
    valid:     (R,R)   bool (True where rasterized)
    gutter:    how many pixels of padding to fill outward
    Returns padded atlas_rgb (same shape).
    """
    assert atlas_rgb.ndim == 3 and atlas_rgb.shape[-1] == 3
    assert valid.ndim == 2
    R = atlas_rgb.shape[0]
    assert atlas_rgb.shape[1] == R and valid.shape[0] == R and valid.shape[1] == R

    out = atlas_rgb.clone()
    m = valid.clone()

    # Iterate: each step expands valid region by 1px using 4-neighborhood
    for _ in range(int(gutter)):
        # Neighbors (shifted masks)
        up    = torch.zeros_like(m); up[1:]  = m[:-1]
        down  = torch.zeros_like(m); down[:-1]= m[1:]
        left  = torch.zeros_like(m); left[:, 1:] = m[:, :-1]
        right = torch.zeros_like(m); right[:, :-1]= m[:, 1:]

        grow = (~m) & (up | down | left | right)
        if not torch.any(grow):
            break

        # For pixels to fill, copy color from any available neighbor.
        # Priority order: up, down, left, right (you can change).
        fill = torch.zeros_like(out)

        # Copy from up neighbor (meaning the pixel above is valid -> copy its color down)
        src_up = torch.zeros_like(out); src_up[1:] = out[:-1]
        take = grow & up
        fill[take] = src_up[take]

        # Copy from down neighbor
        src_down = torch.zeros_like(out); src_down[:-1] = out[1:]
        take = grow & (~(grow & up)) & down
        fill[take] = src_down[take]

        # Copy from left neighbor
        src_left = torch.zeros_like(out); src_left[:, 1:] = out[:, :-1]
        take = grow & (~(grow & up)) & (~(grow & down)) & left
        fill[take] = src_left[take]

        # Copy from right neighbor
        src_right = torch.zeros_like(out); src_right[:, :-1] = out[:, 1:]
        take = grow & (~(grow & up)) & (~(grow & down)) & (~(grow & left)) & right
        fill[take] = src_right[take]

        out[grow] = fill[grow]
        m[grow] = True

    return out

def bake_multiview_atlas_nvdiffrast(
    V2: np.ndarray,                 # (N,3) float32
    F2: np.ndarray,                 # (M,3) int32
    UV2: np.ndarray,                # (N,2) float32 in [0,1] per-vertex UV (split-vertex recommended)
    views: List[Dict[str, Any]],    # each: {K(3,3), c2w(4,4), im(H,W,3), depth_im(H,W)}
    cfg: BakeCfg,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray]:

    dev = torch.device(device)
    R = int(cfg.atlas_res)

    V = _to_torch(V2, dev)                                    # (N,3)
    tri = torch.from_numpy(np.asarray(F2, np.int32)).to(dev)  # (M,3)
    UV = _to_torch(UV2, dev)                                  # (N,2)

    # Vertex normals
    v0 = V[tri[:, 0]]
    v1 = V[tri[:, 1]]
    v2 = V[tri[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=1)  # (M,3)
    N = torch.zeros_like(V)
    for k in range(3):
        N.index_add_(0, tri[:, k], fn)
    N = N / (torch.norm(N, dim=1, keepdim=True) + 1e-12)

    # Rasterize once in UV space
    uv_clip = torch.stack([
        2.0 * UV[:, 0] - 1.0,
        1.0 - 2.0 * UV[:, 1],                 # flip v so (0,0) is top-left in atlas image
        torch.zeros_like(UV[:, 0]),
        torch.ones_like(UV[:, 0])
    ], dim=1)  # (N,4)

    ctx = dr.RasterizeCudaContext()
    rast, _ = dr.rasterize(ctx, uv_clip[None, ...], tri, resolution=(R, R))  # (1,R,R,4)
    mask = (rast[..., 3] > 0)                                               # (1,R,R)

    Xw, _ = dr.interpolate(V[None, ...], rast, tri)  # (1,R,R,3)
    Nw, _ = dr.interpolate(N[None, ...], rast, tri)  # (1,R,R,3)
    Nw = Nw / (torch.norm(Nw, dim=-1, keepdim=True) + 1e-12)

    # Flatten valid texels
    mask_flat = mask.view(-1)
    idxs = torch.nonzero(mask_flat, as_tuple=False).squeeze(1)
    Xw_flat = Xw.view(-1, 3)[idxs]
    Nw_flat = Nw.view(-1, 3)[idxs]
    ys = (idxs // R).to(torch.int64)
    xs = (idxs % R).to(torch.int64)

    atlas_rgb = torch.zeros((R, R, 3), device=dev, dtype=torch.float32)
    if cfg.mode == "blend":
        atlas_w = torch.zeros((R, R, 1), device=dev, dtype=torch.float32)
    else:
        atlas_best_w = torch.full((R, R, 1), -1e9, device=dev, dtype=torch.float32)

    # Per-view accumulate
    for vi in tqdm.tqdm(range(len(views))):
        v = views[vi]
        K = _to_torch(v["K"], dev)
        c2w = _to_torch(v["c2w"], dev)
        w2c = torch.linalg.inv(c2w)
        Cw = c2w[:3, 3]

        im = _to_torch(v["im"], dev)          # (H,W,3) RGB float
        depth = _to_torch(v["depth_im"], dev) # (H,W) view-space Z
        H, W = im.shape[0], im.shape[1]

        Xc = (_make_hom(Xw_flat) @ w2c.T)[:, :3]
        z = Xc[:, 2]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        u = fx * (Xc[:, 0] / z) + cx
        vv = fy * (Xc[:, 1] / z) + cy

        ok = z > 1e-6
        b = cfg.clamp_border
        ok = ok & (u >= b) & (u <= (W - 1 - b)) & (vv >= b) & (vv <= (H - 1 - b))
        if not torch.any(ok):
            continue

        u = u[ok]; vv = vv[ok]; z = z[ok]
        Xw_ok = Xw_flat[ok]
        Nw_ok = Nw_flat[ok]
        xs_ok = xs[ok]
        ys_ok = ys[ok]
            
        # --- exposure correction to anchor (view 0) ---
        if getattr(cfg, "exposure_aware", True):
            a_view, b_view = compute_exposure_to_anchor_affine(
                Xw_flat=Xw_flat,
                Nw_flat=Nw_flat,
                views=views,
                cfg=cfg,
                dev=dev,
                min_corr_points=getattr(cfg, "min_corr_points", 4096),
            )
        else:
            a_view = torch.ones((len(views), 3), device=dev, dtype=torch.float32)
            b_view = torch.zeros((len(views), 3), device=dev, dtype=torch.float32) 
        

        # grid_sample coords
        gx = (u / (W - 1)) * 2.0 - 1.0
        gy = (vv / (H - 1)) * 2.0 - 1.0
        grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)

        im_t = im.permute(2, 0, 1).unsqueeze(0)         # (1,3,H,W)
        depth_t = depth.unsqueeze(0).unsqueeze(0)       # (1,1,H,W)

        col = F.grid_sample(im_t, grid, mode="bilinear", align_corners=True)
        col = col.squeeze(0).squeeze(-1).T  # (P,3)
        
       
        # Map this view into anchor photometric space
        col = srgb_to_linear(col) * a_view[vi].view(1, 3) + b_view[vi].view(1, 3) 
        col = linear_to_srgb(col).clamp(0.0, 1.0) 

        zbuf = F.grid_sample(depth_t, grid, mode=cfg.depth_sample_mode, align_corners=True)
        zbuf = zbuf.squeeze(0).squeeze(0).squeeze(-1)   # (P,)

        if cfg.enable_depth_test:
            eps = cfg.depth_eps_abs + cfg.depth_eps_rel * z
            okd = torch.isfinite(zbuf) & (torch.abs(zbuf - z) <= eps)
            if not torch.any(okd):
                continue
            col = col[okd]; z = z[okd]
            Xw_ok = Xw_ok[okd]; Nw_ok = Nw_ok[okd]
            xs_ok = xs_ok[okd]; ys_ok = ys_ok[okd]

        if cfg.reject_backfaces:
            Vdir = Cw.view(1, 3) - Xw_ok
            Vdir = Vdir / (torch.norm(Vdir, dim=1, keepdim=True) + 1e-12)
            ndv = torch.sum(Nw_ok * Vdir, dim=1)
            okf = ndv > 1e-6
            if not torch.any(okf):
                continue
            col = col[okf]; z = z[okf]
            xs_ok = xs_ok[okf]; ys_ok = ys_ok[okf]
            ndv = ndv[okf]
        else:
            ndv = torch.ones((col.shape[0],), device=dev, dtype=torch.float32)

        w = torch.clamp(ndv, 0.0, 1.0) ** cfg.weight_power
        if cfg.use_distance_weight:
            w = w / (z * z + 1e-12)
        w = w.view(-1, 1)
        view_bias = 1e-4 * vi
        w = w + view_bias
 

        if cfg.mode == "blend":
            atlas_rgb[ys_ok, xs_ok] += col * w
            atlas_w[ys_ok, xs_ok] += w
        else:
            prev = atlas_best_w[ys_ok, xs_ok]
            better = w > prev
            if torch.any(better):
                sel = better.squeeze(1)
                yb = ys_ok[sel]; xb = xs_ok[sel]
                atlas_best_w[yb, xb] = w[sel]
                atlas_rgb[yb, xb] = col[sel]

    if cfg.mode == "blend":
        valid = atlas_w[..., 0] > 1e-12
        atlas_rgb[valid] = atlas_rgb[valid] / atlas_w[valid]
    else:
        valid = atlas_best_w[..., 0] > -1e8
    
    atlas_rgb = pad_atlas_gutter(atlas_rgb, valid, gutter=getattr(cfg, "gutter_px", 16))
    atlas = torch.clamp(atlas_rgb, 0.0, 1.0).detach().cpu().numpy().astype(np.float32)
    mask = (valid.detach().cpu().numpy().astype(np.uint8) * 255)
    return atlas, mask


# ============================================================
# Your loop (build views) + MAIN
# ============================================================


def select_largest_mesh_component(mesh):
    print("Cluster connected triangles")
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    mesh_1 = copy.deepcopy(mesh)
    largest_cluster_idx = cluster_n_triangles.argmax()
    triangles_to_remove = triangle_clusters != largest_cluster_idx
    mesh_1.remove_triangles_by_mask(triangles_to_remove)
    return mesh_1

def make_mesh_manifold(mesh):
    mesh = mesh.remove_non_manifold_edges()
    while True:
        non_manifold_vertices = mesh.get_non_manifold_vertices()
        if len(non_manifold_vertices) == 0: break
        mesh.remove_vertices_by_index(non_manifold_vertices)
    #self_intersec = mesh.get_self_intersecting_triangles()
    #mesh.remove_triangles_by_index(np.unique(np.asarray(self_intersec)).tolist())
    mesh = mesh.remove_duplicated_vertices()
    mesh = mesh.remove_degenerate_triangles()
    mesh = mesh.remove_duplicated_triangles()
    mesh = select_largest_mesh_component(mesh)
    return mesh


def save_atlas_png(path: str, atlas_rgb: np.ndarray):
    atlas_u8 = (np.clip(atlas_rgb, 0.0, 1.0) * 255).astype(np.uint8)
    cv2.imwrite(path, atlas_u8[..., ::-1])  # RGB -> BGR


def write_textured_ply_binary_le(
    ply_path: str,
    texture_filename: str,
    V2: np.ndarray,      # (N,3) float32/float64
    F2: np.ndarray,      # (M,3) int32/int64
    UV2: np.ndarray,     # (N,2) float32/float64 in [0,1]
):
    """
    Binary little-endian PLY with per-face-vertex UVs (texcoord list).

    Face properties:
      - vertex_indices: list uchar int   (count=3, then 3x int32)
      - texcoord:       list uchar float (count=6, then 6x float32: u0 v0 u1 v1 u2 v2)

    Header includes:
      comment TextureFile <texture_filename>
    """
    V2 = np.asarray(V2)
    F2 = np.asarray(F2)
    UV2 = np.asarray(UV2)

    assert V2.ndim == 2 and V2.shape[1] == 3
    assert F2.ndim == 2 and F2.shape[1] == 3
    assert UV2.ndim == 2 and UV2.shape[1] == 2
    assert V2.shape[0] == UV2.shape[0], "V2 and UV2 must have same vertex count (seam-split buffers)."

    # Convert to desired dtypes
    V2_f = V2.astype(np.float32, copy=False)
    F2_i = F2.astype(np.int32, copy=False)
    UV2_f = UV2.astype(np.float32, copy=False)

    N = V2_f.shape[0]
    M = F2_i.shape[0]

    # ---------------- Header (ASCII) ----------------
    header = []
    header.append("ply")
    header.append("format binary_little_endian 1.0")
    header.append(f"comment TextureFile {texture_filename}")

    header.append(f"element vertex {N}")
    header.append("property float x")
    header.append("property float y")
    header.append("property float z")

    header.append(f"element face {M}")
    header.append("property list uchar int vertex_indices")
    header.append("property list uchar float texcoord")
    header.append("end_header\n")  # newline required

    header_bytes = ("\n".join(header)).encode("ascii")

    # ---------------- Write binary ----------------
    with open(ply_path, "wb") as f:
        f.write(header_bytes)

        # vertices: N * (3*float32)
        f.write(V2_f.tobytes(order="C"))

        # faces:
        # for each face:
        #   uchar 3
        #   int32 i0 i1 i2
        #   uchar 6
        #   float32 u0 v0 u1 v1 u2 v2
        pack_u8 = struct.Struct("<B").pack
        pack_i3 = struct.Struct("<iii").pack
        pack_f6 = struct.Struct("<ffffff").pack

        for tri in F2_i:
            i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])

            # indices list
            f.write(pack_u8(3))
            f.write(pack_i3(i0, i1, i2))

            # texcoord list (per-corner UVs)
            uv = UV2_f[[i0, i1, i2]].reshape(-1)  # (6,)
            f.write(pack_u8(6))
            f.write(pack_f6(float(uv[0]), float(uv[1]),
                            float(uv[2]), float(uv[3]),
                            float(uv[4]), float(uv[5])))

    print(f"Wrote binary textured PLY: {ply_path}")

def main():
    # ---- your parameters ----
    step = 50
    mesh_path = "/tmp/mesh_tsdf_10000.ply"
    data_dir = "/home/fabianb/dev/masterthesis/data/iphone_recording/fabian_front_noaf_hq_jpegs"
    img_dir = os.path.join(data_dir, "images")
    atlas_res = 2048
    resize_to_depthmap_size = True
    use_feats = True
    
    mapx = np.load(os.path.join(data_dir, "map_x.npy"), allow_pickle=True).item()
    mapy = np.load(os.path.join(data_dir, "map_y.npy"), allow_pickle=True).item()
    print("Undistortion map_x Keys: ", mapx.keys())
    print("Undistortion map_y Keys: ", mapy.keys())
    mapx, mapy = (
        mapx[(1,1)],
        mapy[(1,1)],
    )
    

    # ---- load mesh ----
    print("Load Mesh")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    print("Make Mesh Manifold")
    mesh = make_mesh_manifold(mesh)
    print("Done")

    # ---- compute Open3D UV atlas ----
    print("Computing UV Atlas")
    V2, F2, UV2 = compute_open3d_uv_atlas(mesh, res=atlas_res, gutter=2.0)
    #V2, F2, UV2 = compute_uv_atlas(mesh, res=atlas_res, gutter=2.0)
    print("UV Atlas Computed")


    # ---- collect views (your loop) ----
    views: List[Dict[str, Any]] = []
    print("Collecting views")
    
    max_image_size = 1280
    
    for i in tqdm.tqdm(range(1, 1000, step)):
        for i_ in range(i, i + step, 1):
            try:
                dm = torch.load(f"/tmp/{i_}_stereo_depth.pth").squeeze()
                transform = torch.load(f"/tmp/{i_}_stereo_depth_transform.pth")
                K_h, K_w = int(dm.shape[0]), int(dm.shape[1])
                
                img_path = f"{img_dir}/{i_:05d}.jpeg"
                if use_feats: bgr = cv2.imread(f"/tmp/{i_}_stereo_depth_feats.png")
                else: 
                    bgr = cv2.imread(img_path)
                    bgr = cv2.remap(bgr, mapx, mapy, cv2.INTER_CUBIC)
                if bgr is None:
                    continue
                image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                if resize_to_depthmap_size:
                    if K_h <= max_image_size and K_w <= max_image_size:
                        image = cv2.resize(image, (K_w, K_h))
                    else:
                        ratio = max_image_size/max(K_h,K_w)
                        image = cv2.resize(image, (int(K_w*ratio), int(K_h*ratio)))

                h, w = int(image.shape[0]), int(image.shape[1])
                K_scale_h = h / K_h
                K_scale_w = w / K_w

                c2w = transform[0].cpu().numpy().squeeze().astype(np.float32)
                K = transform[1].cpu().numpy().squeeze().astype(np.float32)

                K_scaled = K.copy()
                K_scaled[0, :] *= K_scale_w
                K_scaled[1, :] *= K_scale_h

                # Render depth at RGB resolution (view-space Z) using your renderer
                with HeadlessRenderer(w, h) as renderer:
                    renderer.add_geometry(mesh)
                    _, depth = renderer.render(K_scaled, np.linalg.inv(c2w), True)

                depth = np.asarray(depth, dtype=np.float32)

                views.append({
                    "K": K_scaled,
                    "c2w": c2w,
                    "im": image,
                    "depth_im": depth,
                })
                break
            except Exception:
                pass

    print("Collected views:", len(views))
    if len(views) == 0:
        raise RuntimeError("No views collected. Check your paths/transforms/rendering.")


    # ---- bake ----
    cfg = BakeCfg(
        atlas_res=atlas_res,
        mode="best_view",
        enable_depth_test=True,
        depth_eps_abs = 5e-3,   # was ~5e-4
        depth_eps_rel=2e-3,
        reject_backfaces=True,
        weight_power=2.0,
        use_distance_weight=True,
        depth_sample_mode="nearest",
        clamp_border=1,
        exposure_aware=False
    )

    print("Baking views")
    atlas_rgb, atlas_mask = bake_multiview_atlas_nvdiffrast(V2, F2, UV2, views, cfg, device="cuda")
    print("Baking views")

    # ---- save ----
    cv2.imwrite("/tmp/atlas.png", (atlas_rgb[..., ::-1] * 255.0).astype(np.uint8))  # RGB->BGR
    #cv2.imwrite("/tmp/atlas_mask.png", atlas_mask)
    print("Wrote /tmp/atlas.png and /tmp/atlas_mask.png")
    
    atlas_path = "/tmp/atlas.png"
    ply_path = "/tmp/mesh_textured.ply"

    save_atlas_png(atlas_path, atlas_rgb)

    # Use a relative name in the PLY comment so tools find it next to the PLY
    write_textured_ply_binary_le(
        ply_path=ply_path,
        texture_filename=os.path.basename(atlas_path),
        V2=V2,
        F2=F2,
        UV2=UV2,
    )
    
    
    opt_cfg = TexOptCfg(
        atlas_res=atlas_res,
        lr=3e-4,
        iters=2000,
        views_per_iter=4,

        # In [0,1] scale: start tight. If too many pixels get rejected -> raise to 3e-3..8e-3
        depth_eps_abs=0,
        depth_eps_rel=3e-3,

        # Tiny TV is important to avoid speckle
        tv_weight=0,

        use_depth_test=True,

        # Clip planes: keep far tight for depth precision
        near=0.01,
        far=5.0,          # if your cameras can see the object at z~1; otherwise far=1.5 is even better

        clamp_border=8,

        # GS-ish but not too strong
        lambda_dssim=0.15,

        debug=True,
        debug_every=50
    )
        
    # atlas_rgb_init, atlas_mask from your current fast bake
    if False:
        islands = compute_island_masks_from_atlas_coverage(atlas_mask)

        # Pick largest island first (face area)
        islands_sorted = sorted(islands, key=lambda m: m.sum(), reverse=True)
        
        atlas_opt = atlas_rgb.copy()
        for i in range(len(islands_sorted)):
            atlas_opt = optimize_texture_atlas(
                V2_np=V2, F2_np=F2, UV2_np=UV2,
                views=views,
                island_mask=islands_sorted[i],
                cfg=opt_cfg,
                init_atlas=atlas_opt,
                device="cuda",
            ) 
        
    else:
        atlas_opt = optimize_texture_atlas_logits_rgb(
            V2_np=V2, F2_np=F2, UV2_np=UV2,
            views=views,
            island_mask=None,#island0,
            cfg=opt_cfg,
            init_atlas=atlas_rgb,
            device="cuda"
        ) 
    
    
    atlas_path = "/tmp/atlas_opt.png"
    ply_path = "/tmp/mesh_textured_opt.ply"
    
    save_atlas_png(atlas_path, atlas_opt)

    # Use a relative name in the PLY comment so tools find it next to the PLY
    write_textured_ply_binary_le(
        ply_path=ply_path,
        texture_filename=os.path.basename(atlas_path),
        V2=V2,
        F2=F2,
        UV2=UV2,
    )
    


if __name__ == "__main__":
    main()
