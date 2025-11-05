import math
from typing import Optional, Dict, Tuple

import numpy as np

import torch
import torch.nn as nn
from pytorch3d import transforms as torchtransforms

import open3d as o3d

from gsplat.optimizers import SelectiveAdam
from utils import rgb_to_sh, knn

from gsplat.strategy import MCMCStrategy, Strategy

# assumes you already have rgb_to_sh and knn available in scope
# rgb_to_sh: (N,3) rgb in [0,1] -> (N,3) SH DC coeffs
# knn(x,k):  (N,3), returns (N,k) distances to k nearest neighbors (just like your current code expects)


class Splats(nn.Module):
    
    
    def type_str(self) -> str:
        pass
    
    def step_pre_backward(
        self,
        *args,
        **kwargs,
    ):
        pass
    
    def step_post_backward(
        self,
        *args,
        **kwargs,
    ):
        pass
    
    def check_sanity(self):
        pass
    
    def create_optimizers(self, max_steps:int, *args, **kwargs):
        pass
    
    def as_renderer_dict(self) ->Dict[str, torch.Tensor]:
        pass
    
    def npoints(self) -> int: 
        pass

    def world_means(self) -> torch.Tensor:
        # Already in world coords, directly learned
        pass

    def world_quats(self) -> torch.Tensor:
        # Normalize so it's a proper unit quaternion
        pass

    def world_scales(self) -> torch.Tensor:
        # Your training loop expects exp(scales) (since you store log scale).
        # This mirrors how you do torch.exp(self.splats["scales"]) elsewhere.
        pass

    def world_opacities(self) -> torch.Tensor:
        # Return sigmoid(opacities) if you need alpha in [0,1].
        # BUT: in the train loop you keep "opacities" as logits to optimize and only do sigmoid() in loss.
        # SurfaceSplats.as_renderer_dict() returned the logits (not the sigmoid).
        # We'll follow that convention for consistency.
        pass


class FreeSplats(Splats):
    """
    Unconstrained (free) Gaussians in world coordinates.

    Trainable params:
        means:        (N,3)        world-space centers
        scales:       (N,3)        log-scale or log-radius per axis
        quats:        (N,4)        orientation quaternions (not normalized yet, we normalize on read)
        opacities:    (N,)         opacity logits
        sh0:          (N,1,3)      SH DC coeffs
        shN:          (N,K-1,3)    SH higher coeffs
    OR (if feature_dim is not None):
        features:     (N,F)
        colors:       (N,3)        color logits for the base color term

    This mirrors SurfaceSplats' interface:
        - world_means()
        - world_quats()
        - world_scales()
        - world_opacities()
        - as_renderer_dict()
    """
    def type_str(self) -> str:
        return "FreeSplats"

    def __init__(
        self,
        means: torch.Tensor,            # (N,3)
        scales: torch.Tensor,           # (N,3)
        quats: torch.Tensor,            # (N,4)
        opacities: torch.Tensor,        # (N,)
        sh0: Optional[torch.Tensor],    # (N,1,3) or None if using features/colors
        shN: Optional[torch.Tensor],    # (N,K-1,3) or None
        features: Optional[torch.Tensor],  # (N,F) or None
        colors: Optional[torch.Tensor],    # (N,3) logits or None
        device: str = "cuda",
    ):
        super().__init__()

        # core parameters
        
        plain_params_dict = {
            "means": means.to(device),# (N,3)
            "scales": scales.to(device),# (N,3) in log-space (that's how you're treating it in your loop)
            "quats": quats.to(device),# (N,4) not guaranteed normalized
            "opacities": opacities.to(device),# (N,) logits
        }
        

        # appearance
        self._use_sh = sh0 is not None and shN is not None
        if self._use_sh:
            plain_params_dict["sh0"] = sh0.to(device)             # (N,1,3)
            plain_params_dict["shN"] = shN.to(device)             # (N,K-1,3)
        else:
            plain_params_dict["features"] = features.to(device)   # (N,F)
            plain_params_dict["colors"] = colors.to(device)       # (N,3) logits (like logit(rgb))
        
        
        self.optimizers = None
        self.params_dict = nn.ParameterDict(plain_params_dict)
        
    def check_sanity(self):
        self.strategy.check_sanity(self.params_dict, self.optimizers)
                              
    # ---------------- world-space getters to match SurfaceSplats ---------------- #
    
    def __get_param_from_dict(self, key):
        if key in self.params_dict: return self.params_dict[key]
        return None
    
    @property
    def means(self):
        return self.__get_param_from_dict("means")
    
    @property
    def scales(self):
        return self.__get_param_from_dict("scales")
    
    @property
    def quats(self):
        return self.__get_param_from_dict("quats")
    
    @property
    def opacities(self):
        return self.__get_param_from_dict("opacities")
    @property
    def sh0(self):
        return self.__get_param_from_dict("sh0")
    @property
    def shN(self):
        return self.__get_param_from_dict("shN")
    @property
    def features(self):
        return self.__get_param_from_dict("features")
    @property
    def colors(self):
        return self.__get_param_from_dict("colors")
    
    def npoints(self) -> int: return len(self.means)

    def world_means(self) -> torch.Tensor:
        # Already in world coords, directly learned
        return self.means

    def world_quats(self) -> torch.Tensor:
        # Normalize so it's a proper unit quaternion
        q = self.quats
        return q  # (N,4) (w,x,y,z) assumed by downstream code

    def world_scales(self) -> torch.Tensor:
        # Your training loop expects exp(scales) (since you store log scale).
        # This mirrors how you do torch.exp(self.splats["scales"]) elsewhere.
        return torch.exp(self.scales)  # (N,3) positive radii

    def world_opacities(self) -> torch.Tensor:
        # Return sigmoid(opacities) if you need alpha in [0,1].
        # BUT: in the train loop you keep "opacities" as logits to optimize and only do sigmoid() in loss.
        # SurfaceSplats.as_renderer_dict() returned the logits (not the sigmoid).
        # We'll follow that convention for consistency.
        return torch.sigmoid(self.opacities)
    
    
    
    
    
    
    # The renderer-facing dict should match what rasterize_splats expects in your loop.
    def as_renderer_dict(self) -> Dict[str, torch.Tensor]:
        d = {
            "means":     self.world_means(),     # (N,3)
            "quats":     self.world_quats(),     # (N,4)
            "scales":    self.world_scales(),    # (N,3)
            "opacities": self.world_opacities(),         # logits (N,)
        }
        if self._use_sh:
            d["sh0"] = self.sh0        # (N,1,3)
            d["shN"] = self.shN        # (N,K-1,3)
        else:
            d["features"] = self.features  # (N,F)
            d["colors"]   = self.colors    # (N,3) logits
        return d
    
    
    
    def step_pre_backward(
        self,
        step, info,
        **kwargs,
    ):
        self.strategy.step_pre_backward(
                    params=self.params_dict,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info, **kwargs)
        pass

    def step_post_backward(
        self,
        step, info,
        **kwargs,
    ):
        for s in self.schedulers.values(): s.step()
        kwargs_for_strategy = kwargs
        if isinstance(self.strategy, MCMCStrategy): 
            kwargs_for_strategy["lr"] = self.schedulers["means"].get_last_lr()[0]
        self.strategy.step_post_backward(step=step, info=info, state = self.strategy_state,
                                         params=self.params_dict, optimizers=self.optimizers, **kwargs_for_strategy)
        pass

    
    
    def initialize_strategy(self, strategy: Strategy = MCMCStrategy(100_000), **kwargs):
        self.strategy = strategy
        self.strategy_state = self.strategy.initialize_state(**kwargs)
    
    def create_optimizers(self, 
                          max_steps : int, 
                          means_lr: float, scales_lr: float, opacities_lr: float,
                          quats_lr: float, sh0_lr: float, shN_lr: float,
                          batch_size: int, world_size: int, 
                          sparse_grad: bool = False, visible_adam: bool = False):
        # 6. build per-parameter optimizers with per-param LRs
        # same scaling rule you had: lr * sqrt(BS), eps / sqrt(BS)
        BS = batch_size * world_size
        if sparse_grad:
            optimizer_class = torch.optim.SparseAdam
        elif visible_adam:
            optimizer_class = SelectiveAdam  # must exist in your code
        else:
            optimizer_class = torch.optim.Adam
        # map param names -> (tensor, lr)
        param_specs = [
            ("means",      self.means,      means_lr),
            ("scales",     self.scales,     scales_lr),
            ("quats",      self.quats,      quats_lr),
            ("opacities",  self.opacities,  opacities_lr),
        ]

        if self._use_sh:
            param_specs += [
                ("sh0",     self.sh0,       sh0_lr),
                ("shN",     self.shN,       shN_lr),
            ]
        else:
            # feature path
            param_specs += [
                ("features", self.features, sh0_lr),
                ("colors",   self.colors,   sh0_lr),
            ]
        self.optimizers = {
            name: optimizer_class(
                [{
                    "params": [tensor],
                    "lr": lr * math.sqrt(BS),
                    "name": name,
                }],
                eps=1e-15 / math.sqrt(BS),
                betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
            )
            for (name, tensor, lr) in param_specs
        }
        
        self.schedulers = {}
        for k in self.optimizers:
            if k == "means":
                self.schedulers[k] = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizers[k], gamma=0.01 ** (1.0 / max_steps)
                )
                
        assert "means" in self.schedulers

# You already have rgb_to_sh in your codebase. We'll assume it's available.
# from utils.sh_utils import rgb_to_sh

def rotation_matrix_to_quat(R: torch.Tensor) -> torch.Tensor:
    return torchtransforms.matrix_to_quaternion(R)


class SurfaceSplats(Splats):
    """
    Surface-attached Gaussian splats.

    We do NOT store world means directly.
    Instead:
        - uv_params: (N,2) learnable offset coords in triangle's tangent basis
        - scale_logits: (N,3) learnable log-scale (later exp -> positive)
        - opacity_logits: (N,) learnable logits for opacity
        - sh0: (N,1,3) learnable SH DC
        - shN: (N,K-1,3) learnable SH higher bands
    Fixed buffers:
        - tri_ids: (N,) long -> which triangle each splat belongs to
        - base_o, base_e1, base_e2, base_n: (F,3) per-triangle tangent frame
        - base_scale: (F,3) per-triangle base scale for anisotropy
    """
    
    def type_str(self):
        return "SurfaceSplats"

    def __init__(
        self,
        uv_params: torch.Tensor,           # (N,2)
        tri_ids: torch.Tensor,             # (N,)
        base_o: torch.Tensor,              # (F,3)
        base_e1: torch.Tensor,             # (F,3)
        base_e2: torch.Tensor,             # (F,3)
        base_n: torch.Tensor,              # (F,3)
        base_scale: torch.Tensor,          # (F,3)
        scale_logits: torch.Tensor,        # (N,3)
        opacity_logits: torch.Tensor,      # (N,)
        sh0: torch.Tensor,                 # (N,1,3)
        shN: torch.Tensor,                 # (N,K-1,3)
        features: torch.Tensor,
        colors: torch.Tensor,
        device: str = "cuda",
    ):
        super().__init__()
        # trainables
        
        
        
        
        z = torch.zeros((uv_params.shape[0],1), dtype=uv_params.dtype)
        plain_params_dict = {
            "uv_params":uv_params.to(device),# (N,2)
            "scales":scale_logits.to(device),# (N,3)
            "opacities":opacity_logits.to(device),# (N,)
            "z":z.to(device),#N,K-1,3
        }
        
        # appearance
        self._use_sh = sh0 is not None and shN is not None
        if self._use_sh:
            plain_params_dict["sh0"] = sh0.to(device)             # (N,1,3)
            plain_params_dict["shN"] = shN.to(device)             # (N,K-1,3)
        else:
            plain_params_dict["features"] = features.to(device)   # (N,F)
            plain_params_dict["colors"] = colors.to(device)       # (N,3) logits (like logit(rgb))

        # fixed buffers
        self.register_buffer("tri_ids", tri_ids.to(device))               # (N,)
        self.register_buffer("base_o", base_o.to(device))                 # (F,3)
        self.register_buffer("base_e1", base_e1.to(device))               # (F,3)
        self.register_buffer("base_e2", base_e2.to(device))               # (F,3)
        self.register_buffer("base_n", base_n.to(device))                 # (F,3)
        self.register_buffer("base_scale", base_scale.to(device))         # (F,3)
        
        self.params_dict = nn.ParameterDict(plain_params_dict)
        
        
    def __get_param_from_dict(self, key):
        if key in self.params_dict: return self.params_dict[key]
        return None
    
    @property
    def z(self): return self.__get_param_from_dict("z")
    @property
    def uv_params(self): return self.__get_param_from_dict("uv_params")
    @property
    def scale(self): return self.__get_param_from_dict("scales")
    @property
    def opacity(self): return self.__get_param_from_dict("opacities")
    @property
    def sh0(self): return self.__get_param_from_dict("sh0")
    @property
    def shN(self): return self.__get_param_from_dict("shN")
    
    @property
    def colors(self): return self.__get_param_from_dict("colors")
    @property
    def features(self): return self.__get_param_from_dict("features")
    
    
    def npoints(self) -> int: return len(self.uv_params)

    # --- world-space reconstructions ---
    def world_means(self) -> torch.Tensor:
        o  = self.base_o[self.tri_ids]      # (N,3)
        e1 = self.base_e1[self.tri_ids]     # (N,3)
        e2 = self.base_e2[self.tri_ids]     # (N,3)
        n  = self.base_n[self.tri_ids]      # (N,3)
        uv = self.uv_params                 # (N,2)
        return o + uv[:,0:1]*e1 + uv[:,1:2]*e2 + self.z * n

    def world_quats(self) -> torch.Tensor:
        e1 = self.base_e1[self.tri_ids]     # (N,3)
        e2 = self.base_e2[self.tri_ids]     # (N,3)
        n  = self.base_n[self.tri_ids]      # (N,3)
        R = torch.stack([e1, e2, n], dim=-1)  # (N,3,3)
        return rotation_matrix_to_quat(R)     # (N,4)

    def world_scales(self) -> torch.Tensor:
        # base_scale gives us a per-triangle "typical" tangent vs normal extent
        # exp(scale_logits) is a multiplicative factor
        bs = self.base_scale[self.tri_ids]                   # (N,3)
        return torch.exp(self.scale) * bs             # (N,3)

    def world_opacities(self) -> torch.Tensor:
        return torch.sigmoid(self.opacity)  # (N,1)

    def as_renderer_dict(self) -> dict:
        d = {
            "means":     self.world_means(),         # (N,3)
            "quats":     self.world_quats(),         # (N,4)
            "scales":    self.world_scales(),        # (N,3)
            "opacities": self.world_opacities(),        # logits, trainable
        }
        
        if self._use_sh:
            d["sh0"] = self.sh0        # (N,1,3)
            d["shN"] = self.shN        # (N,K-1,3)
        else:
            d["features"] = self.features  # (N,F)
            d["colors"]   = self.colors    # (N,3) logits
            
        return d
        
        
    def step_pre_backward(
        self,
        step, info,
        **kwargs,
    ):
        pass

    def step_post_backward(
        self,
        step, info,
        **kwargs,
    ):
        for s in self.schedulers.values(): s.step()
        pass
        
    def check_sanity(self):
        pass
        
    def create_optimizers(
    self,
    max_steps:int, 
    means_lr: float,
    scales_lr: float,
    opacities_lr: float,
    quats_lr: float,
    sh0_lr: float,
    shN_lr: float,
    batch_size: int,
    world_size: int,
    sparse_grad: bool = False,
    visible_adam: bool = False) -> Dict[str, torch.optim.Optimizer]:
        """
        For SurfaceSplats we don't optimize world means or quats directly.
        Instead:
          - "uv_params"   replaces "means"
          - "scale_logits" replaces "scales"
          - "opacity_logits" replaces "opacities"
          - implicit orientation comes from the triangle frame, so "quats" is not a param.
        We'll still accept quats_lr argument just to keep the same signature,
        but we won't use it (orientation is determined by surface frame).
        """
        
        # 6. build per-parameter optimizers with per-param LRs
        # same scaling rule you had: lr * sqrt(BS), eps / sqrt(BS)
        BS = batch_size * world_size
        if sparse_grad:
            optimizer_class = torch.optim.SparseAdam
        elif visible_adam:
            optimizer_class = SelectiveAdam  # must exist in your code
        else:
            optimizer_class = torch.optim.Adam
        # map param names -> (tensor, lr)
        param_specs = [
            ("means",      self.uv_params,      means_lr),
            ("scales",     self.scale,     scales_lr),
            ("opacities",  self.opacity,  opacities_lr),
            ("z",  self.z,  means_lr),
        ]

        if self._use_sh:
            param_specs += [
                ("sh0",     self.sh0,       sh0_lr),
                ("shN",     self.shN,       shN_lr),
            ]
        else:
            # feature path
            param_specs += [
                ("features", self.features, sh0_lr),
                ("colors",   self.colors,   sh0_lr),
            ]

        self.optimizers = {
            name: optimizer_class(
                [{
                    "params": [tensor],
                    "lr": lr * math.sqrt(BS),
                    "name": name,
                }],
                eps=1e-15 / math.sqrt(BS),
                betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
            )
            for (name, tensor, lr) in param_specs
        }

        self.schedulers = {}
        for k in self.optimizers:
            if k == "means":
                self.schedulers[k] = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizers[k], gamma=0.01 ** (1.0 / max_steps)
                )
                
        assert "means" in self.schedulers



def create_surface_splats_from_mesh(
    mesh_o3d: o3d.geometry.TriangleMesh,
    K: int,
    sh_degree: int,
    init_opacity: float,
    init_scale: float,
    feature_dim: Optional[int],
    device: str = "cuda",
)->SurfaceSplats:
    """
    Attach exactly K Gaussians per triangle of mesh_o3d.

    Steps:
    - compute per-triangle tangent frames (o,e1,e2,n)
    - for each triangle, sample K barycentric points
    - convert those points to (u,v) coords in that triangle's local (e1,e2) basis
    - assign color from vertex colors (or normals fallback)
    - set per-triangle base scale from triangle edge lengths
    - build a SurfaceSplats module
    """

    # --- gather mesh data ---
    verts = np.asarray(mesh_o3d.vertices, dtype=np.float32)   # (V,3)
    faces = np.asarray(mesh_o3d.triangles, dtype=np.int64)    # (F,3)

    if mesh_o3d.has_vertex_colors():
        vcols = np.asarray(mesh_o3d.vertex_colors, dtype=np.float32)  # (V,3) in [0,1]
    else:
        mesh_o3d.compute_vertex_normals()
        norms = np.asarray(mesh_o3d.vertex_normals, dtype=np.float32)
        vcols = 0.5 * (norms + 1.0)

    V0 = verts[faces[:,0]]  # (F,3)
    V1 = verts[faces[:,1]]  # (F,3)
    V2 = verts[faces[:,2]]  # (F,3)

    C0 = vcols[faces[:,0]]  # (F,3)
    C1 = vcols[faces[:,1]]  # (F,3)
    C2 = vcols[faces[:,2]]  # (F,3)

    Fcount = faces.shape[0]
    N = Fcount * K  # total splats

    # --- build per-triangle tangent frame ---
    # tangent e1 = (V1 - V0) normalized
    e1_all = V1 - V0
    e1_norm = np.linalg.norm(e1_all, axis=1, keepdims=True) + 1e-9
    e1_all_norm = e1_all / e1_norm  # (F,3)

    # temporary t2 = (V2 - V0), make e2 orthonormal to e1
    t2 = V2 - V0
    proj = (t2 * e1_all_norm).sum(axis=1, keepdims=True) * e1_all_norm
    e2_all = t2 - proj
    e2_norm = np.linalg.norm(e2_all, axis=1, keepdims=True) + 1e-9
    e2_all_norm = e2_all / e2_norm  # (F,3)

    # normal
    n_all = np.cross(e1_all_norm, e2_all_norm)
    n_norm = np.linalg.norm(n_all, axis=1, keepdims=True) + 1e-9
    n_all_norm = n_all / n_norm     # (F,3)

    # --- per-triangle base scale ---
    # We'll use mean edge length as a base tangential extent,
    # and shrink normal axis so splats stay "thin" against surface.
    edge0 = np.linalg.norm(V1 - V0, axis=1)  # (F,)
    edge1 = np.linalg.norm(V2 - V1, axis=1)
    edge2 = np.linalg.norm(V0 - V2, axis=1)
    mean_edge = (edge0 + edge1 + edge2) / 3.0  # (F,)

    # base scale per tri, anisotropic: [tangent,tangent,normal]
    # init_scale is a global multiplicative factor.
    base_scale = np.stack([
        mean_edge * init_scale,
        mean_edge * init_scale,
        mean_edge * init_scale * 0.05,   # squash along normal
    ], axis=1).astype(np.float32)  # (F,3)

    # --- sample K points per triangle uniformly ---
    # Draw K barycentric samples per face.
    # For uniform sampling in a triangle: sample u,v ~ U(0,1), if u+v>1 flip.
    u = np.random.rand(Fcount, K, 1).astype(np.float32)
    v = np.random.rand(Fcount, K, 1).astype(np.float32)
    mask = (u + v > 1.0)
    v[mask] = 1.0 - v[mask]
    u[mask] = 1.0 - u[mask]
    w = 1.0 - u - v  # weights for V0 (w), V1 (u), V2 (v)

    # splat world positions (for sanity / debug)
    P = w * V0[:,None,:] + u * V1[:,None,:] + v * V2[:,None,:]    # (F,K,3)

    # per-splat color via barycentric blend
    Col = w * C0[:,None,:] + u * C1[:,None,:] + v * C2[:,None,:]  # (F,K,3)

    # convert world P to (a,b) in triangle tangent frame:
    # rel = P - V0
    # a = dot(rel, e1), b = dot(rel, e2)
    rel = P - V0[:,None,:]                          # (F,K,3)
    a = (rel * e1_all_norm[:,None,:]).sum(axis=2, keepdims=True)  # (F,K,1)
    b = (rel * e2_all_norm[:,None,:]).sum(axis=2, keepdims=True)  # (F,K,1)
    uv = np.concatenate([a,b], axis=2)  # (F,K,2)

    # Flatten across triangles
    uv_flat   = uv.reshape(-1, 2)            # (N,2)
    cols_flat = Col.reshape(-1, 3)           # (N,3)
    tri_ids   = np.repeat(np.arange(Fcount, dtype=np.int64), K)  # (N,)

    # --- build SH coefficients for color ---
    # SH layout: (N, (sh_degree+1)^2, 3)
    
    
    if feature_dim is None:
        Ksh = (sh_degree + 1) ** 2
        colors_sh = np.zeros((N, Ksh, 3), dtype=np.float32)
        colors_sh[:, 0, :] = rgb_to_sh(torch.from_numpy(cols_flat)).cpu().numpy()
        # remaining bands start at 0
        sh0 = colors_sh[:, :1, :]        # (N,1,3)
        shN = colors_sh[:, 1:, :]        # (N,Ksh-1,3)
        features_param = None
        color_logits_param = None
        
        sh0 = torch.from_numpy(sh0).float()
        shN = torch.from_numpy(shN).float()
    else:
        # Learnable per-splat features for an appearance network.
        sh0 = None
        shN = None
        features_param = torch.rand((N, feature_dim), dtype=torch.float32)
        # Store color as logits (inverse-sigmoid style) just like your original code
        color_logits_param = torch.logit(torch.from_numpy(cols_flat).cpu().clamp(1e-4, 1-1e-4))  # (N,3)

    # --- opacity / scale logits init ---
    # opacity_logits = logit(init_opacity)
    opacity_init = torch.full((N,), init_opacity, dtype=torch.float32)
    opacity_logits = torch.logit(opacity_init.clamp(1e-4, 1-1e-4))  # (N,)

    # scale_logits: start at zeros so exp(scale_logits)=1 → scale = base_scale[tri]
    scale_logits_init = torch.zeros((N,3), dtype=torch.float32)

    # --- pack per-triangle bases as torch tensors ---
    base_o_t  = torch.from_numpy(V0).float()               # (F,3) origin v0
    base_e1_t = torch.from_numpy(e1_all_norm).float()      # (F,3)
    base_e2_t = torch.from_numpy(e2_all_norm).float()      # (F,3)
    base_n_t  = torch.from_numpy(n_all_norm).float()       # (F,3)
    base_scale_t = torch.from_numpy(base_scale).float()    # (F,3)

    # --- create SurfaceSplats instance ---
    splats = SurfaceSplats(
        uv_params        = torch.from_numpy(uv_flat).float(),          # (N,2)
        tri_ids          = torch.from_numpy(tri_ids).long(),           # (N,)
        base_o           = base_o_t,                                   # (F,3)
        base_e1          = base_e1_t,                                  # (F,3)
        base_e2          = base_e2_t,                                  # (F,3)
        base_n           = base_n_t,                                   # (F,3)
        base_scale       = base_scale_t,                               # (F,3)
        scale_logits     = scale_logits_init,                          # (N,3)
        opacity_logits   = opacity_logits,                             # (N,)
        sh0              = sh0,                                        # (N,1,3)
        shN              = shN,                                        # (N,Ksh-1,3)
        features         = features_param,                             # (N, Kfeatures)
        colors           = color_logits_param,                         # (N,3)
        device           = device,
    )

    return splats



def create_free_splats_from_points(
    points: torch.Tensor,           # (N,3) world coords
    rgbs: torch.Tensor,             # (N,3) in [0,1]
    sh_degree: int,
    init_opacity: float,
    init_scale: float,
    feature_dim: Optional[int],
    device: str = "cuda",
) -> FreeSplats:
    """
    Builds FreeSplats + per-parameter optimizers,
    mirroring your original create_splats_with_optimizers behavior.
    """

    # 1. estimate initial scale per splat
    # knn(...) must return distances (N,k). We do same trick as in your code
    dist_k = knn(points, 4)                    # (N,4)
    dist2_avg = (dist_k[:, 1:] ** 2).mean(-1)  # ignore self (index 0), avg of neighbors^2
    dist_avg = torch.sqrt(dist2_avg + 1e-12)
    scales0 = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1,3)  # (N,3)

    N = points.shape[0]

    # 2. init quats randomly
    quats0 = torch.rand((N,4), dtype=torch.float32)

    # 3. init opacity logits
    opacity_init = torch.full((N,), init_opacity, dtype=torch.float32)
    opacity_logits0 = torch.logit(opacity_init.clamp(1e-4, 1-1e-4))  # (N,)

    # 4. SH colors or feature/colors
    if feature_dim is None:
        Ksh = (sh_degree + 1) ** 2
        sh_full = torch.zeros((N, Ksh, 3), dtype=torch.float32)
        sh_full[:,0,:] = rgb_to_sh(rgbs)  # DC term
        sh0 = sh_full[:, :1, :]          # (N,1,3)
        shN = sh_full[:, 1:, :]          # (N,Ksh-1,3)

        features_param = None
        color_logits_param = None
    else:
        # Learnable per-splat features for an appearance network.
        sh0 = None
        shN = None
        features_param = torch.rand((N, feature_dim), dtype=torch.float32)
        # Store color as logits (inverse-sigmoid style) just like your original code
        color_logits_param = torch.logit(rgbs.clamp(1e-4, 1-1e-4))  # (N,3)

    # 5. wrap in FreeSplats module
    free_splats = FreeSplats(
        means      = points,
        scales     = scales0,
        quats      = quats0,
        opacities  = opacity_logits0,
        sh0        = sh0,
        shN        = shN,
        features   = features_param,
        colors     = color_logits_param,
        device     = device,
    )
    return free_splats
