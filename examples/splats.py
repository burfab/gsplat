import math
from typing import Optional, Dict, Tuple
from gsplat.strategy import ops as strategy_ops

import numpy as np

import torch
import torch.nn as nn
from pytorch3d import transforms as torchtransforms

import open3d as o3d

from gsplat.optimizers import SelectiveAdam
from utils import rgb_to_sh, knn

from gsplat.strategy import MCMCStrategy, Strategy
import surface_splat_utils
import pytorch3d.structures
import pytorch3d.loss
import pytorch3d.ops

# assumes you already have rgb_to_sh and knn available in scope
# rgb_to_sh: (N,3) rgb in [0,1] -> (N,3) SH DC coeffs
# knn(x,k):  (N,3), returns (N,k) distances to k nearest neighbors (just like your current code expects)


class Splats(nn.Module):
    def fn_opacities_activation(self,x): return torch.sigmoid(x)
    def fn_opacities_inverse_activation(self,x): return torch.logit(x)
    def fn_scales_activation(self,x): return torch.exp(x)
    def fn_scales_inverse_activation(self,x): return torch.log(x)
    
    def type_str(self) -> str:
        pass
    
    def prepare_render(self):
        pass
    def pre_optimizers_step(self, step, *args, **kwargs):
        pass
    def post_optimizers_step(self, step, *args, **kwargs):
        pass
    def step_pre_backward(
        self,
        *args,
        **kwargs,
    ):
        pass
    
    def on_loss_grad_computed(
        self,
        step,
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


    def post_optimizers_step(self, step, *args, **kwargs):
        for s in self.schedulers.values(): s.step()

    def step_post_backward(
        self,
        step, info,
        **kwargs,
    ):
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
                          batch_size: int, 
                          sparse_grad: bool = False, visible_adam: bool = False):
        # 6. build per-parameter optimizers with per-param LRs
        # same scaling rule you had: lr * sqrt(BS), eps / sqrt(BS)
        BS = batch_size 
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
        bary_logits: torch.Tensor,         # (N,3)
        tri_ids: torch.Tensor,             # (N,)
        triangles : torch.Tensor,          # (F,3)
        vertices: torch.Tensor,            # (V,3)
        rotations: torch.Tensor,            # (N,4)
        scale: torch.Tensor,        # (N,3)
        opacity: torch.Tensor,      # (N,)
        sh0: torch.Tensor,                 # (N,1,3)
        shN: torch.Tensor,                 # (N,K-1,3)
        features: torch.Tensor,
        colors: torch.Tensor,
        device: str = "cuda",
    ):
        super().__init__()
        # trainables
        
        self.render_buffer = {}
        self.strategy_state = {}
        
        
        self.last_geometry_update = 0
        self.initial_learning_rates = {}
        self.schedulers = {}
        self.optimizers = {}
        
        self._geometry_frozen = False
        
        self.register_buffer("tri_ids", tri_ids.to(device))               # (N,)
        self.register_buffer("triangles", triangles.to(device))               # (N,)
        
        normal_shifts = torch.zeros_like(opacity)
        
        plain_params_dict = {
            "vertices":vertices.to(device),# (V,3)
            "bary_logits":bary_logits.to(device),# (N,3)
            "normal_shifts":normal_shifts.to(device),# (N,3)
            "scales": self.fn_scales_inverse_activation(scale.to(device)),# (N,3)
            "opacities": self.fn_opacities_inverse_activation(opacity.to(device)),# (N,)
            "rotations": rotations.to(device),# (N,4)
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
        
        self.params_dict = nn.ParameterDict(plain_params_dict)
        
        # Optional scene scale for absolute thickness (median keeps it robust)
        
        #compute stuff needed for later
        with torch.no_grad():
            meshes = pytorch3d.structures.Meshes(self.vertices.unsqueeze(0), self.triangles.unsqueeze(0))
            self.scene_scale = meshes.get_bounding_boxes()[0]
            edge_lenghts = torch.sqrt(surface_splat_utils.get_edge_lengths(meshes, self.triangles))
            self.edge_len_guideline = torch.median(edge_lenghts)
            self.prepare_render()
        
        
        
    def __get_param_from_dict(self, key):
        if key in self.params_dict: return self.params_dict[key]
        return None
    
    @property
    def bary_logits(self): return self.__get_param_from_dict("bary_logits")
    @property
    def normal_shifts(self): return self.__get_param_from_dict("normal_shifts")
    @property
    def vertices(self): return self.__get_param_from_dict("vertices")
    @property
    def rotations(self): return self.__get_param_from_dict("rotations")
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
    
    
    def prepare_render(self):
        meshes = pytorch3d.structures.Meshes(self.vertices[None,...], self.triangles[None,...])
        splat_edge_lengths = surface_splat_utils.get_edge_lengths(meshes, self.tri_ids)
        #splat_edge_lengths_sorted = torch.sort(splat_edge_lengths,dim=-1,descending=True).values[:,:2]
        
        tri_points = self.vertices[self.triangles]
        tri_normals = meshes.verts_normals_list()[0][self.triangles]
        
        tri_stats = surface_splat_utils.compute_tri_stats(tri_points, meshes, True, True)
        bary = surface_splat_utils.barycentric_from_reduced_parameter_space(self.bary_logits)
        splat_mesh_normals = torch.nn.functional.normalize(surface_splat_utils.points_from_barycentric(tri_normals[self.tri_ids], bary));
        splat_means = surface_splat_utils.points_from_barycentric(tri_points[self.tri_ids], bary)# + self.normal_shifts.unsqueeze(-1) * splat_mesh_normals
        
        splat_opacities = self.fn_opacities_activation(self.opacity)
        splat_scales = self.fn_scales_activation(self.scale).minimum(splat_edge_lengths)#.clamp_max(self.scene_scale.max() * 1e-2)
        splat_rotations = self.rotations
        
        
        self.render_buffer = { 
                              "meshes": meshes,
                              "splat_edge_lenghts": splat_edge_lengths,
                              "tri_points": tri_points,
                              "tri_stats": tri_stats,
                              "bary": bary,
                              "splat_mesh_normals": splat_mesh_normals,
                              "splat_means": splat_means,
                              "splat_opacities": splat_opacities,
                              "splat_scales": splat_scales,
                              "splat_rotations": splat_rotations,
                              }
    
    def normal_consistency_loss(self, mode="default", **kwargs):
        return surface_splat_utils.mesh_normal_consistency_with_modes(self.render_buffer["meshes"], mode, **kwargs)
    def laplacian_loss(self, weight_type="uniform"): #cot weights seem to promote slivers/fans
        return pytorch3d.loss.mesh_laplacian_smoothing(self.render_buffer["meshes"],weight_type)
    
    def edge_loss(self):
        return pytorch3d.loss.mesh_edge_loss(self.render_buffer["meshes"],self.edge_len_guideline)
    
    def regularization_loss(self, lambda_normal_shifts = 10.0, lambda_scale_loss = 10.0, lambda_tilt_loss = 1.0):
        loss = torch.zeros(1).requires_grad_(True).to(self.vertices.device)
        if lambda_normal_shifts > 0:
            normal_shift_loss = self.normal_shifts.abs().mean()
            loss = loss + normal_shift_loss * lambda_normal_shifts
        if lambda_scale_loss > 0:
            splat_edge_lengths_sorted = torch.sort(self.render_buffer["splat_edge_lenghts"].detach(),dim=-1,descending=True).values[:,:2]
            ri = (self.world_scales()[...,:2]-splat_edge_lengths_sorted).max(dim=-1).values
            scale_loss = torch.maximum(ri,torch.zeros_like(ri)).mean()
            loss =loss + scale_loss * lambda_scale_loss 
        if lambda_tilt_loss > 0:
            ri = ((1.0-self.splat_cos_angle_to_face())) -  0.1
            tilt_loss = torch.maximum(ri,torch.zeros_like(ri)).mean()
            loss = loss + tilt_loss * lambda_tilt_loss
        
        return loss
    
        
    
    
    def find_nn_splat(self, tri_pts, weights):
        query_pts = surface_splat_utils.points_from_barycentric(tri_pts,weights)
        _, nn_idx, _ = pytorch3d.ops.knn_points(query_pts[None,...], self.world_means()[None,...], K=1)
        assert nn_idx.shape[0] == 1 and nn_idx.shape[2] == 1
        nn_idx = nn_idx[0,:,0]
        return nn_idx
    
    def get_splat_params_from_indices(self, indices):
        update = {}
        update["scales"] = self.scale[indices[:]]
        update["rotations"] = self.rotations[indices]
        update["opacities"] = self.opacity[indices]
        update["bary_logits"] = self.bary_logits[indices]
        update["normal_shifts"] = self.normal_shifts[indices]
        if self._use_sh:
            update["sh0"] = self.sh0[indices[:]]
            update["shN"] = self.shN[indices[:]]
        else:
            update["features"]= self.features[indices[:]]
            update["colors"]= self.colors[indices[:]]
        return update
    
    
    @torch.no_grad()
    def face_error_proj_on_normal(self):
        proj_grad = self.strategy_state["vertices_grad"] * self.render_buffer["meshes"].verts_normals_list()[0]
        rij = torch.linalg.norm(proj_grad, dim=-1)
        ri = rij[self.triangles].mean(dim=-1)
        return ri
    
    def do_subdivide(self, step):
        return step % 2000 == 0 and step >= 1500 and step <= 20_000
    
    def splat_cos_angle_to_face(self):
        quats = self.world_quats()
        mesh_normals = self.render_buffer["splat_mesh_normals"]
        splat_normals = torchtransforms.quaternion_to_matrix(quats)[...,2]
        cosa = (splat_normals * mesh_normals).sum(-1)
        return cosa
    
    def face_error_top_q_mask(self, q:float, th_min:float = 0.06):
        k = q * self.triangles.shape[0]
        k = int(min(self.triangles.shape[0]-1, k))
        if k <= 0: return None
        """
        with torch.enable_grad():
            V = self.vertices.clone().detach().requires_grad_(True)
            T = self.triangles.clone()
            M = pytorch3d.structures.Meshes(V[None,...], T[None,...])
            loss = surface_splat_utils.mesh_normal_consistency_with_modes(M)
            loss.backward()
            ri = torch.linalg.norm(V.grad[T], dim=-1).sum(-1)
        """
        
        
        #ri = self.face_error_proj_on_normal()
        
        ri = torch.zeros(len(self.triangles),dtype=torch.float32,device="cpu")
        cnts = torch.zeros(len(self.triangles),dtype=torch.float32,device="cpu")
        inds = self.tri_ids.detach().cpu()
        cosa = (1.0-self.splat_cos_angle_to_face()).cpu()
        for i in range(len(inds)):
            ri[inds[i]] += cosa[i]
            cnts+=1
        ri/=cnts
        adaptive_th = max(ri.sort(descending=True)[0][k], th_min)
        #NV = self.render_buffer["meshes"].verts_normals_list()[0]
        #NF = self.render_buffer["meshes"].faces_normals_list()[0]
        #cosa = (NV[self.triangles] * NF[...,None,:]).sum(-1).mean(-1)
        #ri = 1.0-(cosa + 1.0)*0.5
        #adaptive_th = max(ri.sort(descending=True)[0][k], 0.02)
        mask = ri >= adaptive_th
        return mask
    



    
    @torch.no_grad()
    def subdivide_and_create_splats(self,q:float,k:int, face_error_min_th:float) -> bool:
        mask_split = self.face_error_top_q_mask(q,face_error_min_th)
        if mask_split is None: return False
        device = self.vertices.device
        
        
        #remove all splats on touched faces
        #create k new splats per face, initialize by nearest splat 
        vertices_new, triangles_new,mask_untouched_faces = surface_splat_utils.masked_midpoint_subdivide(self.vertices, self.triangles, mask_split)
        triangle_index_shift = (~mask_untouched_faces).cumsum(0) #to subtract from old triangle indices, as they will be shifted
        
        
        cnt_verts_old = len(vertices_new[0]); cnt_verts_new = len(vertices_new[1]);
        cnt_tris_old = len(triangles_new[0]); cnt_tris_new = len(triangles_new[1]);
        
        vertices_new = torch.cat(vertices_new)
        triangles_new = torch.cat(triangles_new)
        
        
        
        assert len(triangles_new) >= len(self.triangles)
        
        new_vertices_mask = torch.ones(len(vertices_new), dtype=bool, device=device)
        new_vertices_mask[:cnt_verts_old] = False
        
        
        
        
        #the splats to keep
        untouched_splats_mask = (mask_untouched_faces)[self.tri_ids]
        touched_splats_mask = (~mask_untouched_faces)[self.tri_ids]
        
        #find closest face to query pts
        query = pytorch3d.structures.Pointclouds(self.world_means()[None,touched_splats_mask,...])
        mesh_new = pytorch3d.structures.Meshes(vertices_new[None,...],triangles_new[None,...])
        dist_to_face, ind_face = surface_splat_utils.find_closest_face(mesh_new, query,1e-8)
        reassign_gaussians_mask = touched_splats_mask.clone()
        reassign_gaussians_mask[touched_splats_mask] = dist_to_face < self.edge_len_guideline * 1e-3
        
        update_reassign_gaussians = {
            }
        update_reassign_gaussians["tri_ids"] = ind_face
        if not torch.any(reassign_gaussians_mask): update_reassign_gaussians = {}
        keep_splats_mask = torch.logical_or(untouched_splats_mask, reassign_gaussians_mask)
        
        tri_ids_to_new_geometry = self.tri_ids-triangle_index_shift[self.tri_ids]
        if "tri_ids" in update_reassign_gaussians:
            tri_ids_to_new_geometry[reassign_gaussians_mask] = update_reassign_gaussians["tri_ids"]
        tri_ids_to_new_geometry = tri_ids_to_new_geometry[keep_splats_mask]
        
        no_gaussians_mask = torch.ones(len(triangles_new), dtype=bool, device=device)
        no_gaussians_mask[tri_ids_to_new_geometry] = False
        
        tri_ids_created = torch.where(no_gaussians_mask)[0].repeat_interleave(k).to(device)
        bary_created = surface_splat_utils.sample_barycentric(len(tri_ids_created)).to(device)
        number_created_splats = len(tri_ids_created)
        
        #find nearest neighbour splat for initialization, but use bary logits from random initialization
        nn_idx = self.find_nn_splat(vertices_new[triangles_new[tri_ids_created]],bary_created)
        update = self.get_splat_params_from_indices(nn_idx)
        update["bary_logits"] = surface_splat_utils.barycentric_to_reduced_parameter_space(bary_created)
        
        
        
        tri_ids_new = torch.cat((tri_ids_to_new_geometry, tri_ids_created.to(self.tri_ids.device)))
        
        def param_fn(name: str, p: torch.Tensor) -> torch.Tensor:
            if name in update_reassign_gaussians: p[reassign_gaussians_mask] = update_reassign_gaussians[name]
            p_new = torch.cat((p[keep_splats_mask], update[name]))
            return torch.nn.Parameter(p_new, requires_grad=p.requires_grad)

        def optimizer_fn(name:str, key: str, v: torch.Tensor) -> torch.Tensor:
            if name in update_reassign_gaussians: v[reassign_gaussians_mask] = 0
            v_new = torch.zeros((number_created_splats, *v.shape[1:]), device=device,dtype=v.dtype)
            return torch.cat((v[keep_splats_mask], v_new))
        def param_fn_vertices(name: str, p: torch.Tensor) -> torch.Tensor:
            return torch.nn.Parameter(vertices_new, requires_grad=p.requires_grad)
        def optimizer_fn_vertices(name:str, key: str, v: torch.Tensor) -> torch.Tensor:
            v_new = torch.zeros((max(0,len(vertices_new)-len(v)), *v.shape[1:]), device=device,dtype=v.dtype)
            return torch.cat((v*0.0,v_new))
        def optimizer_fn_step_vertices(name:str, key: str, v: torch.Tensor) -> torch.Tensor:
            v[...] = 0.0
            return v
        
        strategy_ops._update_param_with_optimizer(param_fn, optimizer_fn, self.params_dict, self.optimizers,update.keys())
        strategy_ops._update_param_with_optimizer(param_fn_vertices, optimizer_fn_vertices, self.params_dict, self.optimizers,set(["vertices"]),optimizer_fn_step_vertices)
        
        self.schedulers["vertices"].last_epoch = -1
        self.schedulers["vertices"]._last_lr = self.schedulers["vertices"].base_lrs  
        
        self.register_buffer("tri_ids", tri_ids_new)
        self.register_buffer("triangles", triangles_new.to(device))
        
        assert not (self.tri_ids < 0).any()
        
        self.prepare_render()
        return True
    
    def npoints(self) -> int: return len(self.tri_ids)

    # --- world-space reconstructions ---
    def world_means(self) -> torch.Tensor: return self.render_buffer["splat_means"]
    
    def world_quats(self) -> torch.Tensor: return self.render_buffer["splat_rotations"]

    def world_scales(self) -> torch.Tensor: return self.render_buffer["splat_scales"] 

    def world_opacities(self) -> torch.Tensor: return self.render_buffer["splat_opacities"]

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
        
        
    def on_loss_grad_computed(
        self,
        step,
        **kwargs,
    ):
        if not "vertices_grad" in self.strategy_state:
            self.strategy_state["vertices_grad"] = torch.zeros_like(self.vertices,requires_grad=False)
            self.strategy_state["vertices_grad_collected_cnt"] = 0
            
        exp_decay = 0.03
        with torch.no_grad():
            self.strategy_state["vertices_grad"] = (torch.zeros_like(self.vertices,requires_grad=False) if self.vertices.grad is None else self.vertices.grad) * (exp_decay) + self.strategy_state["vertices_grad"] * (1.0-exp_decay)
            self.strategy_state["vertices_grad_collected_cnt"] += 1
        
    def step_pre_backward(
        self,
        step, info,
        **kwargs,
    ):
        pass
    
    def pre_optimizers_step(self, step):
        if self._geometry_frozen:
            if self.vertices.grad is not None: self.vertices.grad.zero_() 
        else:
            if self.vertices.grad is not None: 
                torch.nn.utils.clip_grad_norm_(
                                [self.vertices], 1#self.edge_len_guideline
                            )
            
    def post_optimizers_step(self, step):
        for key in self.schedulers: 
            if self._geometry_frozen and key == "vertices": continue
            self.schedulers[key].step()
            

    def fix_geometry(self, val: bool):
        if val == self._geometry_frozen: return
        opt = self.optimizers["vertices"]

        if val:
            self._geometry_frozen = True
            opt.param_groups[0]["lr"] = 0.0
        else:
            self._geometry_frozen = False
            opt.param_groups[0]["lr"] = self.initial_learning_rates["vertices"]

    def step_post_backward(
        self,
        step, info, subdivision_options = {},
        **kwargs,
    ):
        if self.do_subdivide(step): 
            torch.save(self.triangles, "/tmp/triangles.pyt");
            torch.save(self.vertices, "/tmp/vertices.pyt") 
            q = subdivision_options.get("q", 0.02) 
            k = subdivision_options.get("splats_per_tri", 1) 
            face_error_min_th = subdivision_options.get("face_error_min_th", 0.001) 
            if self.subdivide_and_create_splats(q,k, face_error_min_th):
                self.last_geometry_update = step
            self.strategy_state.pop("vertices_grad")
            self.strategy_state.pop("vertices_grad_collected_cnt")
            torch.cuda.empty_cache()
            torch.save(self.triangles, "/tmp/triangles_subdived.pyt");
            torch.save(self.vertices, "/tmp/vertices_subdived.pyt") 
            
        
    def check_sanity(self):
        pass
        
    def create_optimizers(
    self,
    max_steps:int, 
    means_lr: float,
    bary_lr: float,
    scales_lr: float,
    opacities_lr: float,
    quats_lr: float,
    sh0_lr: float,
    shN_lr: float,
    batch_size: int,
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
        BS = batch_size 
        # map param names -> (tensor, lr)
        param_specs = [
            ("vertices",      self.vertices, means_lr),
            ("bary_logits",      self.bary_logits, bary_lr),
            ("scales",     self.scale,     scales_lr),
            ("opacities",  self.opacity,  opacities_lr),
            ("normal_shifts",  self.normal_shifts,  means_lr),
            ("rotations",  self.rotations,  quats_lr),
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
            
        def get_optimizer_class(param_name):
            if param_name == "vertices": return torch.optim.Adam
            if sparse_grad:
                optimizer_class = torch.optim.SparseAdam
            elif visible_adam:
                optimizer_class = SelectiveAdam  # must exist in your code
            else:
                optimizer_class = torch.optim.Adam
            return optimizer_class

        assert len(self.initial_learning_rates) == 0 and len(self.schedulers) == 0 and len(self.optimizers) == 0
        for (name,tensor,lr) in param_specs: 
            self.initial_learning_rates[name] = lr * math.sqrt(BS)
            if name != "vertices" or True:
                self.optimizers[name] = get_optimizer_class(name)(
                    [{
                        "params": [tensor],
                        "lr": self.initial_learning_rates[name],
                        "name": name,
                    }],
                    eps=1e-15 / math.sqrt(BS),
                    betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
                )
            else: self.optimizers[name] = torch.optim.SGD([tensor], lr=self.initial_learning_rates[name])
        
        self.schedulers["vertices"] = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizers["vertices"], gamma=0.01 ** (1.0 / max_steps)
        )
            
        assert "vertices" in self.schedulers



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
    mesh_o3d = mesh_o3d.remove_unreferenced_vertices()

    verts = torch.from_numpy(np.asarray(mesh_o3d.vertices, dtype=np.float32))   # (V,3)
    faces = torch.from_numpy(np.asarray(mesh_o3d.triangles, dtype=np.int64))   # (F,3)
    mesh = pytorch3d.structures.Meshes(verts[None,...], faces[None,...])
    normals = mesh.verts_normals_list()[0]

    if mesh_o3d.has_vertex_colors():
        vcols = torch.from_numpy(np.asarray(mesh_o3d.vertex_colors, dtype=np.float32)).float()  # (V,3) in [0,1]
    else:
        vcols = 0.5 * (normals + 1.0) # random colors

    Fcount = faces.shape[0]
    N_splats = Fcount * K
    splat_faces = torch.arange(0, Fcount, dtype=torch.int64)
    splat_faces = torch.repeat_interleave(splat_faces, K)
    
    
    assert splat_faces.shape[0] == N_splats
    
    weights = surface_splat_utils.sample_barycentric(N_splats)
    splat_means = surface_splat_utils.points_from_barycentric(verts[faces[splat_faces]], weights)
    splat_mesh_normals = torch.nn.functional.normalize(surface_splat_utils.points_from_barycentric(normals[faces[splat_faces]], weights))
    
    colors = surface_splat_utils.points_from_barycentric(vcols[faces[splat_faces]], weights)
    splat_quats = surface_splat_utils.quaternion_from_normal(splat_mesh_normals)
    splat_edge_lengths = surface_splat_utils.get_edge_lengths(mesh, splat_faces)
    splat_edge_lengths_sorted = torch.sort(splat_edge_lengths,dim=-1,descending=True).values[:,:2]
    
    
    # --- build SH coefficients for color ---
    # SH layout: (N, (sh_degree+1)^2, 3)
    
    
    if feature_dim is None:
        Ksh = (sh_degree + 1) ** 2
        colors_sh = np.zeros((N_splats, Ksh, 3), dtype=np.float32)
        colors_sh[:, 0, :] = rgb_to_sh(colors).cpu().numpy()
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
        features_param = torch.rand((N_splats, feature_dim), dtype=torch.float32)
        # Store color as logits (inverse-sigmoid style) just like your original code
        color_logits_param = torch.logit(colors.cpu().clamp(1e-4, 1-1e-4))  # (N,3)

    # --- opacity / scale logits init ---
    splat_opacities = torch.full((N_splats,), init_opacity, dtype=torch.float32).clamp(1e-4,1-1e-4)
    splat_scales = torch.hstack((splat_edge_lengths_sorted,torch.full((N_splats,1),1e-7, dtype=torch.float32)))
    
    bary_logits = surface_splat_utils.barycentric_to_reduced_parameter_space(weights)


    # --- create SurfaceSplats instance ---
    splats = SurfaceSplats(
        bary_logits      = bary_logits,          
        tri_ids          = splat_faces,          
        triangles        = faces,                
        vertices         = verts,                                   
        rotations        = splat_quats,                             
        scale = splat_scales,                       
        opacity = splat_opacities,                          
        sh0              = sh0,                                     
        shN              = shN,                                     
        features         = features_param,                          
        colors           = color_logits_param,                      
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
