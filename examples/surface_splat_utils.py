import pytorch3d
import pytorch3d.ops
import pytorch3d.loss
import pytorch3d.structures
import pytorch3d.utils
import torch
import open3d as o3d
import numpy as np

def sample_barycentric(N:int)->torch.Tensor:
    P = torch.distributions.Dirichlet(torch.tensor([1.0,1.0,1.0]).float())
    D = P.sample((N,1))
    return D.float()
def barycentric_to_parameter_space(X:torch.Tensor)->torch.Tensor:
    return torch.log(X)
def barycentric_from_parameter_space(X:torch.Tensor, dim=-1)->torch.Tensor:
    return torch.softmax(X,dim=-1)
def points_from_barycentric(triangle_points:torch.Tensor, weights:torch.Tensor)->torch.Tensor:
    return torch.einsum("kij,kli->kj", triangle_points, weights)

#we want smooth tangent and bitangents to avoid discontinuitis in parameter space
def onb_from_normal_frisvad(n: torch.Tensor):
    # n: (..., 3), assumed unit
    sign = torch.sign(n[..., 2])
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    a = -1.0 / (sign + n[..., 2])
    b = n[..., 0] * n[..., 1] * a
    t = torch.stack([
        1.0 + sign * n[..., 0] ** 2 * a,
        sign * b,
        -sign * n[..., 0]
    ], dim=-1)
    b = torch.stack([
        b,
        sign + n[..., 1] ** 2 * a,
        -n[..., 1]
    ], dim=-1)
    t = torch.nn.functional.normalize(t, dim=-1)
    b = torch.nn.functional.normalize(b, dim=-1)
    return t, b

def compute_tri_stats(triangle_points:torch.Tensor, meshes: pytorch3d.structures.Meshes, compute_normal:bool=True, compute_edge_lengths:bool=True, compute_max_edge_length:bool=True)->dict:
    u = triangle_points[...,0,:]
    v = triangle_points[...,1,:]
    w = triangle_points[...,2,:]

    stats = {
         "l1**2": None,
         "l2**2": None,
         "l3**2": None,
         "lmax**2": None,
         "normal": None,
        }

    if compute_normal:
        stats["normal"] = meshes.faces_normals_list()[0]
    if compute_edge_lengths or compute_max_edge_length:
        e1 = (v-u)
        e2 = (w-u)
        e3 = (w-v)

        e1_2 = (e1*e1).sum(-1)
        e2_2 = (e2*e2).sum(-1)
        e3_2 = (e3*e3).sum(-1)
        stats["l1**2"] = e1_2
        stats["l2**2"] = e2_2
        stats["l3**2"] = e3_2
    if compute_max_edge_length:
        max_edge_len2 = torch.maximum(torch.maximum(e1_2, e2_2),e3_2)
        stats["lmax**2"] = max_edge_len2
    return stats
    