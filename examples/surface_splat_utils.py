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

@torch.no_grad
def compute_tri_stats(triangle_points:torch.Tensor, meshes: pytorch3d.structures.Meshes, compute_edge_lengths:bool=True, compute_max_edge_length:bool=True)->dict:
    u = triangle_points[...,0,:]
    v = triangle_points[...,1,:]
    w = triangle_points[...,2,:]

    stats = {
         "l1**2": None,
         "l2**2": None,
         "l3**2": None,
         "lmax**2": None,
        }
        
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


def midpoint_subdivide(vertices:torch.Tensor, triangles:torch.Tensor, mask_split:torch.Tensor):
    masked_faces = triangles[mask_split]
    unmasked_faces = triangles[~mask_split]
    #just parts of triangles, but same vertices
    mesh_new = pytorch3d.structures.Meshes(vertices[None,...],masked_faces[None,...])
    mesh_new = pytorch3d.ops.SubdivideMeshes()(mesh_new)
    

    V_new = mesh_new.verts_list()[0]
    T_new = torch.cat((unmasked_faces,mesh_new.faces_list()[0])) 
    return V_new, T_new
    
@torch.no_grad()
def midpoint_subdivide_masked(
    verts: torch.Tensor,      # (V,3), float
    faces: torch.Tensor,      # (F,3), long
    face_mask: torch.Tensor,  # (F,), bool
):
    """
    Vectorized midpoint subdivision on edges incident to masked faces.
    - Original faces are preserved and kept at the beginning.
    - New triangles are appended at the end (no originals are removed).
    - Edge midpoint indexing is deterministic.

    Returns:
        verts_new: (V',3)
        faces_new: (F + F_add, 3) long
    """
    device = verts.device
    V = int(verts.shape[0])
    F = int(faces.shape[0])

    if F == 0 or verts.numel() == 0:
        return verts, faces

    # --- Build the 3 undirected edges per face (sorted endpoint indices) ---
    v0 = faces[:, 0]
    v1 = faces[:, 1]
    v2 = faces[:, 2]

    e01 = torch.stack([torch.minimum(v0, v1), torch.maximum(v0, v1)], dim=1)  # (F,2)
    e12 = torch.stack([torch.minimum(v1, v2), torch.maximum(v1, v2)], dim=1)  # (F,2)
    e20 = torch.stack([torch.minimum(v2, v0), torch.maximum(v2, v0)], dim=1)  # (F,2)

    edges_all = torch.cat([e01, e12, e20], dim=0)  # (3F,2)
    # Face index for each edge occurrence (0..F-1, repeated 3 times)
    face_idx_for_occ = torch.arange(F, device=device, dtype=torch.long).repeat(3)

    # --- Unique undirected edges + inverse map to occurrences ---
    edges_unique, inverse = torch.unique(edges_all, dim=0, return_inverse=True)  # inverse: (3F,)
    E = edges_unique.shape[0]

    # --- Which unique edges are incident to any masked face? ---
    # For each occurrence, it's "split" iff its face is masked.
    occ_split = face_mask[face_idx_for_occ]  # (3F,) bool
    # Reduce by unique edge id (inverse) -> any occurrence masked => edge split
    # Use scatter_reduce if available, otherwise do amax via segment trick
    split_by_edge = torch.zeros(E, dtype=torch.int32, device=device)
    split_by_edge.scatter_reduce_(
        0, inverse, occ_split.to(torch.int32), reduce="amax", include_self=True
    )
    split_by_edge = split_by_edge.bool()
    # --- Create midpoints only for split edges, deterministically ---
    split_indices = torch.nonzero(split_by_edge, as_tuple=False).squeeze(1)  # (Es,)
    Es = int(split_indices.numel())

    if Es == 0:
        # Nothing to do: just return original verts and faces
        return verts, faces

    # Map unique-edge-id -> midpoint vertex index (or -1 if not split)
    edgeid_to_mid = torch.full((E,), -1, dtype=torch.long, device=device)
    edgeid_to_mid[split_indices] = V + torch.arange(Es, device=device, dtype=torch.long)

    # Midpoint positions
    i = edges_unique[split_indices, 0]
    j = edges_unique[split_indices, 1]
    midpos = 0.5 * (verts[i] + verts[j])  # (Es,3)

    verts_new = torch.cat([verts, midpos], dim=0)

    # --- For each face, gather midpoint indices for its three edges (m01,m12,m20) ---
    # inverse maps each occurrence to its unique-edge-id; reshape to (F,3)
    inv_reshaped = inverse.view(3, F).transpose(0, 1).contiguous()  # (F,3) rows: [e01,e12,e20] edge IDs
    m_all = edgeid_to_mid[inv_reshaped]  # (F,3) -> [m01, m12, m20], -1 if not split
    m01, m12, m20 = m_all[:, 0], m_all[:, 1], m_all[:, 2]

    # Count how many split edges per face
    c = (m_all >= 0).sum(dim=1)  # (F,)

    # --- Build new triangles (do NOT remove originals). Keep orientation consistent. ---
    # We append in deterministic groups: c==1 (3 subcases), then c==2 (3 subcases), then c==3.

    faces_extra = []

    # c == 1: exactly one edge split -> 2 triangles
    mask_c1 = (c == 1)
    if mask_c1.any():
        # Subcase: only m01 present
        s01 = mask_c1 & (m01 >= 0) & (m12 < 0) & (m20 < 0)
        if s01.any():
            idx = torch.nonzero(s01, as_tuple=False).squeeze(1)
            # triangles: [v2, v0, m01], [v2, m01, v1]
            t1 = torch.stack([v2[idx], v0[idx], m01[idx]], dim=1)
            t2 = torch.stack([v2[idx], m01[idx], v1[idx]], dim=1)
            faces_extra.append(t1)
            faces_extra.append(t2)

        # Subcase: only m12 present
        s12 = mask_c1 & (m01 < 0) & (m12 >= 0) & (m20 < 0)
        if s12.any():
            idx = torch.nonzero(s12, as_tuple=False).squeeze(1)
            # triangles: [v0, v1, m12], [v0, m12, v2]
            t1 = torch.stack([v0[idx], v1[idx], m12[idx]], dim=1)
            t2 = torch.stack([v0[idx], m12[idx], v2[idx]], dim=1)
            faces_extra.append(t1)
            faces_extra.append(t2)

        # Subcase: only m20 present
        s20 = mask_c1 & (m01 < 0) & (m12 < 0) & (m20 >= 0)
        if s20.any():
            idx = torch.nonzero(s20, as_tuple=False).squeeze(1)
            # triangles: [v1, v2, m20], [v1, m20, v0]
            t1 = torch.stack([v1[idx], v2[idx], m20[idx]], dim=1)
            t2 = torch.stack([v1[idx], m20[idx], v0[idx]], dim=1)
            faces_extra.append(t1)
            faces_extra.append(t2)

    # c == 2: two edges split -> 3 triangles
    mask_c2 = (c == 2)
    if mask_c2.any():
        # Subcase: m01 & m12
        s0112 = mask_c2 & (m01 >= 0) & (m12 >= 0) & (m20 < 0)
        if s0112.any():
            idx = torch.nonzero(s0112, as_tuple=False).squeeze(1)
            # triangles: [v2, v0, m01], [m01, v1, m12], [v2, m12, v0]
            t1 = torch.stack([v2[idx], v0[idx], m01[idx]], dim=1)
            t2 = torch.stack([m01[idx], v1[idx], m12[idx]], dim=1)
            t3 = torch.stack([v2[idx], m12[idx], v0[idx]], dim=1)
            faces_extra.append(t1)
            faces_extra.append(t2)
            faces_extra.append(t3)

        # Subcase: m12 & m20
        s1220 = mask_c2 & (m01 < 0) & (m12 >= 0) & (m20 >= 0)
        if s1220.any():
            idx = torch.nonzero(s1220, as_tuple=False).squeeze(1)
            # triangles: [v0, v1, m12], [m12, v2, m20], [v0, m20, v1]
            t1 = torch.stack([v0[idx], v1[idx], m12[idx]], dim=1)
            t2 = torch.stack([m12[idx], v2[idx], m20[idx]], dim=1)
            t3 = torch.stack([v0[idx], m20[idx], v1[idx]], dim=1)
            faces_extra.append(t1)
            faces_extra.append(t2)
            faces_extra.append(t3)

        # Subcase: m20 & m01
        s2001 = mask_c2 & (m01 >= 0) & (m12 < 0) & (m20 >= 0)
        if s2001.any():
            idx = torch.nonzero(s2001, as_tuple=False).squeeze(1)
            # triangles: [v1, v2, m20], [m20, v0, m01], [v1, m01, v2]
            t1 = torch.stack([v1[idx], v2[idx], m20[idx]], dim=1)
            t2 = torch.stack([m20[idx], v0[idx], m01[idx]], dim=1)
            t3 = torch.stack([v1[idx], m01[idx], v2[idx]], dim=1)
            faces_extra.append(t1)
            faces_extra.append(t2)
            faces_extra.append(t3)

    # c == 3: all three edges split -> 4 triangles
    mask_c3 = (c == 3)
    if mask_c3.any():
        idx = torch.nonzero(mask_c3, as_tuple=False).squeeze(1)
        # triangles: [v0, m01, m20], [m01, v1, m12], [m20, m12, v2], [m01, m12, m20]
        t1 = torch.stack([v0[idx], m01[idx], m20[idx]], dim=1)
        t2 = torch.stack([m01[idx], v1[idx], m12[idx]], dim=1)
        t3 = torch.stack([m20[idx], m12[idx], v2[idx]], dim=1)
        t4 = torch.stack([m01[idx], m12[idx], m20[idx]], dim=1)
        faces_extra.append(t1)
        faces_extra.append(t2)
        faces_extra.append(t3)
        faces_extra.append(t4)

    if len(faces_extra) == 0:
        # no new faces (shouldn't happen since Es>0, but safe)
        faces_new = faces
    else:
        faces_extra = torch.cat(faces_extra, dim=0).to(device=device, dtype=faces.dtype)
        # Append new triangles AFTER the original faces
        faces_new = torch.cat([faces, faces_extra], dim=0).contiguous()

    return verts_new, faces_new