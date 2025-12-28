import pytorch3d
import pytorch3d.ops
import pytorch3d.loss
import pytorch3d.structures
import pytorch3d.utils
import torch
import open3d as o3d
import numpy as np
import pytorch3d._C

def rotation_matrix_to_quat(R: torch.Tensor) -> torch.Tensor:
    return pytorch3d.transforms.matrix_to_quaternion(R)

def compute_camera_view_sample_weights(dataset, scene_scale):
    cam_centers = torch.zeros((len(dataset),3), dtype=torch.float32)
    for i in range(len(dataset)):
        campos = torch.inverse(dataset.get_camtoworld(i))[:3,3]
        cam_centers[i] = campos
        
    dists = torch.cdist(cam_centers, cam_centers)  # (N, N)
    r = scene_scale * 0.1
    neighbor_counts = (dists < r).float().sum(dim=1).clamp_min(1.0)  # (N,)
    raw_weights = 1.0 / neighbor_counts
    weights = raw_weights / raw_weights.sum()
    return weights

def sample_barycentric(N:int)->torch.Tensor:
    P = torch.distributions.Dirichlet(torch.tensor([1.0,1.0,1.0]).float())
    D = P.sample((N,1))
    return D.float()
def barycentric_to_parameter_space(X:torch.Tensor)->torch.Tensor:
    return torch.log(X)
def barycentric_from_parameter_space(X:torch.Tensor, dim=-1)->torch.Tensor:
    return torch.softmax(X,dim=-1)
def barycentric_to_reduced_parameter_space(bary: torch.Tensor) -> torch.Tensor:
    """
    bary: (..., 3) tensor with [u, v, w], u+v+w=1
    returns: (..., 2) logits [a, b]
    """
    eps = 1e-8
    u = bary[..., 0]
    v = bary[..., 1]
    w = bary[..., 2].clamp_min(eps)  # avoid division by zero

    a = torch.log(u.clamp_min(eps) / w)
    b = torch.log(v.clamp_min(eps) / w)
    return torch.stack((a, b), dim=-1)
def barycentric_from_reduced_parameter_space(X:torch.Tensor)->torch.Tensor:
    Xmax = torch.amax(X, dim=-1, keepdim=True)
    eX = torch.exp(X - Xmax)
    ones = torch.ones_like(eX[..., :1])
    numer = torch.cat((eX, ones), dim=-1)
    denom = numer.sum(dim=-1, keepdim=True)
    return numer / denom

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

def quaternion_from_tangent_frame(t,b,n):
    R = torch.stack([t, b, n], dim=-1)
    return rotation_matrix_to_quat(R)
def quaternion_from_normal(n):
    t, b = onb_from_normal_frisvad(n)  # (F, 3), (F, 3)
    return quaternion_from_tangent_frame(t,b,n)

def get_edges(mesh, faces):
    edges = mesh.edges_packed()[mesh.faces_packed_to_edges_packed()[faces]]
    return edges

def get_edge_lengths(mesh, faces):
    edge_verts = mesh.verts_packed()[get_edges(mesh, faces)]
    edge_dirs = edge_verts[:,:,0,:]-edge_verts[:,:,1,:]
    return torch.linalg.norm(edge_dirs, dim=1)


@torch.no_grad()
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
    return (V_new[:len(vertices)], V_new[len(vertices):]), (unmasked_faces, mesh_new.faces_list()[0]), ~mask_split


def masked_midpoint_subdivide(V,T, mask):
    def compute_keys(T, max_el, edge_indices):
        AB = T[...,edge_indices[0]]
        CA = T[...,edge_indices[1]]
        BC = T[...,edge_indices[2]]
        keys = torch.stack((AB,CA,BC),dim=1)
        keys = torch.sort(keys)[0]
        keys[...,0] *= max_el
        keys = keys.sum(-1)
        return keys
        
        
    edge_indices = [[0,1], [2,0], [1,2]]
    all_keys = compute_keys(T, T.max(), edge_indices).cpu().detach().numpy()
    masked_keys = compute_keys(T[mask], T.max(), edge_indices).cpu().detach().numpy()

    V_cpu = V.cpu().detach().numpy()
    T_cpu = T.cpu().detach().numpy()
    T_masked_cpu = T[mask].cpu().detach().numpy()
    midpoints = {}
    new_vertices_idx_counter = len(V)
    for i in range(len(masked_keys)):
        tri_keys = masked_keys[i]
        for ei in range(3):
            key_ei = tri_keys[ei].item()
            if not key_ei in midpoints:
                midpoints[key_ei] = (V_cpu[T_masked_cpu[i,edge_indices[ei]]].mean(0).tolist(),new_vertices_idx_counter)
                new_vertices_idx_counter += 1

    new_vertices = np.zeros((len(midpoints),3), dtype=V_cpu.dtype)
    for v, idx in midpoints.values(): new_vertices[idx-len(V)] = v
            
    new_faces_cnt = np.zeros(len(T),dtype=int)
    for i in range(len(T_cpu)):
        tri_keys = all_keys[i]
        split_edges = [False, False, False]
        split_cnt = 0
        for ei in range(3):
            if tri_keys[ei].item() in midpoints:
                split_edges[ei]=True
                split_cnt+=1
        if split_cnt == 0: continue
        new_faces_cnt[i] = (split_cnt+1)

    new_faces_start_index = new_faces_cnt.cumsum()
    new_faces = np.zeros((new_faces_start_index[-1],3), dtype=T_cpu.dtype)
    for i in range(len(T_cpu)):
        split_cnt = new_faces_cnt[i]-1
        if split_cnt <= 0: continue
        #ATTENTION: this depends on order of edge_indices
        key_AB,key_CA, key_BC = all_keys[i]
        A,B,C = T_cpu[i]
        d,e,f = (None,None,None)
        if key_AB in midpoints: d = midpoints[key_AB][1]
        if key_BC in midpoints: e = midpoints[key_BC][1]
        if key_CA in midpoints: f = midpoints[key_CA][1]
        start_idx = 0 if i == 0 else new_faces_start_index[i-1]
        faces_to_add = None
        if split_cnt == 3:
            faces_to_add = [[A,d,f], [d,B,e], [e,C,f], [d,e,f]]
        elif split_cnt == 2:
            if f is None: faces_to_add = [[d,B,e], [d,e,C], [A,d,C]]
            elif e is None: faces_to_add = [[A,d,f], [d,C,f], [d,B,C]]
            elif d is None: faces_to_add = [[A,e,f], [f,e,C], [A,B,e]]
        elif split_cnt == 1:
            if not d is None: faces_to_add = [[A,d,C], [d,B,C]]
            elif not e is None: faces_to_add = [[A,B,e], [e,C,A]]
            elif not f is None: faces_to_add = [[B,f,A], [f,B,C]]
        for j, nf in enumerate(faces_to_add):
            new_faces[start_idx+j] = nf
            
    mask_untouched_faces = torch.from_numpy(new_faces_cnt == 0).to(T.device)
    T_keep = T[mask_untouched_faces]
    T_new = torch.from_numpy(new_faces).to(T.device).to(T.dtype)
    V_new = torch.from_numpy(new_vertices).to(V.device).to(V.dtype)
    
    return (V, V_new), (T_keep, T_new), mask_untouched_faces

def _cos_angle_between(u: torch.Tensor, v: torch.Tensor):
    """
    u, v: (..., 3)
    returns angles in radians: (...,)
    """
    u = torch.nn.functional.normalize(u, dim=-1)
    v = torch.nn.functional.normalize(v, dim=-1)
    cos = (u * v).sum(dim=-1)
    return cos


def triangle_cos_angles(meshes: pytorch3d.structures.Meshes):
    """
    Encourage all triangle internal angles to be close to 60 degrees (pi/3).

    meshes: PyTorch3D Meshes
    returns: scalar loss
    """
    verts = meshes.verts_packed()   # (sum_V, 3)
    faces = meshes.faces_packed()   # (sum_F, 3)

    v0 = verts[faces[:, 0]]  # (F, 3)
    v1 = verts[faces[:, 1]]  # (F, 3)
    v2 = verts[faces[:, 2]]  # (F, 3)

    # Edges from each vertex
    e0_1 = v1 - v0  # edge v0->v1
    e0_2 = v2 - v0  # edge v0->v2

    e1_0 = v0 - v1  # edge v1->v0
    e1_2 = v2 - v1  # edge v1->v2

    e2_0 = v0 - v2  # edge v2->v0
    e2_1 = v1 - v2  # edge v2->v1

    # Internal angles at v0, v1, v2
    cos_a0 = _cos_angle_between(e0_1, e0_2)  # (F,)
    cos_a1 = _cos_angle_between(e1_0, e1_2)  # (F,)
    cos_a2 = _cos_angle_between(e2_0, e2_1)  # (F,)

    cos_angles = torch.stack([cos_a0, cos_a1, cos_a2], dim=-1)  # (F, 3)
    return cos_angles

    return loss
def mesh_normal_consistency_with_modes(meshes, mode="default",**kwargs):
    r"""
    Computes the normal consistency of each mesh in meshes.
    We compute the normal consistency for each pair of neighboring faces.
    If e = (v0, v1) is the connecting edge of two neighboring faces f0 and f1,
    then the normal consistency between f0 and f1

    .. code-block:: python

                    a
                    /\
                   /  \
                  / f0 \
                 /      \
            v0  /____e___\ v1
                \        /
                 \      /
                  \ f1 /
                   \  /
                    \/
                    b

    The normal consistency is

    .. code-block:: python

        nc(f0, f1) = 1 - cos(n0, n1)

        where cos(n0, n1) = n0^n1 / ||n0|| / ||n1|| is the cosine of the angle
        between the normals n0 and n1, and

        n0 = (v1 - v0) x (a - v0)
        n1 = - (v1 - v0) x (b - v0) = (b - v0) x (v1 - v0)

    This means that if nc(f0, f1) = 0 then n0 and n1 point to the same
    direction, while if nc(f0, f1) = 2 then n0 and n1 point opposite direction.

    .. note::
        For well-constructed meshes the assumption that only two faces share an
        edge is true. This assumption could make the implementation easier and faster.
        This implementation does not follow this assumption. All the faces sharing e,
        which can be any in number, are discovered.

    Args:
        meshes: Meshes object with a batch of meshes.

    Returns:
        loss: Average normal consistency across the batch.
        Returns 0 if meshes contains no meshes or all empty meshes.
    """
    if meshes.isempty():
        return torch.tensor(
            [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
        )

    N = len(meshes)
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    faces_packed = meshes.faces_packed()  # (sum(F_n), 3)
    edges_packed = meshes.edges_packed()  # (sum(E_n), 2)
    verts_packed_to_mesh_idx = meshes.verts_packed_to_mesh_idx()  # (sum(V_n),)
    face_to_edge = meshes.faces_packed_to_edges_packed()  # (sum(F_n), 3)
    E = edges_packed.shape[0]  # sum(E_n)
    F = faces_packed.shape[0]  # sum(F_n)

    # We don't want gradients for the following operation. The goal is to
    # find for each edge e all the vertices associated with e. In the example
    # above, the vertices associated with e are (a, b), i.e. the points connected
    # on faces to e.
    with torch.no_grad():
        edge_idx = face_to_edge.reshape(F * 3)  # (3 * F,) indexes into edges
        vert_idx = (
            faces_packed.view(1, F, 3).expand(3, F, 3).transpose(0, 1).reshape(3 * F, 3)
        )
        edge_idx, edge_sort_idx = edge_idx.sort()
        vert_idx = vert_idx[edge_sort_idx]

        # In well constructed meshes each edge is shared by precisely 2 faces
        # However, in many meshes, this assumption is not always satisfied.
        # We want to find all faces that share an edge, a number which can
        # vary and which depends on the topology.
        # In particular, we find the vertices not on the edge on the shared faces.
        # In the example above, we want to associate edge e with vertices a and b.
        # This operation is done more efficiently in cpu with lists.
        # TODO(gkioxari) find a better way to do this.

        # edge_idx represents the index of the edge for each vertex. We can count
        # the number of vertices which are associated with each edge.
        # There can be a different number for each edge.
        edge_num = edge_idx.bincount(minlength=E)

        # This calculates all pairs of vertices which are opposite to the same edge.
        vert_edge_pair_idx = pytorch3d._C.mesh_normal_consistency_find_verts(edge_num.cpu()).to(
            edge_num.device
        )

    if vert_edge_pair_idx.shape[0] == 0:
        return torch.tensor(
            [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
        )

    v0_idx = edges_packed[edge_idx, 0]
    v0 = verts_packed[v0_idx]
    v1_idx = edges_packed[edge_idx, 1]
    v1 = verts_packed[v1_idx]

    # two of the following cross products are zeros as they are cross product
    # with either (v1-v0)x(v1-v0) or (v1-v0)x(v0-v0)
    n_temp0 = (v1 - v0).cross(verts_packed[vert_idx[:, 0]] - v0, dim=1)
    n_temp1 = (v1 - v0).cross(verts_packed[vert_idx[:, 1]] - v0, dim=1)
    n_temp2 = (v1 - v0).cross(verts_packed[vert_idx[:, 2]] - v0, dim=1)
    n = n_temp0 + n_temp1 + n_temp2
    n0 = n[vert_edge_pair_idx[:, 0]]
    n1 = -n[vert_edge_pair_idx[:, 1]]
    
    if mode == "L1":
        n0 = torch.nn.functional.normalize(n0, dim=-1)
        n1 = torch.nn.functional.normalize(n1, dim=-1)
        loss = torch.abs(n0-n1).sum(-1)
    else:
        costheta = torch.cosine_similarity(n0, n1, dim=1)
        if mode == "edge_aware":
            sigma = kwargs.pop("sigma")
            #small angle approx, as acos is non diff
            theta2 = 2 * (1 - costheta)
            if kwargs.get("edge_aware_rational_weight",True):
                wij = 1.0/(1+(theta2/sigma**2)) 
            else:
                wij = torch.exp(-(theta2)/(2.0*sigma**2))
            loss = wij*(1 - costheta)
        elif mode == "default": 
            loss = (1 - costheta)
        else:
            assert False, f"Unknown mode {mode}"

    verts_packed_to_mesh_idx = verts_packed_to_mesh_idx[vert_idx[:, 0]]
    verts_packed_to_mesh_idx = verts_packed_to_mesh_idx[vert_edge_pair_idx[:, 0]]
    num_normals = verts_packed_to_mesh_idx.bincount(minlength=N)
    weights = 1.0 / num_normals[verts_packed_to_mesh_idx].float()

    loss = loss * weights
    return loss.sum() / N

def find_closest_face(
    meshes: pytorch3d.structures.Meshes,
    pcls: pytorch3d.structures.Pointclouds,
    min_triangle_area: float = 5e-3
):
    """
    Computes the distance between a pointcloud and a mesh within a batch.
    Given a pair `(mesh, pcl)` in the batch, we define the distance to be the
    sum of two distances, namely `point_face(mesh, pcl) + face_point(mesh, pcl)`

    `point_face(mesh, pcl)`: Computes the squared distance of each point p in pcl
        to the closest triangular face in mesh and averages across all points in pcl
    `face_point(mesh, pcl)`: Computes the squared distance of each triangular face in
        mesh to the closest point in pcl and averages across all faces in mesh.

    The above distance functions are applied for all `(mesh, pcl)` pairs in the batch
    and then averaged across the batch.

    Args:
        meshes: A Meshes data structure containing N meshes
        pcls: A Pointclouds data structure containing N pointclouds
        min_triangle_area: (float, defaulted) Triangles of area less than this
            will be treated as points/lines.

    Returns:
        loss: The `point_face(mesh, pcl) + face_point(mesh, pcl)` distance
            between all `(mesh, pcl)` in a batch averaged across the batch.
    """

    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    N = len(meshes)
    assert N == 1, "Only supports batch size of 1"
        

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()

    # point to face distance: shape (P,)
    dists, inds = pytorch3d._C.point_face_dist_forward(
            points,
            points_first_idx,
            tris,
            tris_first_idx,
            max_points,
            min_triangle_area,
        )
    return dists, inds 



    
def triangle_incenter(V, F):
    # V: (N,3), Fsel: (K,3) selected faces
    A = V[F[:,0]]; B = V[F[:,1]]; C = V[F[:,2]]
    a = (B - C).norm(dim=1, keepdim=True)
    b = (C - A).norm(dim=1, keepdim=True)
    c = (A - B).norm(dim=1, keepdim=True)
    P = (a*A + b*B + c*C) / (a + b + c)
    return P



def triangle_incenter_subdivide(V:torch.Tensor, F:torch.Tensor, mask:torch.Tensor)->tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """
    appends new vertices at the end, appends new faces at the end, where V_created[i] belongs to F[mask][i]
    vertices = (V, V_created)
    faces = (F[~mask], F_created)
    """
    
    def add_faces_with_triangle_incenter_vertex(Fsel,Vidx):
        A = Fsel[...,0:1]; B = Fsel[...,1:2]; C = Fsel[...,2:3]; v = Vidx[:, None]
        # maintain same (A,B,C) orientation
        F1 = torch.hstack((A, B, v))
        F3 = torch.hstack((B, C, v))
        F2 = torch.hstack((C, A, v))
        return torch.cat([F1,F2,F3])
    
    def compute_new_vertex_indices(V, V_new):
        return torch.arange(len(V), len(V)+len(V_new)).to(V.device)
    
    V_created = triangle_incenter(V, F[mask]); temp = V_created.cpu().detach().numpy().astype(np.float32)
    V_new_ind = compute_new_vertex_indices(V, V_created)
    F_created = add_faces_with_triangle_incenter_vertex(F[mask], V_new_ind)

    return (V, V_created), (F[~mask], F_created), ~mask