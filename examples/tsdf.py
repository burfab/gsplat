import open3d as o3d
from typing import NamedTuple
import numpy as np

class TSDFArgs(NamedTuple):
    max_depth: float = 1300.0
    depth_scale: float = 1000.0
    sdf_voxel_length: float = 1/256
    sdf_trunc_multiplier: float = 8.0
    sdf_volume_unit_resolution: int = 16
    sdf_block_count: int = 10000
    mesh_cluster_area_th: float = (1 * 1)
    mask_closing_kernel_size: int = 5
    mask_erosion_kernel_size: int = 5
    use_gpu: bool = False


class TSDF:
    def __init__(self, args):
        """
        Initialize the TSDF fusion algorithm.

        Parameters:
        renderer (Renderer): Renderer class object.
        stereo (Stereo): Stereo class object.
        args (ArgParser): Program arguments.
        out_name (str): Output name for saving the mesh.
        """
        self.args = args
        self.cleaned_mesh = None
        self.mesh = None
        self.device = o3d.core.Device("CUDA:0" if self.args.use_gpu else "CPU:0")
    
    def legacy_to_tensor_rgbd(legacy_rgbd: o3d.geometry.RGBDImage, device):
        color_t = o3d.t.geometry.Image(
            o3d.core.Tensor(np.asarray(legacy_rgbd.color).astype(np.float32)/255.0, o3d.core.Dtype.Float32, device=device)
        )
        depth_t = o3d.t.geometry.Image(
            o3d.core.Tensor(np.asarray(legacy_rgbd.depth), o3d.core.Dtype.Float32, device=device)
        )
        return o3d.t.geometry.RGBDImage(color_t, depth_t)

    def create(self):

        # === Create voxel block grid (sparse TSDF volume) ===
        voxel_size = float(self.args.sdf_voxel_length)
        block_resolution = self.args.sdf_volume_unit_resolution
        block_count = self.args.sdf_block_count # adjust as needed

        # The typical TSDF attributes: tsdf, weight, color
        tsdf_attrs = ["tsdf", "weight", "color"]
        tsdf_dtypes = [o3d.core.Dtype.Float32, o3d.core.Dtype.Float32, o3d.core.Dtype.Float32]
        tsdf_channels = [[1], [1], [3]]

        self.volume = o3d.t.geometry.VoxelBlockGrid(
            attr_names=tsdf_attrs,
            attr_dtypes=tsdf_dtypes,
            attr_channels=tsdf_channels,
            voxel_size=voxel_size,
            block_resolution=block_resolution,
            block_count=block_count,
            device=self.device,
        )
        self.target_pcl = None
        
        
    def integrate(self, o3d_rgbd, transform_np, K_np, update_target_pcl=False):
        voxel_size = self.args.sdf_voxel_length
        rgbd = TSDF.legacy_to_tensor_rgbd(o3d_rgbd, self.device)
        K = o3d.core.Tensor(K_np, o3d.core.Dtype.Float64, self.device)
        T = transform_np.copy()
        if self.target_pcl is not None:
            intrinsics = o3d.camera.PinholeCameraIntrinsic(np.asarray(o3d_rgbd.depth).shape[1], np.asarray(o3d_rgbd.depth).shape[0], 
                                                            K_np[0,0], K_np[1,1], K_np[0,2], K_np[1,2])
            pcl = o3d.geometry.PointCloud.create_from_rgbd_image(o3d_rgbd, intrinsics)
            #o3d.visualization.draw_geometries([pcl])
            reg = o3d.pipelines.registration.registration_icp(
                source=pcl,
                target=self.target_pcl,
                max_correspondence_distance=voxel_size*10,  
                init=np.linalg.inv(transform_np),  
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            reg = o3d.pipelines.registration.registration_icp(
                source=pcl,
                target=self.target_pcl,
                max_correspondence_distance=voxel_size*5,  # in meters, adjust based on scale
                init=reg.transformation,  # because `pcl` is already roughly aligned
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            if reg.fitness > 0.5: 
                T = np.linalg.inv(reg.transformation)
            print(reg)
                
        extr = o3d.core.Tensor((T), o3d.core.Dtype.Float64, self.device)
        block_coords = self.volume.compute_unique_block_coordinates(
            depth=rgbd.depth,
            intrinsic=K,
            extrinsic=extr,
            depth_scale=1.0,
            depth_max=self.args.max_depth/self.args.depth_scale,
            trunc_voxel_multiplier=self.args.sdf_trunc_multiplier,
        )
        self.volume.integrate(
            block_coords=block_coords,
            depth=rgbd.depth,
            color=rgbd.color,
            intrinsic=K,
            extrinsic=extr,
            depth_scale=1.0,
            depth_max=self.args.max_depth/self.args.depth_scale,
            trunc_voxel_multiplier=self.args.sdf_trunc_multiplier,
        )
        if update_target_pcl:
            self.target_pcl = self.volume.extract_point_cloud(weight_threshold=0.0).to_legacy()
        return T
        
    def extract_mesh(self, weight_th=3.0):
        mesh = self.volume.extract_triangle_mesh(weight_threshold=weight_th).to_legacy()
        mesh.scale(self.args.depth_scale, (0, 0, 0))
        mesh.compute_vertex_normals()
        return mesh
    
    def extract_mesh_poisson(self, weight_th=3.0, poisson_depth = 9, remove_low_density_vertices_th=0.01):
        pcl = self.volume.extract_point_cloud(weight_threshold=weight_th).to_legacy()
        mesh,densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcl, depth=poisson_depth)
        vertices_to_remove = densities < np.quantile(densities, remove_low_density_vertices_th)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        mesh.scale(self.args.depth_scale, (0, 0, 0))
        mesh.compute_vertex_normals()
        return mesh
        
        
    def extract_pcl(self, weight_th=3.0):
        pcl = self.volume.extract_point_cloud(weight_threshold=weight_th).to_legacy()
        pcl.scale(self.args.depth_scale, (0, 0, 0))
        return pcl