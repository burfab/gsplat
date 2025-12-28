import numpy as np
import open3d as o3d
import open3d.visualization.rendering as headless_rendering

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
        out_color_path="color.png",
        out_depth_u16_path="depth_mm_u16.png"):
        self.renderer.setup_camera(K, T_wc, self.width, self.height)

        color_o3d = self.renderer.render_to_image()
        
        if not out_color_path is None and len(out_color_path) > 0:
            o3d.io.write_image(out_color_path, color_o3d)

        depth_o3d = self.renderer.render_to_depth_image(z_in_view_space=True)
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