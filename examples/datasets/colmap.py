import json
import os
from typing import Any, Dict, List, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import PIL
from PIL import Image
from pycolmap import SceneManager
from tqdm import tqdm
from typing_extensions import assert_never
from multiprocessing import Value as SMValue

from .normalize import (
    align_principal_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)


def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


def _resize_image_folder(image_dir: str, tasks, interpolation : int = cv2.INTER_CUBIC, ext=".png") -> str:
    """Resize image folder."""
    for factor, resized_dir in tasks:
        print(f"Downscaling images by {factor}x from {image_dir} to {resized_dir}.")
        os.makedirs(resized_dir, exist_ok=True)

    image_files = _get_rel_paths(image_dir)
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_dir, image_file)
        image = None;
        for factor, resized_dir in tasks:
            resized_path = os.path.join(
                resized_dir, os.path.splitext(image_file)[0] + ext
            )
            if os.path.isfile(resized_path): continue
            #lazy load
            if image is None: image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            resized_size = (
                int(round(image.shape[1] / factor)),
                int(round(image.shape[0] / factor)),
            )
            resized_image = cv2.resize(image, resized_size, interpolation)
            cv2.imwrite(resized_path, resized_image)
    return resized_dir


class Parser:
    """COLMAP parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every

        colmap_dir = os.path.join(data_dir, "sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse")
        assert os.path.exists(
            colmap_dir
        ), f"COLMAP directory {colmap_dir} does not exist."

        manager = SceneManager(colmap_dir)
        manager.load_cameras()
        manager.load_images()
        manager.load_points3D()

        # Extract extrinsic matrices in world-to-camera format.
        imdata = manager.images
        w2c_mats = []
        camera_ids = []
        camera_Ks_dict = dict()
        camera_params_dict = dict()
        camera_imsize_dict = dict()  # width, height
        mask_dict = dict()
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            rot = im.R()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)

            # support different camera intrinsics
            camera_id = im.camera_id
            camera_ids.append(camera_id)

            # camera intrinsics
            cam = manager.cameras[camera_id]
            fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

            # Get distortion parameters.
            type_ = cam.camera_type
            if type_ == 0 or type_ == "SIMPLE_PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif type_ == 1 or type_ == "PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            if type_ == 2 or type_ == "SIMPLE_RADIAL":
                params = np.array([cam.k1, 0.0, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 3 or type_ == "RADIAL":
                params = np.array([cam.k1, cam.k2, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 4 or type_ == "OPENCV":
                params = np.array([cam.k1, cam.k2, cam.p1, cam.p2], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 5 or type_ == "OPENCV_FISHEYE":
                params = np.array([cam.k1, cam.k2, cam.k3, cam.k4], dtype=np.float32)
                camtype = "fisheye"
            assert (
                camtype == "perspective" or camtype == "fisheye"
            ), f"Only perspective and fisheye cameras are supported, got {type_}"

            camera_Ks_dict[camera_id] = K
            camera_params_dict[camera_id] = params
            camera_imsize_dict[camera_id] = (cam.width, cam.height)
            mask_dict[camera_id] = None
        print(
            f"[Parser] {len(imdata)} images, taken by {len(set(camera_ids))} cameras."
        )

        if len(imdata) == 0:
            raise ValueError("No images found in COLMAP.")
        if not (type_ == 0 or type_ == 1):
            print("Warning: COLMAP Camera is not PINHOLE. Images have distortion.")

        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        camtoworlds = np.linalg.inv(w2c_mats)

        # Image names from COLMAP. No need for permuting the poses according to
        # image names anymore.
        image_names = [imdata[k].name for k in imdata]

        # Previous Nerf results were generated with images sorted by filename,
        # ensure metrics are reported on the same test set.
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        # Load extended metadata. Used by Bilarf dataset.
        self.extconf = {
            "spiral_radius_scale": 1.0,
            "no_factor_suffix": False,
        }
        extconf_file = os.path.join(data_dir, "ext_metadata.json")
        if os.path.exists(extconf_file):
            with open(extconf_file) as f:
                self.extconf.update(json.load(f))

        # Load bounds if possible (only used in forward facing scenes).
        self.bounds = np.array([0.01, 1.0])
        posefile = os.path.join(data_dir, "poses_bounds.npy")
        if os.path.exists(posefile):
            self.bounds = np.load(posefile)[:, -2:]

        colmap_image_dir = os.path.join(data_dir, "images")
        colmap_mask_dir = os.path.join(data_dir, "masks_matted")
        image_dir = os.path.join(data_dir, "images")
        mask_dir = os.path.join(data_dir, "masks_matted")
        resize_input = False
        for d in [image_dir, colmap_image_dir, mask_dir, colmap_mask_dir]:
            if not os.path.exists(d):
                raise ValueError(f"Image folder {d} does not exist.")

        # Downsampled images may have different names vs images used for COLMAP,
        # so we need to map between the two sorted lists of files.
        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        image_files = sorted(_get_rel_paths(image_dir))

        if factor > 1:
            mask_tasks = []
            image_tasks = []
            for f_down in range(2,factor+1):
                image_dir_f_down = os.path.join(os.path.dirname(image_dir), os.path.basename(image_dir) + f"_{f_down}")
                mask_dir_f_down = os.path.join(os.path.dirname(mask_dir), os.path.basename(mask_dir) + f"_{f_down}")
                image_tasks.append((f_down, image_dir_f_down))
                mask_tasks.append((f_down, mask_dir_f_down))
            _resize_image_folder(colmap_image_dir, image_tasks, ext=".jpeg")
            _resize_image_folder(colmap_mask_dir, mask_tasks, interpolation=cv2.INTER_CUBIC, ext=".png")
        
        mask_paths_by_factor = {}
        image_paths_by_factor = {}
        for f in range(1, factor+1):
            suffix = "" if f == 1 else f"_{f}"
            image_dir_f = os.path.join(os.path.dirname(image_dir), os.path.basename(image_dir) + suffix)
            mask_dir_f = os.path.join(os.path.dirname(mask_dir), os.path.basename(mask_dir) + suffix)
            image_files = sorted(_get_rel_paths(image_dir_f))
            mask_files = sorted(_get_rel_paths(mask_dir_f))

            colmap_to_image = dict(zip(colmap_files, image_files))
            image_paths_by_factor[f] = [os.path.join(image_dir_f, colmap_to_image[f]) for f in image_names]
            colmap_to_image = dict(zip(colmap_files, mask_files))
            mask_paths_by_factor[f] = [os.path.join(mask_dir_f, colmap_to_image[f]) for f in image_names]

        # 3D points and {image_name -> [point_idx]}
        points = manager.points3D.astype(np.float32)
        points_err = manager.point3D_errors.astype(np.float32)
        points_rgb = manager.point3D_colors.astype(np.uint8)
        point_indices = dict()

        image_id_to_name = {v: k for k, v in manager.name_to_image_id.items()}
        for point_id, data in manager.point3D_id_to_images.items():
            for image_id, _ in data:
                image_name = image_id_to_name[image_id]
                point_idx = manager.point3D_id_to_point3D_idx[point_id]
                point_indices.setdefault(image_name, []).append(point_idx)
        point_indices = {
            k: np.array(v).astype(np.int32) for k, v in point_indices.items()
        }

        # Normalize the world space.
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points = transform_points(T1, points)

            T2 = align_principal_axes(points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)

            transform = T2 @ T1

            # Fix for up side down. We assume more points towards
            # the bottom of the scene which is true when ground floor is
            # present in the images.
            if np.median(points[:, 2]) > np.mean(points[:, 2]):
                # rotate 180 degrees around x axis such that z is flipped
                T3 = np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
                camtoworlds = transform_cameras(T3, camtoworlds)
                points = transform_points(T3, points)
                transform = T3 @ transform
        else:
            transform = np.eye(4)

        self.image_names = image_names  # List[str], (num_images,)
        self.image_paths = image_paths_by_factor # Dict[Int,List[str]], (factor, (num_images,))
        self.mask_paths = mask_paths_by_factor # Dict[Int, List[str]], (factor, (num_images,))
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = {} # Dict of (factor,camera_id) -> K
        self.params_dict = camera_params_dict  # Dict of camera_id -> params
        self.imsize_dict = {} # Dict of (factor, camera_id) -> (width, height)
        self.mask_dict = mask_dict  # Dict of (factor, camera_id) -> mask
        self.points = points  # np.ndarray, (num_points, 3)
        self.points_err = points_err  # np.ndarray, (num_points,)
        self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)
        self.point_indices = point_indices  # Dict[str, np.ndarray], image_name -> [M,]
        self.transform = transform  # np.ndarray, (4, 4)

        # undistortion
        self.mapx_dict = dict() # (factor, camera_id) -> mapx
        self.mapy_dict = dict() # (factor, camera_id) -> mapy
        self.roi_undist_dict = dict() # (factor, camera_id) -> roi

        # load one image per factor to fill the intrinsics, sizes and generate undistort maps etc.
        for f in range(1, factor+1):
            actual_image = imageio.imread(self.image_paths[f][0])[..., :3]
            actual_height, actual_width = actual_image.shape[:2]
            colmap_width, colmap_height = camera_imsize_dict[self.camera_ids[0]]
            s_height, s_width = actual_height / colmap_height, actual_width / colmap_width
            for camera_id, K_original in camera_Ks_dict.items():
                K = K_original.copy()
                K[0, :] *= s_width
                K[1, :] *= s_height

                self.Ks_dict[(f,camera_id)] = K
                self.imsize_dict[(f,camera_id)] = (actual_width, actual_height)

            for camera_id in self.params_dict.keys():
                params = self.params_dict[camera_id]
                if len(params) == 0:
                    continue  # no distortion
                assert camera_id in camera_Ks_dict, f"Missing K for camera {camera_id}"
                assert (
                    camera_id in self.params_dict
                ), f"Missing params for camera {camera_id}"
                K = camera_Ks_dict[camera_id]

                if camtype == "perspective":
                    K_undist, _ = cv2.getOptimalNewCameraMatrix(
                        K_original, params, (colmap_width, colmap_height), 0, newImgSize=(actual_width, actual_height)
                    )
                    mapx, mapy = cv2.initUndistortRectifyMap(
                        K_original, params, None, K_undist, (actual_width, actual_height), cv2.CV_32FC1
                    )
                    self.Ks_dict[(f,camera_id)] = K_undist
                    mask = None
                elif camtype == "fisheye":
                    fx = K[0, 0]
                    fy = K[1, 1]
                    cx = K[0, 2]
                    cy = K[1, 2]
                    grid_x, grid_y = np.meshgrid(
                        np.arange(actual_width, dtype=np.float32),
                        np.arange(actual_height, dtype=np.float32),
                        indexing="xy",
                    )
                    x1 = (grid_x - cx) / fx
                    y1 = (grid_y - cy) / fy
                    theta = np.sqrt(x1**2 + y1**2)
                    r = (
                        1.0
                        + params[0] * theta**2
                        + params[1] * theta**4
                        + params[2] * theta**6
                        + params[3] * theta**8
                    )
                    mapx = (fx * x1 * r + actual_width // 2).astype(np.float32)
                    mapy = (fy * y1 * r + actual_height // 2).astype(np.float32)

                    # Use mask to define ROI
                    mask = np.logical_and(
                        np.logical_and(mapx > 0, mapy > 0),
                        np.logical_and(mapx < actual_width - 1, mapy < actual_height - 1),
                    )
                    y_indices, x_indices = np.nonzero(mask)
                    y_min, y_max = y_indices.min(), y_indices.max() + 1
                    x_min, x_max = x_indices.min(), x_indices.max() + 1
                    mask = mask[y_min:y_max, x_min:x_max]
                    K_undist = K.copy()
                    K_undist[0, 2] -= x_min
                    K_undist[1, 2] -= y_min
                else:
                    assert_never(camtype)

                self.mapx_dict[(f,camera_id)] = mapx
                self.mapy_dict[(f,camera_id)] = mapy
                self.Ks_dict[(f,camera_id)] = K_undist
                self.imsize_dict[(f,camera_id)] = (actual_width, actual_height)
                self.mask_dict[(f,camera_id)] = mask

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)


class Dataset:
    """A simple dataset class."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
        shared_factor : Optional[SMValue] = None
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        self.shared_factor = shared_factor
        indices = np.arange(len(self.parser.image_names))
        if split == "train":
            self.indices = indices[indices % self.parser.test_every != 0]
        else:
            self.indices = indices[indices % self.parser.test_every == 0]

    def __len__(self):
        return len(self.indices)
    def get_camtoworld(self, item:int)->torch.Tensor:
        index = self.indices[item]
        return torch.from_numpy(self.parser.camtoworlds[index])
    def __getitem__(self, item: int) -> Dict[str, Any]:
        factor = 1 if self.shared_factor is None else self.shared_factor.value
        index = self.indices[item]

        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[(factor,camera_id)].copy()
        params = self.parser.params_dict[camera_id]
        camtoworlds = self.parser.camtoworlds[index]
        mask = self.parser.mask_dict[(factor,camera_id)]
        
        load_factor = 1 if len(params) > 0 else factor

        image = cv2.imread(self.parser.image_paths[load_factor][index], cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if len(self.parser.mask_paths) > 0:
            fgmask = cv2.imread(self.parser.mask_paths[load_factor][index], cv2.IMREAD_UNCHANGED).squeeze()
        else: fgmask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)

        if len(params) > 0:
            # Images are distorted. Undistort them.
            mapx, mapy = (
                self.parser.mapx_dict[(factor,camera_id)],
                self.parser.mapy_dict[(factor,camera_id)],
            )
            image = cv2.remap(image, mapx, mapy, cv2.INTER_CUBIC)
            fgmask = cv2.remap(fgmask, mapx, mapy, cv2.INTER_CUBIC)

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            fgmask = fgmask[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "fgmask": torch.from_numpy(fgmask).float().unsqueeze(-1)/255,
            "image_id": item,  # the index of the image in the dataset
            "factor": factor
        }
        if mask is not None:
            data["mask"] = torch.from_numpy(mask).bool()

        if self.load_depths:
            # projected points to image plane to get depths
            worldtocams = np.linalg.inv(camtoworlds)
            image_name = self.parser.image_names[index]
            point_indices = self.parser.point_indices[image_name]
            points_world = self.parser.points[point_indices]
            points_cam = (worldtocams[:3, :3] @ points_world.T + worldtocams[:3, 3:4]).T
            points_proj = (K @ points_cam.T).T
            points = points_proj[:, :2] / points_proj[:, 2:3]  # (M, 2)
            depths = points_cam[:, 2]  # (M,)
            # filter out points outside the image
            selector = (
                (points[:, 0] >= 0)
                & (points[:, 0] < image.shape[1])
                & (points[:, 1] >= 0)
                & (points[:, 1] < image.shape[0])
                & (depths > 0)
            )
            points = points[selector]
            depths = depths[selector]
            data["points"] = torch.from_numpy(points).float()
            data["depths"] = torch.from_numpy(depths).float()

        return data


if __name__ == "__main__":
    import argparse

    import imageio.v2 as imageio

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/360_v2/garden")
    parser.add_argument("--factor", type=int, default=4)
    args = parser.parse_args()

    # Parse COLMAP data.
    parser = Parser(
        data_dir=args.data_dir, factor=args.factor, normalize=True, test_every=8
    )
    dataset = Dataset(parser, split="train", load_depths=True)
    print(f"Dataset: {len(dataset)} images.")

    writer = imageio.get_writer("results/points.mp4", fps=30)
    for data in tqdm(dataset, desc="Plotting points"):
        image = data["image"].numpy().astype(np.uint8)
        points = data["points"].numpy()
        depths = data["depths"].numpy()
        for x, y in points:
            cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)
        writer.append_data(image)
    writer.close()
