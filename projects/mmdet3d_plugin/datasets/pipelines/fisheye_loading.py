# Copyright (c) OpenMMLab. All rights reserved.
# Fisheye-specific data loading pipelines

import os
import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion

from mmdet3d.datasets.builder import PIPELINES


def mmlabNormalize(img):
    from mmcv.image.photometric import imnormalize
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    to_rgb = True
    img = imnormalize(np.array(img), mean, std, to_rgb)
    img = torch.tensor(img).float().permute(2, 0, 1).contiguous()
    return img


@PIPELINES.register_module()
class FisheyePrepareImageInputs(object):
    """Prepare image inputs for fisheye cameras.

    Similar to PrepareImageInputs but handles:
    - sensor2ego from cam_to_base (4x4) instead of sensor2ego_rotation/translation
    - ego2global from info-level (same for all cameras)
    - distortion coefficients (Kannala-Brandt k1-k4)
    """

    def __init__(self, data_config, is_train=False, sequential=False):
        self.is_train = is_train
        self.data_config = data_config
        self.normalize_img = mmlabNormalize
        self.sequential = sequential

    def choose_cams(self):
        if self.is_train and self.data_config.get('Ncams', 4) < len(
                self.data_config['cams']):
            cam_names = np.random.choice(
                self.data_config['cams'],
                self.data_config['Ncams'],
                replace=False)
        else:
            cam_names = self.data_config['cams']
        return cam_names

    def sample_augmentation(self, H, W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        if self.is_train:
            resize = float(fW) / float(W)
            resize += np.random.uniform(*self.data_config.get('resize', (-0.06, 0.11)))
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config.get('crop_h', (0.0, 0.0)))) *
                         newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config.get('flip', False) and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config.get('rot', (-5.4, 5.4)))
        else:
            resize = float(fW) / float(W)
            if scale is not None:
                resize += scale
            else:
                resize += self.data_config.get('resize_test', 0.0)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config.get('crop_h', (0.0, 0.0)))) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def get_rot(self, h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran, resize, resize_dims,
                      crop, flip, rotate):
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    def get_sensor_transforms(self, info, cam_name):
        """Get sensor2ego and ego2global transforms for fisheye cameras.

        sensor2ego: from cam_to_base (4x4) directly
        ego2global: from info-level ego2global quaternion (same for all cameras)
        """
        cam_data = info['cams'][cam_name]

        # sensor2ego from cam_to_base
        cam_to_base = torch.Tensor(cam_data['cam_to_base'])
        sensor2ego = cam_to_base.clone()

        # ego2global from info-level
        w, x, y, z = info['ego2global_rotation']
        ego2global_rot = torch.Tensor(Quaternion(w, x, y, z).rotation_matrix)
        ego2global_tran = torch.Tensor(info['ego2global_translation'])
        ego2global = ego2global_rot.new_zeros((4, 4))
        ego2global[3, 3] = 1
        ego2global[:3, :3] = ego2global_rot
        ego2global[:3, -1] = ego2global_tran

        return sensor2ego, ego2global

    def get_inputs(self, results, flip=None, scale=None):
        """Get image inputs for all cameras.

        Returns:
            imgs: (N_views, 3, H, W)
            sensor2egos: (N_views, 4, 4)
            ego2globals: (N_views, 4, 4)
            intrins: (N_views, 3, 3)
            post_rots: (N_views, 3, 3)
            post_trans: (N_views, 3)
            distortions: (N_views, 4)
        """
        imgs = []
        sensor2egos = []
        ego2globals = []
        intrins = []
        post_rots = []
        post_trans = []
        distortions = []
        cam_names = self.choose_cams()
        results['cam_names'] = cam_names
        canvas = []

        for cam_name in cam_names:
            cam_data = results['curr']['cams'][cam_name]
            filename = cam_data['data_path']
            img = Image.open(filename)

            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            intrin = torch.Tensor(cam_data['cam_intrinsic'])
            distort = torch.Tensor(cam_data['distortion'])

            sensor2ego, ego2global = \
                self.get_sensor_transforms(results['curr'], cam_name)

            img_augs = self.sample_augmentation(
                H=img.height, W=img.width, flip=flip, scale=scale)
            resize, resize_dims, crop, flip, rotate = img_augs

            img, post_rot2, post_tran2 = \
                self.img_transform(img, post_rot, post_tran,
                                   resize=resize, resize_dims=resize_dims,
                                   crop=crop, flip=flip, rotate=rotate)

            post_tran_3 = torch.zeros(3)
            post_rot_3 = torch.eye(3)
            post_tran_3[:2] = post_tran2
            post_rot_3[:2, :2] = post_rot2

            canvas.append(np.array(img))
            imgs.append(self.normalize_img(img))

            intrins.append(intrin)
            sensor2egos.append(sensor2ego)
            ego2globals.append(ego2global)
            post_rots.append(post_rot_3)
            post_trans.append(post_tran_3)
            distortions.append(distort)

        imgs = torch.stack(imgs)
        sensor2egos = torch.stack(sensor2egos)
        ego2globals = torch.stack(ego2globals)
        intrins = torch.stack(intrins)
        post_rots = torch.stack(post_rots)
        post_trans = torch.stack(post_trans)
        distortions = torch.stack(distortions)
        results['canvas'] = canvas

        return imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, distortions

    def __call__(self, results):
        results['img_inputs'] = self.get_inputs(results)
        return results


@PIPELINES.register_module()
class FisheyeLoadAnnotationsBEVDepth(object):
    """Load BEV annotations for fisheye dataset (simplified, no 3D bboxes)."""

    def __init__(self, bda_aug_conf, classes, is_train=True):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train
        self.classes = classes

    def sample_bda_augmentation(self):
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
        return rotate_bda, scale_bda, flip_dx, flip_dy

    def __call__(self, results):
        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation()

        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1

        rotate_angle = torch.tensor(rotate_bda / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0],
                                [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_bda, 0, 0],
                                  [0, scale_bda, 0],
                                  [0, 0, scale_bda]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        bda_rot = flip_mat @ (scale_mat @ rot_mat)
        bda_mat[:3, :3] = bda_rot

        # Unpack: fisheye img_inputs has 7 elements (includes distortions)
        imgs, sensor2egos, ego2globals, intrins = results['img_inputs'][:4]
        post_rots, post_trans, distortions = results['img_inputs'][4:7]

        results['img_inputs'] = (imgs, sensor2egos, ego2globals, intrins,
                                 post_rots, post_trans, bda_rot, distortions)

        results['flip_dx'] = flip_dx
        results['flip_dy'] = flip_dy
        results['rotate_bda'] = rotate_bda
        results['scale_bda'] = scale_bda

        if results.get('flip_dx', False):
            if 'voxel_semantics' in results:
                results['voxel_semantics'] = results['voxel_semantics'][::-1, ...].copy()
            if 'mask_camera' in results:
                results['mask_camera'] = results['mask_camera'][::-1, ...].copy()
        if results.get('flip_dy', False):
            if 'voxel_semantics' in results:
                results['voxel_semantics'] = results['voxel_semantics'][:, ::-1, ...].copy()
            if 'mask_camera' in results:
                results['mask_camera'] = results['mask_camera'][:, ::-1, ...].copy()

        return results


@PIPELINES.register_module()
class FisheyeLoadOccGTFromFile(object):
    """Load occupancy GT from NPZ file for fisheye dataset.

    The fisheye NPZ contains: 'semantics' (100,100,20) uint8 and
    'occupied' (100,100,20) uint8.

    7-class label mapping (mirrors nuScenes: 1 free + 6 occupied):
        occupied=0              -> class 0 (free)
        occupied=1, semantics=0 -> class 1 (unknown occupied)
        occupied=1, semantics=1 -> class 2 (person)
        occupied=1, semantics=2 -> class 3 (table)
        occupied=1, semantics=3 -> class 4 (chair)
        occupied=1, semantics=4 -> class 5 (floor)
        occupied=1, semantics=5 -> class 6 (car)

    mask_lidar = occupied (for metrics evaluation).
    mask_camera = all ones (4 fisheye cameras cover 360 deg).
    """

    def __call__(self, results):
        occ_gt_path = results['occ_gt_path']
        occ_labels = np.load(occ_gt_path, allow_pickle=True)
        semantics = occ_labels['semantics']    # (Dx, Dy, Dz)
        occupied = occ_labels['occupied']      # (Dx, Dy, Dz)

        semantics = torch.from_numpy(semantics.astype(np.int64))
        occupied = torch.from_numpy(occupied.astype(np.int64))

        # Remap to 7-class label space
        # occupied=1, semantics=0 -> class 1 (unknown occupied)
        unknown_mask = (occupied == 1) & (semantics == 0)
        semantics[unknown_mask] = 1
        # occupied=1, semantics=1..5 -> shift to class 2..6
        known_mask = (occupied == 1) & (semantics > 0)
        semantics[known_mask] += 1
        # occupied=0 stays class 0 (free)
        voxel_semantics = semantics

        # Apply BEV flips if augmentation was applied
        if results.get('flip_dx', False):
            voxel_semantics = torch.flip(voxel_semantics, [0])
            occupied = torch.flip(occupied, [0])
        if results.get('flip_dy', False):
            voxel_semantics = torch.flip(voxel_semantics, [1])
            occupied = torch.flip(occupied, [1])

        results['voxel_semantics'] = voxel_semantics
        results['mask_lidar'] = occupied      # only LiDAR-observed voxels for eval
        # mask_camera = all voxels in grid are camera-observable (4 fisheye cameras cover 360 deg)
        results['mask_camera'] = torch.ones_like(occupied)

        return results


@PIPELINES.register_module()
class FisheyePointToMultiViewDepth(object):
    """Project LiDAR points to multi-view depth maps for fisheye cameras.

    Uses Kannala-Brandt projection for fisheye.
    """

    def __init__(self, grid_config, downsample=1):
        self.downsample = downsample
        self.grid_config = grid_config

    def _kb_project(self, points_cam, intrins, distortions):
        """Project 3D camera points to image using Kannala-Brandt model.

        Args:
            points_cam: (N, 3) in camera frame
            intrins: (3, 3) camera intrinsic matrix
            distortions: (4,) distortion coefficients [k1, k2, k3, k4]

        Returns:
            uv: (N, 2) pixel coordinates
        """
        fx, fy = intrins[0, 0], intrins[1, 1]
        cx, cy = intrins[0, 2], intrins[1, 2]
        k1, k2, k3, k4 = distortions

        x = points_cam[:, 0]
        y = points_cam[:, 1]
        z = points_cam[:, 2]

        r = torch.sqrt(x * x + y * y)
        theta = torch.atan2(r, z + 1e-8)

        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        theta8 = theta4 * theta4

        theta_d = theta * (1 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8)

        scaling = theta_d / (r + 1e-8)
        x_distorted = scaling * x
        y_distorted = scaling * y

        u = fx * x_distorted + cx
        v = fy * y_distorted + cy

        return torch.stack([u, v], dim=-1)

    def points2depthmap(self, points, height, width):
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.downsample)
        depth = points[:, 2]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & \
                (coor[:, 1] >= 0) & (coor[:, 1] < height) & \
                (depth < self.grid_config['depth'][1]) & \
                (depth >= self.grid_config['depth'][0])
        coor, depth = coor[kept1], depth[kept1]
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]
        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map

    def _build_4x4_transform(self, rot, trans):
        """Build 4x4 transform from rotation + translation.

        rot can be quaternion (4,) or 3x3 rotation matrix.
        trans is a 3-element vector.
        """
        T = np.eye(4, dtype=np.float32)
        if isinstance(rot, (list, np.ndarray)):
            rot = np.array(rot)
            if rot.shape == (4,):
                T[:3, :3] = Quaternion(rot[0], rot[1], rot[2], rot[3]).rotation_matrix
            elif rot.shape == (3, 3):
                T[:3, :3] = rot
            elif rot.shape == (9,):
                T[:3, :3] = rot.reshape(3, 3)
        T[:3, 3] = np.array(trans[:3])
        return torch.from_numpy(T)

    def __call__(self, results):
        points_lidar = results['points']
        imgs, sensor2egos, ego2globals, intrins = results['img_inputs'][:4]
        post_rots, post_trans, bda = results['img_inputs'][4:7]
        distortions = results['img_inputs'][7] if len(results['img_inputs']) > 7 else None

        depth_map_list = []
        for cid in range(len(results['cam_names'])):
            cam_name = results['cam_names'][cid]
            cam_data = results['curr']['cams'][cam_name]

            # Build lidar2ego transform
            lidar2ego = self._build_4x4_transform(
                results['curr']['lidar2ego_rotation'],
                results['curr']['lidar2ego_translation'])

            # Build cam2ego from cam_to_base (4x4 matrix)
            cam2ego = torch.from_numpy(
                np.array(cam_data['cam_to_base'], dtype=np.float32))

            # lidar2cam = inv(cam2ego) @ lidar2ego
            lidar2cam = torch.inverse(cam2ego.double()).matmul(lidar2ego.double()).float()

            points_cam = points_lidar.tensor[:, :3].matmul(
                lidar2cam[:3, :3].T) + lidar2cam[:3, 3].unsqueeze(0)

            d = points_cam[:, 2]
            valid_depth = (d > 0) & (d < self.grid_config['depth'][1])

            if distortions is not None and valid_depth.sum() > 0:
                uv = self._kb_project(points_cam[valid_depth],
                                      intrins[cid], distortions[cid])
                uv_depth = torch.cat([uv, d[valid_depth].unsqueeze(-1)], dim=-1)
            else:
                uv_depth = torch.stack([
                    intrins[cid, 0, 0] * points_cam[:, 0] / (points_cam[:, 2] + 1e-8) + intrins[cid, 0, 2],
                    intrins[cid, 1, 1] * points_cam[:, 1] / (points_cam[:, 2] + 1e-8) + intrins[cid, 1, 2],
                    points_cam[:, 2]
                ], dim=-1)

            # Apply post augmentation
            points_img = uv_depth.matmul(post_rots[cid].T) + post_trans[cid:cid + 1, :]
            depth_map = self.points2depthmap(points_img, imgs.shape[2], imgs.shape[3])
            depth_map_list.append(depth_map)

        depth_map = torch.stack(depth_map_list)
        results['gt_depth'] = depth_map
        return results
