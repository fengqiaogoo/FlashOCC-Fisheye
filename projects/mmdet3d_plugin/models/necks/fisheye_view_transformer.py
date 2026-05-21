# Copyright (c) OpenMMLab. All rights reserved.
# Fisheye LSS View Transformer with Kannala-Brandt inverse projection

import torch
import torch.nn as nn
from mmcv.runner import BaseModule, force_fp32
from mmdet3d.models.builder import NECKS
from ...ops import bev_pool_v2
from ..model_utils import DepthNet
from torch.cuda.amp.autocast_mode import autocast
import torch.nn.functional as F


@NECKS.register_module(force=True)
class LSSViewTransformerFisheye(BaseModule):
    r"""Lift-Splat-Shoot view transformer for fisheye cameras.

    Uses Kannala-Brandt inverse projection instead of pinhole K^{-1}
    for converting image coordinates to camera 3D coordinates.

    Args:
        grid_config (dict): Grid config in format
            (lower_bound, upper_bound, interval) for {x,y,z,depth}.
        input_size (tuple(int)): Input image size (H, W).
        downsample (int): Down sample factor from input to feature size.
        in_channels (int): Channels of input feature.
        out_channels (int): Channels of transformed feature.
        accelerate (bool): Pre-compute frustum coords.
        sid (bool): Spacing Increasing Discretization.
        collapse_z (bool): Collapse z dimension.
    """

    def __init__(self, grid_config, input_size, downsample=16,
                 in_channels=512, out_channels=64, accelerate=False,
                 sid=False, collapse_z=True):
        super(LSSViewTransformerFisheye, self).__init__()
        self.grid_config = grid_config
        self.downsample = downsample
        self.create_grid_infos(**grid_config)
        self.sid = sid
        self.frustum = self.create_frustum(grid_config['depth'],
                                           input_size, downsample)
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.depth_net = nn.Conv2d(
            in_channels, self.D + self.out_channels, kernel_size=1, padding=0)
        self.accelerate = accelerate
        self.initial_flag = True
        self.collapse_z = collapse_z

    def create_grid_infos(self, x, y, z, **kwargs):
        self.grid_lower_bound = torch.Tensor([cfg[0] for cfg in [x, y, z]])
        self.grid_interval = torch.Tensor([cfg[2] for cfg in [x, y, z]])
        self.grid_size = torch.Tensor([(cfg[1] - cfg[0]) / cfg[2]
                                       for cfg in [x, y, z]])

    def create_frustum(self, depth_cfg, input_size, downsample):
        H_in, W_in = input_size
        H_feat = H_in // downsample
        W_feat = W_in // downsample
        d = torch.arange(*depth_cfg, dtype=torch.float)\
            .view(-1, 1, 1).expand(-1, H_feat, W_feat)
        self.D = d.shape[0]
        if self.sid:
            d_sid = torch.arange(self.D).float()
            depth_cfg_t = torch.tensor(depth_cfg).float()
            d_sid = torch.exp(torch.log(depth_cfg_t[0]) + d_sid / (self.D - 1) *
                              torch.log((depth_cfg_t[1] - 1) / depth_cfg_t[0]))
            d = d_sid.view(-1, 1, 1).expand(-1, H_feat, W_feat)
        x = torch.linspace(0, W_in - 1, W_feat, dtype=torch.float)\
            .view(1, 1, W_feat).expand(self.D, H_feat, W_feat)
        y = torch.linspace(0, H_in - 1, H_feat, dtype=torch.float)\
            .view(1, H_feat, 1).expand(self.D, H_feat, W_feat)
        return torch.stack((x, y, d), -1)

    def kb_inverse_project(self, points, cam2imgs, distortions, sensor2ego):
        """Kannala-Brandt inverse: (u, v, d) -> camera 3D -> ego 3D.

        Args:
            points: (B, N, D, fH, fW, 3) — frustum points (u, v, d)
            cam2imgs: (B, N, 3, 3) — camera intrinsic matrices
            distortions: (B, N, 4) — KB distortion [k1, k2, k3, k4]
            sensor2ego: (B, N, 4, 4) — camera to ego transform

        Returns:
            points_ego: (B, N, D, fH, fW, 3)
        """
        B, N, D, fH, fW, _ = points.shape

        # Extract intrinsic params
        fx = cam2imgs[..., 0, 0].view(B, N, 1, 1, 1)
        fy = cam2imgs[..., 1, 1].view(B, N, 1, 1, 1)
        cx = cam2imgs[..., 0, 2].view(B, N, 1, 1, 1)
        cy = cam2imgs[..., 1, 2].view(B, N, 1, 1, 1)

        # Distortion params
        k1 = distortions[..., 0:1].view(B, N, 1, 1, 1)
        k2 = distortions[..., 1:2].view(B, N, 1, 1, 1)
        k3 = distortions[..., 2:3].view(B, N, 1, 1, 1)
        k4 = distortions[..., 3:4].view(B, N, 1, 1, 1)

        u = points[..., 0]
        v = points[..., 1]
        d = points[..., 2]

        # Normalize pixel coordinates
        x_norm = (u - cx) / fx
        y_norm = (v - cy) / fy
        r_d = torch.sqrt(x_norm ** 2 + y_norm ** 2)

        # Newton's method to solve for theta
        # theta_d(theta) = theta + k1*theta^3 + k2*theta^5 + k3*theta^7 + k4*theta^9
        theta = r_d.clone()
        for _ in range(5):
            t2 = theta * theta
            t4 = t2 * t2
            t6 = t4 * t2
            t8 = t4 * t4
            theta_d = theta + k1 * theta * t2 + k2 * theta * t4 + \
                      k3 * theta * t6 + k4 * theta * t8
            theta_d_prime = 1.0 + 3.0 * k1 * t2 + 5.0 * k2 * t4 + \
                            7.0 * k3 * t6 + 9.0 * k4 * t8
            theta = theta - (theta_d - r_d) / (theta_d_prime + 1e-8)

        # Camera 3D coordinates from angle and depth
        tan_theta = torch.tan(theta)
        scaling = tan_theta / (r_d + 1e-8)
        Xc = scaling * x_norm * d
        Yc = scaling * y_norm * d
        Zc = d

        points_cam = torch.stack([Xc, Yc, Zc], dim=-1).unsqueeze(-1)

        # Rotate to ego frame
        R = sensor2ego[..., :3, :3]
        points_ego = R.view(B, N, 1, 1, 1, 3, 3).matmul(points_cam).squeeze(-1)
        points_ego = points_ego + sensor2ego[..., :3, 3].view(B, N, 1, 1, 1, 3)

        return points_ego

    def get_ego_coor(self, sensor2ego, ego2global, cam2imgs, post_rots,
                     post_trans, bda, distortions=None):
        """Calculate frustum points in ego coordinate system.

        Uses Kannala-Brandt inverse for fisheye cameras.

        Args:
            sensor2ego: (B, N, 4, 4)
            ego2global: (B, N, 4, 4) (unused, kept for interface compat)
            cam2imgs: (B, N, 3, 3)
            post_rots: (B, N, 3, 3)
            post_trans: (B, N, 3)
            bda: (B, 3, 3)
            distortions: (B, N, 4) — KB distortion coefficients

        Returns:
            points: (B, N, D, fH, fW, 3) in ego/LiDAR coordinate
        """
        B, N, _, _ = sensor2ego.shape

        # Step 1: Undo image augmentation
        points = self.frustum.to(sensor2ego) - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3)\
            .matmul(points.unsqueeze(-1))
        points = points.squeeze(-1)  # (B, N, D, fH, fW, 3)

        # Step 2: Kannala-Brandt inverse projection
        if distortions is not None:
            points = self.kb_inverse_project(points, cam2imgs, distortions, sensor2ego)
        else:
            # Fallback to pinhole
            points = torch.cat(
                (points[..., :2] * points[..., 2:3], points[..., 2:3]), dim=-1)
            points = points.unsqueeze(-1)
            combine = sensor2ego[:, :, :3, :3].matmul(torch.inverse(cam2imgs))
            points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
            points = points + sensor2ego[:, :, :3, 3].view(B, N, 1, 1, 1, 3)

        # Step 3: Apply BEV data augmentation
        points = bda.view(B, 1, 1, 1, 1, 3, 3)\
            .matmul(points.unsqueeze(-1)).squeeze(-1)
        return points

    def init_acceleration_v2(self, coor):
        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.voxel_pooling_prepare_v2(coor)
        self.ranks_bev = ranks_bev.int().contiguous()
        self.ranks_feat = ranks_feat.int().contiguous()
        self.ranks_depth = ranks_depth.int().contiguous()
        self.interval_starts = interval_starts.int().contiguous()
        self.interval_lengths = interval_lengths.int().contiguous()

    def voxel_pooling_v2(self, coor, depth, feat):
        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.voxel_pooling_prepare_v2(coor)
        if ranks_feat is None:
            dummy = torch.zeros(size=[
                feat.shape[0], feat.shape[2],
                int(self.grid_size[2]),
                int(self.grid_size[1]),
                int(self.grid_size[0])
            ]).to(feat)
            dummy = torch.cat(dummy.unbind(dim=2), 1)
            return dummy
        feat = feat.permute(0, 1, 3, 4, 2)
        bev_feat_shape = (depth.shape[0], int(self.grid_size[2]),
                          int(self.grid_size[1]), int(self.grid_size[0]),
                          feat.shape[-1])
        bev_feat = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                               bev_feat_shape, interval_starts, interval_lengths)
        if self.collapse_z:
            bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)
        return bev_feat

    def voxel_pooling_prepare_v2(self, coor):
        B, N, D, H, W, _ = coor.shape
        num_points = B * N * D * H * W
        ranks_depth = torch.arange(
            0, num_points, dtype=torch.int, device=coor.device)
        ranks_feat = torch.arange(
            0, num_points // D, dtype=torch.int, device=coor.device)
        ranks_feat = ranks_feat.reshape(B, N, 1, H, W)
        ranks_feat = ranks_feat.expand(B, N, D, H, W).flatten()

        coor = ((coor - self.grid_lower_bound.to(coor)) /
                self.grid_interval.to(coor))
        coor = coor.long().view(num_points, 3)
        batch_idx = torch.arange(0, B).reshape(B, 1).\
            expand(B, num_points // B).reshape(num_points, 1).to(coor)
        coor = torch.cat((coor, batch_idx), 1)

        kept = (coor[:, 0] >= 0) & (coor[:, 0] < self.grid_size[0]) & \
               (coor[:, 1] >= 0) & (coor[:, 1] < self.grid_size[1]) & \
               (coor[:, 2] >= 0) & (coor[:, 2] < self.grid_size[2])
        if len(kept) == 0:
            return None, None, None, None, None

        coor, ranks_depth, ranks_feat = \
            coor[kept], ranks_depth[kept], ranks_feat[kept]

        ranks_bev = coor[:, 3] * (
            self.grid_size[2] * self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 2] * (self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 1] * self.grid_size[0] + coor[:, 0]
        order = ranks_bev.argsort()
        ranks_bev, ranks_depth, ranks_feat = \
            ranks_bev[order], ranks_depth[order], ranks_feat[order]

        kept = torch.ones(
            ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
        interval_starts = torch.where(kept)[0].int()
        if len(interval_starts) == 0:
            return None, None, None, None, None
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
        return ranks_bev.int().contiguous(), ranks_depth.int().contiguous(
        ), ranks_feat.int().contiguous(), interval_starts.int().contiguous(
        ), interval_lengths.int().contiguous()

    def pre_compute(self, input):
        if self.initial_flag:
            coor = self.get_ego_coor(*input[1:8])
            self.init_acceleration_v2(coor)
            self.initial_flag = False

    def view_transform_core(self, input, depth, tran_feat):
        B, N, C, H, W = input[0].shape
        if self.accelerate:
            feat = tran_feat.view(B, N, self.out_channels, H, W)
            feat = feat.permute(0, 1, 3, 4, 2)
            depth = depth.view(B, N, self.D, H, W)
            bev_feat_shape = (depth.shape[0], int(self.grid_size[2]),
                              int(self.grid_size[1]), int(self.grid_size[0]),
                              feat.shape[-1])
            bev_feat = bev_pool_v2(depth, feat, self.ranks_depth,
                                   self.ranks_feat, self.ranks_bev,
                                   bev_feat_shape, self.interval_starts,
                                   self.interval_lengths)
            bev_feat = bev_feat.squeeze(2)
        else:
            coor = self.get_ego_coor(*input[1:8])
            bev_feat = self.voxel_pooling_v2(
                coor, depth.view(B, N, self.D, H, W),
                tran_feat.view(B, N, self.out_channels, H, W))
        return bev_feat, depth

    def view_transform(self, input, depth, tran_feat):
        if self.accelerate:
            self.pre_compute(input)
        return self.view_transform_core(input, depth, tran_feat)

    def forward(self, input):
        """Transform image-view feature into bird-eye-view feature.

        Args:
            input (list):
                imgs: (B, N, C, H, W)
                sensor2egos: (B, N, 4, 4)
                ego2globals: (B, N, 4, 4)
                intrins: (B, N, 3, 3)
                post_rots: (B, N, 3, 3)
                post_trans: (B, N, 3)
                bda_rot: (B, 3, 3)
                distortions: (B, N, 4)
                [mlp_input]: (B, N, 27) optional

        Returns:
            bev_feat: (B, C, Dy, Dx)
            depth: (B*N, D, fH, fW)
        """
        x = input[0]
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.depth_net(x)
        depth_digit = x[:, :self.D, ...]
        tran_feat = x[:, self.D:self.D + self.out_channels, ...]
        depth = depth_digit.softmax(dim=1)
        return self.view_transform(input, depth, tran_feat)


@NECKS.register_module()
class LSSViewTransformerBEVDepthFisheye(LSSViewTransformerFisheye):
    """Fisheye LSS with DepthNet and depth supervision."""

    def __init__(self, loss_depth_weight=3.0, depthnet_cfg=dict(), **kwargs):
        super(LSSViewTransformerBEVDepthFisheye, self).__init__(**kwargs)
        self.loss_depth_weight = loss_depth_weight
        self.depth_net = DepthNet(
            in_channels=self.in_channels,
            mid_channels=self.in_channels,
            context_channels=self.out_channels,
            depth_channels=self.D,
            **depthnet_cfg)

    def get_mlp_input(self, sensor2ego, ego2global, intrin, post_rot,
                      post_tran, bda):
        B, N, _, _ = sensor2ego.shape
        bda = bda.view(B, 1, 3, 3).repeat(1, N, 1, 1)
        mlp_input = torch.stack([
            intrin[:, :, 0, 0],
            intrin[:, :, 1, 1],
            intrin[:, :, 0, 2],
            intrin[:, :, 1, 2],
            post_rot[:, :, 0, 0],
            post_rot[:, :, 0, 1],
            post_tran[:, :, 0],
            post_rot[:, :, 1, 0],
            post_rot[:, :, 1, 1],
            post_tran[:, :, 1],
            bda[:, :, 0, 0],
            bda[:, :, 0, 1],
            bda[:, :, 1, 0],
            bda[:, :, 1, 1],
            bda[:, :, 2, 2]
        ], dim=-1)
        sensor2ego_flat = sensor2ego[:, :, :3, :].reshape(B, N, -1)
        mlp_input = torch.cat([mlp_input, sensor2ego_flat], dim=-1)
        return mlp_input

    def forward(self, input, stereo_metas=None):
        # Unpack 9 elements: x + (8 from prepare_inputs: s2keyego, ego2g, K, pR, pT, bda, dist, mlp)
        (x, rots, trans, intrins, post_rots, post_trans, bda,
         distortions, mlp_input) = input[:9]

        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.depth_net(x, mlp_input, stereo_metas)
        depth_digit = x[:, :self.D, ...]
        tran_feat = x[:, self.D:self.D + self.out_channels, ...]
        depth = depth_digit.softmax(dim=1)
        # Pack with distortions
        new_input = [input[0], rots, trans, intrins, post_rots, post_trans,
                     bda, distortions]
        bev_feat, depth = self.view_transform(new_input, depth, tran_feat)
        return bev_feat, depth

    def get_downsampled_gt_depth(self, gt_depths):
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N,
                                   H // self.downsample, self.downsample,
                                   W // self.downsample, self.downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample,
                                   W // self.downsample)
        if not self.sid:
            gt_depths = (gt_depths - (self.grid_config['depth'][0] -
                                      self.grid_config['depth'][2])) / \
                        self.grid_config['depth'][2]
        else:
            gt_depths = torch.log(gt_depths) - torch.log(
                torch.tensor(self.grid_config['depth'][0]).float())
            gt_depths = gt_depths * (self.D - 1) / torch.log(
                torch.tensor(self.grid_config['depth'][1] - 1.).float() /
                self.grid_config['depth'][0])
            gt_depths = gt_depths + 1.
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0),
                                gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(
            gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:, 1:]
        return gt_depths.float()

    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds):
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds, depth_labels, reduction='none').sum() / \
                max(1.0, fg_mask.sum())
        return self.loss_depth_weight * depth_loss
