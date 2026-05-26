# Copyright (c) OpenMMLab. All rights reserved.
import os
import mmcv
import numpy as np
import torch
from tqdm import tqdm

from mmdet3d.datasets import DATASETS
from mmdet3d.datasets.custom_3d import Custom3DDataset
from ..core.evaluation.fisheye_metrics import FisheyeMetric_mIoU


@DATASETS.register_module()
class FisheyeDataset(Custom3DDataset):
    r"""Fisheye Dataset for 3D occupancy prediction.

    Reads fisheye PKL format with 4 Kannala-Brandt cameras.

    Args:
        ann_file (str): Path of annotation file (PKL).
        pipeline (list[dict], optional): Pipeline for data processing.
        data_root (str): Path of dataset root.
        classes (tuple[str], optional): Classes in the dataset.
        test_mode (bool): Whether the dataset is in test mode.
    """

    def __init__(self,
                 ann_file,
                 pipeline=None,
                 data_root=None,
                 classes=None,
                 load_interval=1,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 img_info_prototype='bevdet',
                 **kwargs):
        self.load_interval = load_interval
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode)
        self.img_info_prototype = img_info_prototype

    def load_annotations(self, ann_file):
        """Load annotations from PKL file."""
        data = mmcv.load(ann_file, file_format='pkl')
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        self.metadata = data.get('metadata', {})
        self.version = self.metadata.get('version', 'fisheye_v1')
        return data_infos

    def get_data_info(self, index):
        """Get data info according to the given index."""
        info = self.data_infos[index]
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info.get('lidar_path', ''),
            sweeps=info.get('sweeps', []),
            timestamp=info['timestamp'] / 1e6,
        )

        if self.img_info_prototype == 'bevdet':
            input_dict.update(dict(curr=info))
        else:
            # mmcv prototype: build image paths and lidar2img matrices
            image_paths = []
            lidar2img_rts = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)
            input_dict.update(dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
            ))

        return input_dict


@DATASETS.register_module()
class FisheyeDatasetOccpancy(FisheyeDataset):
    """Fisheye Dataset for occupancy prediction."""

    def get_data_info(self, index):
        """Get data info with occupancy GT path."""
        input_dict = super(FisheyeDatasetOccpancy, self).get_data_info(index)
        input_dict['occ_gt_path'] = self.data_infos[index]['occ_path']
        return input_dict

    def evaluate(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        """Evaluate occupancy predictions using 6-class mIoU.

        Args:
            occ_results: list of occupancy predictions, each is
                (Dx, Dy, Dz) numpy array or dict with 'pred_occ' key.
        """
        metric = eval_kwargs.get('metric', ['map'])
        if isinstance(metric, (list, tuple)):
            metric = metric[0]

        self.occ_eval_metrics = FisheyeMetric_mIoU(
            num_classes=7,
            use_lidar_mask=False,
            use_image_mask=False,
            use_occupied_mask=False)

        print('\nStarting Fisheye OCC Evaluation...')
        for index, occ_pred in enumerate(tqdm(occ_results)):
            info = self.data_infos[index]
            occ_gt = np.load(info['occ_path'], allow_pickle=True)
            gt_semantics = occ_gt['semantics']    # (Dx, Dy, Dz), labels 0-6
            mask_camera = occ_gt['mask_camera']   # camera visibility mask
            mask_lidar = occ_gt['mask_lidar']     # LiDAR visibility mask

            pred = occ_pred['pred_occ'] if isinstance(occ_pred, dict) else occ_pred
            self.occ_eval_metrics.add_batch(
                pred,           # (Dx, Dy, Dz)
                gt_semantics,   # (Dx, Dy, Dz), already 0-6
                mask_lidar=mask_lidar,
                mask_camera=mask_camera,
            )

            if show_dir is not None:
                mmcv.mkdir_or_exist(show_dir)
                sample_token = info['token']
                save_path = os.path.join(show_dir, f'{sample_token}.npz')
                np.savez_compressed(save_path,
                                    pred=pred,
                                    gt=gt_semantics,
                                    mask_camera=mask_camera,
                                    mask_lidar=mask_lidar,
                                    sample_token=sample_token)

        eval_results = self.occ_eval_metrics.count_miou()
        return eval_results
