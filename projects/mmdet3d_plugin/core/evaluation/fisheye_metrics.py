# Fisheye occupancy metrics: 6-class mIoU evaluation

import numpy as np
import torch


class FisheyeMetric_mIoU:
    """Mean IoU metric for 6-class fisheye occupancy prediction.

    Args:
        num_classes (int): Number of semantic classes (default 6).
        use_lidar_mask (bool): Whether to use LiDAR visibility mask.
        use_image_mask (bool): Whether to use camera visibility mask.
        use_occupied_mask (bool): Whether to filter by occupied mask.
    """

    def __init__(self, num_classes=7, use_lidar_mask=False,
                 use_image_mask=False, use_occupied_mask=True):
        self.num_classes = num_classes
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.use_occupied_mask = use_occupied_mask
        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes),
                                         dtype=np.int64)

    def add_batch(self, pred, gt, mask_lidar=None, mask_camera=None):
        """Add a batch of predictions.

        Args:
            pred: (Dx, Dy, Dz) or torch.Tensor, predicted class labels
            gt: (Dx, Dy, Dz) numpy, ground truth class labels
            mask_lidar: optional LiDAR visibility mask
            mask_camera: optional camera/occupied mask
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.cpu().numpy()

        pred = pred.astype(np.int64)
        gt = gt.astype(np.int64)

        # Build valid mask
        valid_mask = np.ones_like(gt, dtype=bool)
        if self.use_occupied_mask and mask_camera is not None:
            if isinstance(mask_camera, torch.Tensor):
                mask_camera = mask_camera.cpu().numpy()
            valid_mask = valid_mask & (mask_camera > 0)
        if self.use_lidar_mask and mask_lidar is not None:
            if isinstance(mask_lidar, torch.Tensor):
                mask_lidar = mask_lidar.cpu().numpy()
            valid_mask = valid_mask & (mask_lidar > 0)
        if self.use_image_mask and mask_camera is not None:
            if isinstance(mask_camera, torch.Tensor):
                mask_camera = mask_camera.cpu().numpy()
            valid_mask = valid_mask & (mask_camera > 0)

        # Filter to valid classes only
        valid_mask = valid_mask & (gt < self.num_classes) & (pred < self.num_classes)

        gt_valid = gt[valid_mask]
        pred_valid = pred[valid_mask]

        # Accumulate confusion matrix
        np.add.at(self.confusion_matrix, (gt_valid, pred_valid), 1)

    def count_miou(self):
        """Compute mIoU from accumulated confusion matrix."""
        intersection = np.diag(self.confusion_matrix)
        union = (self.confusion_matrix.sum(axis=0) +
                 self.confusion_matrix.sum(axis=1) -
                 intersection)

        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = intersection.astype(np.float32) / union.astype(np.float32)
            iou = np.nan_to_num(iou, nan=0.0, posinf=0.0, neginf=0.0)

        miou = iou.mean()

        results = {'mIoU': miou}
        class_names = ['free', 'unknown', 'person', 'table', 'chair', 'floor', 'car']
        for i in range(self.num_classes):
            results[f'IoU_{class_names[i]}'] = iou[i]

        # Print confusion matrix for debugging
        print('\nConfusion matrix (rows=GT, cols=Pred):')
        header = '         ' + ''.join(f'{n:>8s}' for n in class_names)
        print(header)
        for i in range(self.num_classes):
            row = f'{class_names[i]:>8s}: ' + ''.join(f'{self.confusion_matrix[i,j]:>8d}' for j in range(self.num_classes))
            print(row)

        return results


class FisheyeMetric_FScore:
    """F-score metric for 6-class fisheye occupancy prediction."""

    def __init__(self, num_classes=7, threshold=0.5):
        self.num_classes = num_classes
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.gt_counts = np.zeros(self.num_classes, dtype=np.int64)
        self.pred_counts = np.zeros(self.num_classes, dtype=np.int64)
        self.tp_counts = np.zeros(self.num_classes, dtype=np.int64)

    def add_batch(self, pred, gt, occupied_mask=None):
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.cpu().numpy()

        pred = pred.astype(np.int64)
        gt = gt.astype(np.int64)

        if occupied_mask is not None:
            if isinstance(occupied_mask, torch.Tensor):
                occupied_mask = occupied_mask.cpu().numpy()
            valid = (occupied_mask > 0) & (gt < self.num_classes) & (pred < self.num_classes)
        else:
            valid = (gt < self.num_classes) & (pred < self.num_classes)

        for c in range(self.num_classes):
            self.gt_counts[c] += ((gt == c) & valid).sum()
            self.pred_counts[c] += ((pred == c) & valid).sum()
            self.tp_counts[c] += ((pred == c) & (gt == c) & valid).sum()

    def count_fscore(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            precision = self.tp_counts.astype(np.float32) / \
                np.maximum(self.pred_counts.astype(np.float32), 1)
            recall = self.tp_counts.astype(np.float32) / \
                np.maximum(self.gt_counts.astype(np.float32), 1)
            precision = np.nan_to_num(precision, nan=0.0)
            recall = np.nan_to_num(recall, nan=0.0)

            fscore = 2 * precision * recall / np.maximum(precision + recall, 1e-8)
            fscore = np.nan_to_num(fscore, nan=0.0)

        return {'mFScore': fscore.mean()}
