#!/usr/bin/env python
# Visualization tool for fisheye occupancy: 3D GT vs prediction comparison

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm

import torch
import mmcv
from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model


CLASS_NAMES = ['unknown', 'person', 'table', 'chair', 'floor', 'car']
CLASS_COLORS = {
    0: '#c8c8c8',  # unknown - gray
    1: '#dc143c',  # person - red
    2: '#771120',  # table - dark red
    3: '#00008e',  # chair - dark blue
    4: '#003c64',  # floor - teal
    5: '#0000e6',  # car - bright blue
}

OBSERVED_CLASSES = [1, 2, 3, 4, 5]  # non-empty classes to show in legend


def build_color_map():
    cmap = np.zeros((6, 3), dtype=np.uint8)
    for i in range(6):
        hex_color = CLASS_COLORS[i].lstrip('#')
        cmap[i] = [int(hex_color[j:j+2], 16) for j in (0, 2, 4)]
    return cmap


def _get_3d_points(occ, stride=4):
    """Extract occupied voxels from occupancy grid, downsampled by stride.

    Args:
        occ: (Dx, Dy, Dz) uint8 array, class labels 0-5
        stride: downsample stride for x,y dimensions

    Returns:
        xs, ys, zs: downsampled occupied voxel coordinates
        colors: list of hex color strings per point
    """
    Dx, Dy, Dz = occ.shape
    xs, ys, zs = [], [], []
    colors = []

    for x in range(0, Dx, stride):
        for y in range(0, Dy, stride):
            for z in range(Dz):
                cls_id = occ[x, y, z]
                if cls_id == 0:
                    continue
                xs.append(x)
                ys.append(y)
                zs.append(z)
                colors.append(CLASS_COLORS[cls_id])

    return np.array(xs), np.array(ys), np.array(zs), colors


def draw_3d_scene(occ, title, ax, stride=4, cmap=None):
    """Draw 3D occupancy as scatter plot."""
    ax.set_title(title, fontsize=10, pad=0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    Dx, Dy, Dz = occ.shape
    ax.set_xlim(0, Dx)
    ax.set_ylim(0, Dy)
    ax.set_zlim(0, Dz)

    # Set consistent aspect
    ax.set_box_aspect((Dx, Dy, Dz))

    # Use a fixed elevation/azimuth for side-by-side comparison
    ax.view_init(elev=25, azim=-60)

    xs, ys, zs, colors = _get_3d_points(occ, stride=stride)

    if len(xs) == 0:
        return

    # Group by color to reduce scatter calls
    color_to_mask = {}
    for i, c in enumerate(colors):
        color_to_mask.setdefault(c, []).append(i)

    for color, indices in color_to_mask.items():
        idx = np.array(indices)
        ax.scatter(xs[idx], ys[idx], zs[idx],
                   c=color, marker='s', s=2, alpha=0.7, edgecolors='none')


def draw_bev(occ, title, ax, cmap):
    """Draw BEV projection of occupancy by collapsing z dimension."""
    H, W, D = occ.shape
    mask = occ > 0
    bev = np.zeros((H, W), dtype=np.int32)
    for z in range(D - 1, -1, -1):
        layer = mask[:, :, z]
        bev[layer] = occ[:, :, z][layer]

    img = np.zeros((H, W, 3), dtype=np.uint8)
    for cls_id in range(6):
        img[bev == cls_id] = cmap[cls_id]

    ax.imshow(img[::-1, :], origin='lower')
    ax.set_title(title, fontsize=10)
    ax.axis('off')


def main():
    parser = argparse.ArgumentParser(description='Fisheye occupancy visualization')
    parser.add_argument('--config', required=True, help='config file path')
    parser.add_argument('--weights', required=True, help='checkpoint file')
    parser.add_argument('--viz-dir', default='vis_fisheye', help='output directory')
    parser.add_argument('--max-samples', type=int, default=20, help='max samples to viz')
    parser.add_argument('--mode', default='3d', choices=['3d', 'bev'],
                        help='visualization mode: 3d scatter or bev+diff')
    parser.add_argument('--stride', type=int, default=3,
                        help='downsample stride for 3d mode (smaller = denser)')
    parser.add_argument('--device', default='cuda:0', help='device')
    args = parser.parse_args()

    mmcv.mkdir_or_exist(args.viz_dir)

    cfg = Config.fromfile(args.config)
    cfg.model.train_cfg = None

    # Load plugin to register custom modules
    if getattr(cfg, 'plugin', False):
        import importlib
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if hasattr(cfg, 'plugin_dir'):
            plugin_dir = cfg.plugin_dir
            _module_dir = os.path.dirname(plugin_dir)
            _module_dir = _module_dir.split('/')
            _module_path = _module_dir[0]
            for m in _module_dir[1:]:
                _module_path = _module_path + '.' + m
            print(_module_path)
            plg_lib = importlib.import_module(_module_path)

    # Build dataset
    dataset = build_dataset(cfg.data.val)
    print(f'Dataset: {len(dataset)} samples')

    # Build model and load checkpoint
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    checkpoint = torch.load(args.weights, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.to(args.device)
    model.eval()

    cmap = build_color_map()
    legend_elements = [Patch(facecolor=CLASS_COLORS[i], label=CLASS_NAMES[i])
                       for i in OBSERVED_CLASSES]

    for idx in tqdm(range(min(len(dataset), args.max_samples)), desc='Visualizing'):
        data = dataset[idx]
        sample_token = dataset.data_infos[idx]['token']

        # Handle MultiScaleFlipAug3D wrapping
        if isinstance(data, list):
            sample = data[0]
        else:
            sample = data

        meta = sample['img_metas']
        if isinstance(meta, list):
            meta = meta[0]
        img_inputs = sample['img_inputs']
        if isinstance(img_inputs, list) and len(img_inputs) > 0:
            img_inputs = img_inputs[0]

        # Add batch dimension
        batch_img_inputs = tuple(
            t.unsqueeze(0).to(args.device) if isinstance(t, torch.Tensor) else t
            for t in img_inputs
        )

        # Run inference
        with torch.no_grad():
            result = model.simple_test(
                points=None,
                img_metas=[meta],
                img=batch_img_inputs,
            )

        pred = result[0]
        if isinstance(pred, dict):
            pred = pred['pred_occ']
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()

        # Load GT
        occ_gt_path = dataset.data_infos[idx]['occ_path']
        occ_gt = np.load(occ_gt_path, allow_pickle=True)
        gt = occ_gt['semantics']

        if args.mode == '3d':
            fig = plt.figure(figsize=(16, 8))

            ax1 = fig.add_subplot(121, projection='3d',
                                   computed_zorder=False)
            draw_3d_scene(pred, 'Prediction', ax1, stride=args.stride)

            ax2 = fig.add_subplot(122, projection='3d',
                                   computed_zorder=False)
            draw_3d_scene(gt, 'Ground Truth', ax2, stride=args.stride)

            ax1.legend(handles=legend_elements, loc='upper right',
                       fontsize=7, ncol=1, markerscale=3)

            plt.suptitle(f'Sample: {sample_token}', fontsize=12, y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.95])

        else:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            draw_bev(gt, 'GT Occupancy', axes[0], cmap)
            draw_bev(pred, 'Prediction', axes[1], cmap)

            diff = np.zeros_like(gt, dtype=np.int32)
            diff[(gt == pred) & (gt > 0)] = 1
            diff[(gt != pred) & (gt > 0)] = 2
            diff_img = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
            diff_bev = np.max(diff, axis=-1)
            diff_img[diff_bev == 1] = [0, 255, 0]
            diff_img[diff_bev == 2] = [255, 0, 0]
            axes[2].imshow(diff_img[::-1, :], origin='lower')
            axes[2].set_title('Diff (green=correct, red=incorrect)', fontsize=10)
            axes[2].axis('off')

            plt.suptitle(f'Sample: {sample_token}', fontsize=12)
            plt.tight_layout()

        save_path = os.path.join(args.viz_dir, f'{sample_token}.png')
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()

    print(f'\nDone. Results saved to {args.viz_dir}/')


if __name__ == '__main__':
    main()
