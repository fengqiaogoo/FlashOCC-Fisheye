#!/usr/bin/env python
# Fisheye occupancy visualization — composite: camera images + 3D occ scene
# Matches the style of tools/analysis_tools/vis_occ.py for fisheye

import argparse
import os
import sys
import numpy as np
import cv2
import torch
import mmcv
from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from tqdm import tqdm

# ---- Fisheye constants ----
CLASS_NAMES = ['unknown', 'person', 'table', 'chair', 'floor', 'car']
OBSERVED_CLASSES = [1, 2, 3, 4, 5]

VOXEL_SIZE = [0.2, 0.2, 0.2]
PC_RANGE = [-10.0, -10.0, -2.0, 10.0, 10.0, 2.0]
CAM_NAMES = ['cam_front', 'cam_rear', 'cam_left', 'cam_right']

# RGBA color map matching CLASS_COLORS_HEX from original
CLASS_COLORS_RGBA = np.array([
    [200, 200, 200, 255],  # 0 unknown - gray
    [220, 20,  60,  255],  # 1 person - red
    [119, 17,  32,  255],  # 2 table - dark red
    [0,   0,   142, 255],  # 3 chair - dark blue
    [0,   60,  100, 255],  # 4 floor - teal
    [0,   0,   230, 255],  # 5 car - bright blue
], dtype=np.float32)


def voxel2points(voxel, voxel_size, pc_range):
    """Extract occupied voxel centers.

    Args:
        voxel: (Dx, Dy, Dz) class labels
        voxel_size: [vx, vy, vz]
        pc_range: [xmin, ymin, zmin, xmax, ymax, zmax]

    Returns:
        points: (N, 3) world coordinates
        labels: (N,) class ids
        occIdx: tuple of (x_idx, y_idx, z_idx) arrays
    """
    occ_show = voxel > 0
    occIdx = np.where(occ_show)
    points = np.stack([
        occIdx[0] * voxel_size[0] + pc_range[0],
        occIdx[1] * voxel_size[1] + pc_range[1],
        occIdx[2] * voxel_size[2] + pc_range[2],
    ], axis=1)
    return points, voxel[occIdx], occIdx


def voxel_profile(points, voxel_size):
    """Build 3D bounding boxes for voxels.

    Returns:
        boxes: (N, 7) — (cx, cy, cz - dz/2, vx, vy, vz, 0)
    """
    centers = np.zeros_like(points)
    centers[:, :2] = points[:, :2]
    centers[:, 2] = points[:, 2] - voxel_size[2] / 2
    wlh = np.tile(np.array(voxel_size, dtype=np.float32), (points.shape[0], 1))
    yaw = np.zeros((points.shape[0], 1), dtype=np.float32)
    return np.concatenate([centers, wlh, yaw], axis=1)


def my_compute_box_3d(center, size):
    """Compute 8 corners of each 3D bounding box.

    Args:
        center: (N, 3) — (cx, cy, cz - dz/2)
        size: (N, 3) — (vx, vy, vz)

    Returns:
        corners_3d: (N, 8, 3)
    """
    h, w, l = size[:, 2], size[:, 0], size[:, 1]
    center = center.copy()
    center[:, 2] = center[:, 2] + h / 2
    l, w, h = l / 2, w / 2, h / 2
    x_corners = np.stack([-l, l, l, -l, -l, l, l, -l], axis=1)[..., None]
    y_corners = np.stack([w, w, -w, -w, w, w, -w, -w], axis=1)[..., None]
    z_corners = np.stack([h, h, h, h, -h, -h, -h, -h], axis=1)[..., None]
    corners = np.concatenate([x_corners, y_corners, z_corners], axis=2)
    corners[..., 0] += center[:, 0:1]
    corners[..., 1] += center[:, 1:2]
    corners[..., 2] += center[:, 2:3]
    return corners


def render_occ_open3d(occ_grid, offset=(0, 0, 0)):
    """Render occupancy grid as Open3D voxel geometry.

    Args:
        occ_grid: (Dx, Dy, Dz) class labels
        offset: (x, y, z) offset for positioning side-by-side

    Returns:
        list of Open3D geometries
    """
    import open3d as o3d

    colors = CLASS_COLORS_RGBA / 255.0
    points, labels, occIdx = voxel2points(occ_grid, VOXEL_SIZE, PC_RANGE)
    if len(points) == 0:
        return []

    _labels = labels % len(colors)
    pcd_colors = colors[_labels]  # (N, 4)

    # Apply offset
    points = points + np.array(offset, dtype=np.float32)

    # Create point cloud for voxel grid
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(pcd_colors[:, :3])

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.2)

    # Wireframe boxes
    bboxes = voxel_profile(points, VOXEL_SIZE)
    bboxes_corners = my_compute_box_3d(bboxes[:, :3], bboxes[:, 3:6])

    bases_ = np.arange(0, bboxes_corners.shape[0] * 8, 8)
    edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ])
    edges = np.tile(edges[None], (bboxes_corners.shape[0], 1, 1))
    edges = edges + bases_[:, None, None]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(bboxes_corners.reshape(-1, 3))
    line_set.lines = o3d.utility.Vector2iVector(edges.reshape(-1, 2))
    line_set.paint_uniform_color((0, 0, 0))

    return [voxel_grid, line_set]


def render_occ_voxel_image(pred, gt, stride=4):
    """Render side-by-side pred/GT occupancy using Open3D headless.

    Falls back to matplotlib if open3d is not available.

    Returns:
        img: (height, width, 3) uint8 BGR numpy array
    """
    try:
        import open3d as o3d

        # Downsample for performance
        Dx, Dy, Dz = pred.shape
        pred_ds = pred[::stride, ::stride, :].copy()
        gt_ds = gt[::stride, ::stride, :].copy()

        # Calculate grid extent in meters
        x_extent = Dx * VOXEL_SIZE[0]  # e.g. 20m

        # Offset GT to the right of prediction
        offset_x = x_extent * 1.25

        geoms_pred = render_occ_open3d(pred_ds, offset=(0, 0, 0))
        geoms_gt = render_occ_open3d(gt_ds, offset=(offset_x, 0, 0))

        # Create headless visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=1600, height=900)

        for g in geoms_pred + geoms_gt:
            vis.add_geometry(g)

        # Coordinate frame at origin
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])
        vis.add_geometry(frame)

        # Position camera to see both pred and GT
        vc = vis.get_view_control()
        lookat = np.array([offset_x / 2, 0, 0])
        front = np.array([0.3, -1.0, 0.5])
        up = np.array([0, 0, 1])
        vc.set_lookat(lookat)
        vc.set_front(front / np.linalg.norm(front))
        vc.set_up(up)
        vc.set_zoom(0.35)

        opt = vis.get_render_option()
        opt.background_color = np.asarray([1, 1, 1])
        opt.line_width = 3
        opt.point_size = 1

        vis.poll_events()
        vis.update_renderer()

        img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        img = (img * 255).astype(np.uint8)
        img = img[..., [2, 1, 0]]  # RGB -> BGR

        vis.destroy_window()
        return img

    except ImportError:
        return _render_occ_matplotlib(pred, gt, stride=stride)


def _render_occ_matplotlib(pred, gt, stride=4):
    """Fallback: matplotlib 3D scatter rendering."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    CLASS_COLORS_HEX = {
        0: '#c8c8c8', 1: '#dc143c', 2: '#771120',
        3: '#00008e', 4: '#003c64', 5: '#0000e6',
    }

    def _get_points(occ):
        xs, ys, zs, cs = [], [], [], []
        for x in range(0, occ.shape[0], stride):
            for y in range(0, occ.shape[1], stride):
                for z in range(occ.shape[2]):
                    cls_id = occ[x, y, z]
                    if cls_id == 0:
                        continue
                    xs.append(x * VOXEL_SIZE[0] + PC_RANGE[0])
                    ys.append(y * VOXEL_SIZE[1] + PC_RANGE[1])
                    zs.append(z * VOXEL_SIZE[2] + PC_RANGE[2])
                    cs.append(CLASS_COLORS_HEX[cls_id])
        return np.array(xs), np.array(ys), np.array(zs), cs

    def _draw(occ, title, ax):
        xs, ys, zs, cs = _get_points(occ)
        if len(xs) == 0:
            ax.text(0.5, 0.5, 0.5, 'No occupied voxels',
                    ha='center', va='center', transform=ax.transAxes)
        else:
            ax.scatter(xs, ys, zs, c=cs, marker='s', s=3,
                       alpha=0.85, edgecolors='none', depthshade=True)
        ax.set_xlim(PC_RANGE[0], PC_RANGE[3])
        ax.set_ylim(PC_RANGE[1], PC_RANGE[4])
        ax.set_zlim(PC_RANGE[2], PC_RANGE[5])
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=5)
        ax.view_init(elev=28, azim=-55)
        Dx, Dy, Dz = occ.shape
        ax.set_box_aspect((Dx, Dy, Dz * 2))

    dpi = 120
    fig_w = 1600 / dpi
    fig_h = 700 / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor='white')
    ax1 = fig.add_subplot(121, projection='3d', computed_zorder=False)
    _draw(pred, 'Prediction', ax1)
    ax2 = fig.add_subplot(122, projection='3d', computed_zorder=False)
    _draw(gt, 'Ground Truth', ax2)

    legend_elements = [Patch(facecolor=CLASS_COLORS_HEX[i], label=CLASS_NAMES[i])
                       for i in OBSERVED_CLASSES]
    fig.legend(handles=legend_elements, loc='lower center', ncol=len(OBSERVED_CLASSES),
               fontsize=9, frameon=False)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img[..., [2, 1, 0]]


def load_camera_images(info):
    """Load and resize fisheye camera images."""
    imgs = {}
    for cam_name in CAM_NAMES:
        img_path = info['cams'][cam_name]['data_path']
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((512, 640, 3), dtype=np.uint8)
        imgs[cam_name] = img
    return imgs


def build_composite(cam_imgs, occ_img, info):
    """Build the full composite image matching the reference layout.

    Layout:
      Row 1:  [cam_front        ] [cam_rear         ]
      Row 2:  [    3D Occupancy: Prediction | GT     ]
      Row 3:  [cam_left         ] [cam_right        ]

    Args:
        cam_imgs: dict of camera_name -> BGR image
        occ_img: (H, W, 3) BGR occupancy rendering
        info: dataset info dict

    Returns:
        composite: BGR image
    """
    cam_h, cam_w = 320, 400
    imgs_resized = {}
    for name in CAM_NAMES:
        imgs_resized[name] = cv2.resize(cam_imgs[name], (cam_w, cam_h))

    gap = 15
    border = 20

    total_w = cam_w * 2 + gap + border * 2

    # Resize occupancy to fit within the canvas width
    occ_margin = border * 2
    max_occ_w = total_w - occ_margin
    occ_w = max_occ_w
    occ_h = int(occ_img.shape[0] * occ_w / occ_img.shape[1])
    occ_resized = cv2.resize(occ_img, (occ_w, occ_h))

    total_h = cam_h * 2 + occ_h + gap * 2 + border * 2 + 40

    composite = np.ones((total_h, total_w, 3), dtype=np.uint8) * 240

    # Row 1: front (left), rear (right)
    y1 = border + 30
    x_left = border
    x_right = border + cam_w + gap
    composite[y1:y1 + cam_h, x_left:x_left + cam_w] = imgs_resized['cam_front']
    composite[y1:y1 + cam_h, x_right:x_right + cam_w] = imgs_resized['cam_rear']

    # Row 2: occupancy scene
    y2 = y1 + cam_h + gap
    occ_x = (total_w - occ_w) // 2
    composite[y2:y2 + occ_h, occ_x:occ_x + occ_w] = occ_resized

    # Row 3: left, right
    y3 = y2 + occ_h + gap
    composite[y3:y3 + cam_h, x_left:x_left + cam_w] = imgs_resized['cam_left']
    composite[y3:y3 + cam_h, x_right:x_right + cam_w] = imgs_resized['cam_right']

    # Camera labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    for name, x, y in [
        ('cam_front', x_left, y1 - 8),
        ('cam_rear', x_right, y1 - 8),
        ('cam_left', x_left, y3 - 8),
        ('cam_right', x_right, y3 - 8),
    ]:
        cv2.putText(composite, name, (x, y), font, 0.45, (80, 80, 80), 1)

    # Occ labels: "Prediction" on left half, "Ground Truth" on right half
    occ_label_y = y2 - 8
    mid_x = total_w // 2
    cv2.putText(composite, 'Prediction', (occ_x, occ_label_y),
                font, 0.5, (80, 80, 80), 1)
    cv2.putText(composite, 'Ground Truth', (occ_x + occ_w // 2, occ_label_y),
                font, 0.5, (80, 80, 80), 1)

    # Title
    sample_token = info['token']
    title = f'Sample: {sample_token}'
    cv2.putText(composite, title, (border, border - 5),
                font, 0.55, (0, 0, 0), 1)

    return composite


# ---- Main ----
def main():
    parser = argparse.ArgumentParser(
        description='Fisheye occupancy viz: camera images + 3D occ (pred vs GT)')
    parser.add_argument('--config', required=True, help='config file path')
    parser.add_argument('--weights', required=True, help='checkpoint file')
    parser.add_argument('--viz-dir', default='vis_fisheye', help='output directory')
    parser.add_argument('--max-samples', type=int, default=20, help='max samples')
    parser.add_argument('--stride', type=int, default=4,
                        help='voxel downsample stride (smaller = denser, slower)')
    parser.add_argument('--device', default='cuda:0', help='device')
    args = parser.parse_args()

    mmcv.mkdir_or_exist(args.viz_dir)

    cfg = Config.fromfile(args.config)
    cfg.model.train_cfg = None

    # Load plugin
    if getattr(cfg, 'plugin', False):
        import importlib
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

    dataset = build_dataset(cfg.data.val)
    print(f'Dataset: {len(dataset)} samples')

    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    checkpoint = torch.load(args.weights, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.to(args.device)
    model.eval()

    for idx in tqdm(range(min(len(dataset), args.max_samples)), desc='Visualizing'):
        data = dataset[idx]
        info = dataset.data_infos[idx]

        # ---- Inference ----
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

        batch_img_inputs = tuple(
            t.unsqueeze(0).to(args.device) if isinstance(t, torch.Tensor) else t
            for t in img_inputs
        )

        with torch.no_grad():
            result = model.simple_test(
                points=None, img_metas=[meta], img=batch_img_inputs)

        pred = result[0]
        if isinstance(pred, dict):
            pred = pred['pred_occ']
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()

        # ---- Load GT ----
        gt_data = np.load(info['occ_path'], allow_pickle=True)
        gt = gt_data['semantics']

        # ---- Load camera images ----
        cam_imgs = load_camera_images(info)

        # ---- Render 3D occupancy ----
        occ_img = render_occ_voxel_image(pred, gt, stride=args.stride)

        # ---- Composite ----
        composite = build_composite(cam_imgs, occ_img, info)

        save_path = os.path.join(args.viz_dir, f'{info["token"]}.png')
        cv2.imwrite(save_path, composite)

    print(f'\nDone. Results saved to {args.viz_dir}/')
    print('Layout: [front|rear] / [3D occ: pred | GT] / [left|right]')


if __name__ == '__main__':
    main()
