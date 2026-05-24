# Fisheye occupancy visualization — adapted from vis_occ.py for 4 fisheye cameras

import os

import mmcv
import open3d as o3d
import numpy as np
import torch
import pickle
import argparse
import cv2

NOT_OBSERVED = -1
FREE = 0
OCCUPIED = 1
FREE_LABEL = 0
BINARY_OBSERVED = 1
BINARY_NOT_OBSERVED = 0

VOXEL_SIZE = [0.2, 0.2, 0.2]
POINT_CLOUD_RANGE = [-10, -10, -2, 10, 10, 2]
SPTIAL_SHAPE = [100, 100, 20]

# Fisheye 6-class colormap: unknown, person, table, chair, floor, car
colormap_to_colors = np.array(
    [
        [0,   0,   0, 255],    # 0 unknown/free - black
        [220, 20, 60, 255],    # 1 person - crimson
        [255, 158, 0, 255],    # 2 table - orange
        [0,   0, 230, 255],    # 3 chair - blue
        [47,  79, 79, 255],    # 4 floor - darkslategray
        [255, 99, 71, 255],    # 5 car - tomato
    ], dtype=np.float32)


def voxel2points(voxel, occ_show, voxelSize):
    occIdx = torch.where(occ_show)
    points = torch.cat((occIdx[0][:, None] * voxelSize[0] + POINT_CLOUD_RANGE[0],
                        occIdx[1][:, None] * voxelSize[1] + POINT_CLOUD_RANGE[1],
                        occIdx[2][:, None] * voxelSize[2] + POINT_CLOUD_RANGE[2]),
                       dim=1)
    return points, voxel[occIdx], occIdx


def voxel_profile(voxel, voxel_size):
    centers = torch.cat((voxel[:, :2], voxel[:, 2][:, None] - voxel_size[2] / 2), dim=1)
    wlh = torch.cat((torch.tensor(voxel_size[0]).repeat(centers.shape[0])[:, None],
                     torch.tensor(voxel_size[1]).repeat(centers.shape[0])[:, None],
                     torch.tensor(voxel_size[2]).repeat(centers.shape[0])[:, None]), dim=1)
    yaw = torch.full_like(centers[:, 0:1], 0)
    return torch.cat((centers, wlh, yaw), dim=1)


def my_compute_box_3d(center, size, heading_angle):
    h, w, l = size[:, 2], size[:, 0], size[:, 1]
    center[:, 2] = center[:, 2] + h / 2
    l, w, h = (l / 2).unsqueeze(1), (w / 2).unsqueeze(1), (h / 2).unsqueeze(1)
    x_corners = torch.cat([-l, l, l, -l, -l, l, l, -l], dim=1)[..., None]
    y_corners = torch.cat([w, w, -w, -w, w, w, -w, -w], dim=1)[..., None]
    z_corners = torch.cat([h, h, h, h, -h, -h, -h, -h], dim=1)[..., None]
    corners_3d = torch.cat([x_corners, y_corners, z_corners], dim=2)
    corners_3d[..., 0] += center[:, 0:1]
    corners_3d[..., 1] += center[:, 1:2]
    corners_3d[..., 2] += center[:, 2:3]
    return corners_3d


def show_point_cloud(points: np.ndarray, colors=True, points_colors=None, bbox3d=None,
                     voxelize=False, bbox_corners=None, linesets=None, vis=None,
                     offset=[0, 0, 0], large_voxel=True, voxel_size=0.4):
    if vis is None:
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
    if isinstance(offset, list) or isinstance(offset, tuple):
        offset = np.array(offset)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points + offset)
    if colors:
        pcd.colors = o3d.utility.Vector3dVector(points_colors[:, :3])
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1, origin=[0, 0, 0])

    voxelGrid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    if large_voxel:
        vis.add_geometry(voxelGrid)
    else:
        vis.add_geometry(pcd)

    if voxelize:
        line_sets = o3d.geometry.LineSet()
        line_sets.points = o3d.open3d.utility.Vector3dVector(bbox_corners.reshape((-1, 3)) + offset)
        line_sets.lines = o3d.open3d.utility.Vector2iVector(linesets.reshape((-1, 2)))
        line_sets.paint_uniform_color((0, 0, 0))
        vis.add_geometry(line_sets)

    vis.add_geometry(mesh_frame)
    return vis


def show_occ(occ_state, occ_show, voxel_size, vis=None, offset=[0, 0, 0]):
    colors = colormap_to_colors / 255
    pcd, labels, occIdx = voxel2points(occ_state, occ_show, voxel_size)
    _labels = labels % len(colors)
    pcds_colors = colors[_labels]

    bboxes = voxel_profile(pcd, voxel_size)
    bboxes_corners = my_compute_box_3d(bboxes[:, 0:3], bboxes[:, 3:6], bboxes[:, 6:7])

    bases_ = torch.arange(0, bboxes_corners.shape[0] * 8, 8)
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7],
                          [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]])
    edges = edges.reshape((1, 12, 2)).repeat(bboxes_corners.shape[0], 1, 1)
    edges = edges + bases_[:, None, None]

    vis = show_point_cloud(
        points=pcd.numpy(),
        colors=True,
        points_colors=pcds_colors,
        voxelize=True,
        bbox3d=bboxes.numpy(),
        bbox_corners=bboxes_corners.numpy(),
        linesets=edges.numpy(),
        vis=vis,
        offset=offset,
        large_voxel=True,
        voxel_size=voxel_size[0],
    )
    return vis


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize fisheye occupancy results')
    parser.add_argument('res', help='Path to the predicted result directory')
    parser.add_argument('--canva-size', type=int, default=400, help='Size of canva in pixel')
    parser.add_argument('--vis-frames', type=int, default=500,
                        help='Number of frames for visualization')
    parser.add_argument('--scale-factor', type=int, default=2,
                        help='Trade-off between image-view and bev in canvas size')
    parser.add_argument('--version', type=str, default='val',
                        help='Version of dataset (val/train)')
    parser.add_argument('--draw-gt', action='store_true')
    parser.add_argument('--config', type=str, required=True,
                        help='Config file path')
    parser.add_argument('--save_path', type=str, default='./vis_fisheye',
                        help='Path to save visualization results')
    parser.add_argument('--format', type=str, default='image',
                        choices=['video', 'image'],
                        help='Output format')
    parser.add_argument('--fps', type=int, default=10, help='Frame rate of video')
    parser.add_argument('--video-prefix', type=str, default='vis_fisheye', help='Name of video')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    from mmcv import Config
    cfg = Config.fromfile(args.config)

    data_root = cfg.data_root
    print('data_root read from config: %s' % data_root)

    # Read camera list from config
    views = cfg.data_config.cams
    print('cameras: %s' % views)

    # Read image size from config
    cam_h, cam_w = cfg.data_config.input_size  # (H, W) = (512, 640)
    print('cam image size: %dx%d' % (cam_w, cam_h))

    # Load dataset info
    info_path = cfg.data[args.version].ann_file
    print('ann_file read from config: %s' % info_path)
    dataset = pickle.load(open(info_path, 'rb'))

    results_dir = args.res
    vis_dir = args.save_path
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    print('saving visualized result to %s' % vis_dir)

    scale_factor = args.scale_factor
    canva_size = args.canva_size

    if args.format == 'video':
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        total_w = int(cam_w / scale_factor * 2)
        total_h = int(cam_h / scale_factor * 2 + canva_size)
        vout = cv2.VideoWriter(
            os.path.join(vis_dir, '%s.mp4' % args.video_prefix), fourcc,
            args.fps, (total_w, total_h))

    print('start visualizing results')

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    n_frames = min(args.vis_frames, len(dataset['infos']))
    for cnt, info in enumerate(dataset['infos'][:n_frames]):
        if cnt % 10 == 0:
            print('%d/%d' % (cnt, n_frames))

        sample_token = info['token']

        # Load predicted + GT from saved npz
        result_npz_path = os.path.join(results_dir, f'{sample_token}.npz')
        result_data = np.load(result_npz_path)
        pred_occ = result_data['pred']        # (Dx, Dy, Dz)

        # Load camera images
        imgs = []
        for view in views:
            img_path = info['cams'][view]['data_path']
            img = cv2.imread(img_path)
            imgs.append(img)

        # Render occupancy — show occupied (non-free) voxels
        voxel_show = pred_occ != FREE_LABEL
        voxel_size = VOXEL_SIZE
        vis = show_occ(torch.from_numpy(pred_occ), torch.from_numpy(voxel_show),
                       voxel_size=voxel_size, vis=vis,
                       offset=[0, pred_occ.shape[0] * voxel_size[0] * 1.2 * 0, 0])

        if args.draw_gt:
            gt_semantics = result_data['gt']      # (Dx, Dy, Dz)
            gt_voxel_show = gt_semantics != FREE_LABEL
            vis = show_occ(torch.from_numpy(gt_semantics),
                           torch.from_numpy(gt_voxel_show),
                           voxel_size=voxel_size, vis=vis,
                           offset=[0, gt_semantics.shape[0] * voxel_size[0] * 1.2 * 1, 0])

        # Set camera view
        view_control = vis.get_view_control()
        look_at = np.array([0.0, 0.0, 1.0])
        front = np.array([0.0, 0.5, 0.4])
        up = np.array([0.0, 0.3, 0.9])
        zoom = np.array([0.15])
        view_control.set_lookat(look_at)
        view_control.set_front(front)
        view_control.set_up(up)
        view_control.set_zoom(zoom)

        opt = vis.get_render_option()
        opt.background_color = np.asarray([1, 1, 1])
        opt.line_width = 5

        vis.poll_events()
        vis.update_renderer()
        vis.run()

        occ_canvas = vis.capture_screen_float_buffer(do_render=True)
        occ_canvas = np.asarray(occ_canvas)
        occ_canvas = (occ_canvas * 255).astype(np.uint8)
        occ_canvas = occ_canvas[..., [2, 1, 0]]

        vis.clear_geometries()

        # Build composite: 2 rows of 2 cameras, occ canvas in the middle
        # Step 1: create big image with camera rows
        big_img = np.zeros((cam_h * 2 + canva_size * scale_factor, cam_w * 2, 3),
                           dtype=np.uint8)
        big_img[:cam_h, :cam_w, :] = imgs[0]          # cam_front  top-left
        big_img[:cam_h, cam_w:, :] = imgs[1]           # cam_rear   top-right
        big_img[cam_h + canva_size * scale_factor:, :cam_w, :] = imgs[2]     # cam_left   bottom-left
        big_img[cam_h + canva_size * scale_factor:, cam_w:, :] = imgs[3]     # cam_right  bottom-right

        # Step 2: resize big image
        final_w = int(cam_w / scale_factor * 2)
        final_h = int(cam_h / scale_factor * 2 + canva_size)
        big_img = cv2.resize(big_img, (final_w, final_h))

        # Step 3: resize occ canvas to fit in available width with margins
        top_row_h = int(cam_h / scale_factor)
        occ_area_h = canva_size
        margin = 10
        max_occ_w = final_w - margin * 2
        max_occ_h = occ_area_h - margin * 2
        occ_h, occ_w = occ_canvas.shape[:2]
        scale = min(max_occ_w / occ_w, max_occ_h / occ_h)
        occ_new_w = int(occ_w * scale)
        occ_new_h = int(occ_h * scale)
        occ_canvas_resize = cv2.resize(occ_canvas, (occ_new_w, occ_new_h),
                                       interpolation=cv2.INTER_CUBIC)

        # Step 4: center occ canvas in the middle area
        w_begin = (final_w - occ_new_w) // 2
        h_begin = top_row_h + (occ_area_h - occ_new_h) // 2
        big_img[h_begin:h_begin + occ_new_h,
                w_begin:w_begin + occ_new_w, :] = occ_canvas_resize

        if args.format == 'image':
            out_dir = os.path.join(vis_dir, f'{sample_token}')
            mmcv.mkdir_or_exist(out_dir)
            for i, img in enumerate(imgs):
                cv2.imwrite(os.path.join(out_dir, f'img_{views[i]}.png'), img)
            cv2.imwrite(os.path.join(out_dir, 'occ.png'), occ_canvas)
            cv2.imwrite(os.path.join(out_dir, 'overall.png'), big_img)
        elif args.format == 'video':
            cv2.putText(big_img, f'{cnt}', (5, 15),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 0),
                        fontScale=0.5)
            cv2.putText(big_img, f'{sample_token}', (5, 35),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 0),
                        fontScale=0.5)
            vout.write(big_img)

    if args.format == 'video':
        vout.release()
    vis.destroy_window()


if __name__ == '__main__':
    main()
