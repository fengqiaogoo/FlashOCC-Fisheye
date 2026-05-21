# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick commands

```bash
# Single-GPU training
python tools/train.py projects/configs/flashocc/flashocc-r50.py

# Multi-GPU training (4 GPUs)
bash tools/dist_train.sh <config> 4

# Single-GPU testing (mIoU metric)
python tools/test.py <config> <checkpoint> --eval map

# Multi-GPU testing
bash tools/dist_test.sh <config> <checkpoint> 4 --eval map

# Panoptic testing (ray-iou metric)
bash tools/dist_test.sh <config> <checkpoint> 4 --eval ray-iou

# Rebuild CUDA extensions after changing C++/CUDA code
cd projects && python setup.py build_ext --inplace

# Install the plugin in dev mode
cd projects && pip install -v -e .
```

## Project overview

FlashOCC is a camera-only 3D semantic occupancy prediction framework for autonomous driving, built as a plugin on top of **MMDetection3D v1.0.0rc4** (OpenMMLab). The core innovation is a "Channel-to-Height" (C2H) head that avoids expensive 3D convolutions — it processes BEV features with 2D convs, then expands channels into the height (z) dimension via MLP.

- **Paper**: Fast and Memory-Efficient Occupancy Prediction (arXiv 2311.12058), Panoptic-FlashOCC (arXiv 2406.10527)
- **Dataset**: nuScenes with occupancy ground truth from CVPR2023 3D-Occupancy-Prediction, or Occ3D-nuScenes
- **License**: Apache 2.0

## Architecture

This is a **plugin** architecture. The base framework (`mmdetection3d/`) is an unmodified copy of mmdetection3d. All custom code lives in `projects/` and is dynamically loaded at runtime via `plugin=True` in configs.

### Source tree

```
projects/
├── configs/                      # .py config files by variant
│   ├── bevdet_occ/               #   Baseline BEVDet-Occupancy
│   ├── flashocc/                 #   FlashOCC (R50, Swin-B, stereo, 4D)
│   └── panoptic-flashocc/        #   Panoptic-FlashOCC (depth, 4D, longterm, TRT)
├── mmdet3d_plugin/
│   ├── datasets/                 # NuScenesDatasetOccpancy, data pipelines
│   ├── models/
│   │   ├── detectors/            # BEVDet, BEVDetOCC, BEVDepthOCC, BEVDepthPano, etc.
│   │   ├── backbones/            # CustomResNet (2D/3D), SwinTransformer variants
│   │   ├── necks/                # CustomFPN, FPN_LSS, LSSViewTransformer (Lift-Splat-Shoot)
│   │   ├── dense_heads/          # BEVOCCHead2D (C2H), BEVOCCHead2D_V2 (focal/lovasz), centerness head
│   │   ├── losses/               # CE, focal, lovasz_softmax, semkitti_loss
│   │   └── model_utils/          # DepthNet (ASPP, DCN, camera-SE, stereo cost volume)
│   ├── ops/                      # CUDA extensions: bev_pool, bev_pool_v2, nearest_assign
│   └── core/                     # Evaluation metrics (mIoU, FScore, ray-iou, ray-pq), hooks
lib/dvr/                          # Depth-aware volume rendering for ray-cast metrics
tools/                            # train.py, test.py, create_data_bevdet.py, benchmark scripts
```

### Forward pass

```
6 camera images (B,N,3,H,W)
  -> ImageBackbone (ResNet50/Swin-B) + CustomFPN
  -> LSSViewTransformer: DepthNet predicts (D,C) per pixel -> BEVPoolv2 (CUDA) sums into BEV grid -> collapse_z -> (B, C*Dz, 200, 200)
  -> BEV Encoder: CustomResNet + FPN_LSS
  -> BEVOCCHead2D: 2D conv -> permute -> MLP expands channels to Dz*18 -> reshape -> (B, 200, 200, 16, 18)
```

### Key detector class hierarchy

```
CenterPoint (mmdet3d)
  -> BEVDet (base: single-frame, no depth)
    -> BEVDetOCC (adds OCC head, replaces detection head)
      -> BEVDetOCCTRT (TensorRT variant)
    -> BEVDepthOCC (adds DepthNet + depth supervision)
      -> BEVDepthPano (adds instance centerness head for panoptic)
    -> BEVStereo4DOCC (adds temporal stereo + multi-frame fusion)
    -> BEVDepth4DOCC (multi-frame + depth loss)
      -> BEVDepth4DPano (panoptic + 4D temporal)
```

### The flashocc head (BEVOCCHead2D)

Defined in `projects/mmdet3d_plugin/models/dense_heads/bev_occ_head.py`. This is the key differentiator:
- **BEVOCCHead3D** — legacy 3D conv head (baseline, slow)
- **BEVOCCHead2D** — 2D conv + MLP predictor, channel-to-height: (B, C, 200, 200) -> permute -> (B, 200, 200, C) -> MLP -> (B, 200, 200, 16*18) -> reshape -> occupancy grid
- **BEVOCCHead2D_V2** — same architecture with stronger loss: focal + sem_scal + geo_scal + lovasz_softmax (used in Panoptic-FlashOCC)

## Key constraints

- **mmdet3d package**: This is a local fork of mmdetection3d v1.0.0rc4. It must be installed (`pip install -e .` from `mmdetection3d/`) before the plugin.
- **CUDA extensions**: Three custom CUDA ops (`bev_pool`, `bev_pool_v2`, `nearest_assign`) are compiled at plugin install time. When changing C++/CUDA sources in `projects/mmdet3d_plugin/ops/`, rebuild with `python setup.py build_ext --inplace` from `projects/`.
- **PyTorch/CUDA compatibility**: For Ada/RTX 40 GPUs, PyTorch 1.13+cu117 is required (CUDA 11.7 min). For older GPUs, PyTorch 1.10+cu111 works.
- **Configs use `plugin=True`**: All project configs set `plugin = True`, which triggers dynamic import of `projects/mmdet3d_plugin/` at runtime. Without this flag, custom components won't be registered.
- **`_delete_=True` in configs**: MMDetection3D config inheritance uses this to override dicts. When overriding a model sub-component, you must set `_delete_=True` on the parent dict to fully replace it.
- **Large `.pth` files**: Checkpoint files in `ckpts/` are in `.gitignore`. Do not commit them.

## Data pipeline

1. Raw nuScenes data in `data/nuscenes/` (v1.0-trainval, sweeps, samples)
2. `tools/create_data_bevdet.py` generates `.pkl` info files
3. Occupancy ground truth from CVPR2023 3D-Occupancy-Prediction repo goes in `data/nuscenes/gts/`
4. For panoptic: Occ3D-nuScenes GT + `gen_instance_info.py` -> `data/nuscenes/occ3d_panoptic/`

## TensorRT / deployment

- `projects/configs/panoptic-flashocc/` contains TRT-ready config variants
- `tools/convert_bevdet_to_TRT.py` for model conversion
- `tools/export_onnx.py` for ONNX export
- `doc/mmdeploy_test.md` has deployment instructions
