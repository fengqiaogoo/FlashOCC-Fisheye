
# FlashOcc
```shell
# step 1. generate result 
# 单张GPU
python tools/test.py projects/configs/flashocc/flashocc-r50.py ckpts/flashocc-r50-256x704.pth --eval mAP --eval-options show_dir=work_dirs/flashocc_r50/results

bash tools/dist_test.sh projects/configs/flashocc/flashocc-r50.py ckpts/flashocc-r50-256x704.pth 1 --eval mAP --eval-options show_dir=work_dirs/flashocc_r50/results

# 多张GPU
bash tools/dist_test.sh projects/configs/flashocc/flashocc-r50.py ckpts/flashocc-r50-256x704.pth 4 --eval map --eval-options show_dir=work_dirs/flashocc_r50/results
# step 2. visualization
python tools/analysis_tools/vis_occ.py work_dirs/flashocc_r50/results/ --config projects/configs/flashocc/flashocc-r50.py --save_path ./vis
```

# Fisheye-FlashOcc
```shell
# step 1. generate result (save npz to results dir)
# single GPU
python tools/test.py projects/configs/fisheye/fisheye_flashocc_r50.py work_dirs/fisheye_flashocc_r50/latest.pth --eval map --eval-options show_dir=work_dirs/fisheye_flashocc_r50/results

# multi-GPU
bash tools/dist_test.sh projects/configs/fisheye/fisheye_flashocc_r50.py work_dirs/fisheye_flashocc_r50/latest.pth 4 --eval map --eval-options show_dir=work_dirs/fisheye_flashocc_r50/results

# step 2. visualization
python tools/analysis_tools/vis_fisheye_occ.py work_dirs/fisheye_flashocc_r50/results/ --config projects/configs/fisheye/fisheye_flashocc_r50.py --save_path ./vis_fisheye --draw-gt
```

# Panoptic-FlashOcc
```shell

exp_name=panoptic-flashocc-r50-depth4d-longterm8f-pano
python tools/vis_occ.py --config projects/configs/panoptic-flashocc/${exp_name}.py --weights work_dirs/${exp_name}/epoch_24_ema.pth --viz-dir vis/${exp_name} --draw-pano-gt

```

