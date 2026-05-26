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

