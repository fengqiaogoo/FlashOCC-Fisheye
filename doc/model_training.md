# Train model
```shell
# train (single GPU)
python tools/train.py projects/configs/fisheye/fisheye_flashocc_r50.py

# train (multi GPU, e.g. 4 GPUs)
bash tools/dist_train.sh projects/configs/fisheye/fisheye_flashocc_r50.py 4
```

# Test model
```bash
# test (single GPU)
python tools/test.py projects/configs/fisheye/fisheye_flashocc_r50.py work_dirs/fisheye_flashocc_r50/latest.pth --eval map

# test with result saving (for visualization)
python tools/test.py projects/configs/fisheye/fisheye_flashocc_r50.py work_dirs/fisheye_flashocc_r50/latest.pth --eval map --eval-options show_dir=work_dirs/fisheye_flashocc_r50/results

# multi-GPU test
bash tools/dist_test.sh projects/configs/fisheye/fisheye_flashocc_r50.py work_dirs/fisheye_flashocc_r50/latest.pth 4 --eval map
```

#### Test model
```shell

# multiple gpu
./tools/dist_test.sh $config $checkpoint num_gpu --eval mAP
# ray-iou metric
./tools/dist_test.sh $config $checkpoint num_gpu --eval ray-iou
```

#### FPS for Panoptic-FlashOcc
```shell
# for single-frame
python tools/analysis_tools/benchmark.py  config ckpt 
python tools/analysis_tools/benchmark.py  config ckpt --w_pano

# for multi-frame
python tools/analysis_tools/benchmark_sequential.py  config ckpt 
python tools/analysis_tools/benchmark_sequential.py  config ckpt --w_pano
```
