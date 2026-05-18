## Environment Setup

### Step 0. Prerequisites - CUDA Environment

> **IMPORTANT**: Your system CUDA version (e.g. `/usr/local/cuda-xx.x`) does NOT need to match the PyTorch CUDA toolkit version. PyTorch ships with its own CUDA runtime. The system CUDA driver just needs to be recent enough to support your GPU.

Check your GPU architecture and choose the correct setup path below:

- **Ada Lovelace GPUs** (RTX 4000 Ada, RTX 4090, RTX 4080, RTX 4070, etc.) → **Use Setup B** (CUDA 11.7 required, CUDA 11.1 is NOT supported)
- **Older GPUs** (Ampere A100/RTX 3090, Turing T4/RTX 2080, Volta V100, etc.) → **Use Setup A** (CUDA 11.1 works fine)

You can check your GPU with:
```bash
nvidia-smi
```
Look at the GPU name. If it contains "Ada" or "RTX 40", use Setup B.

---

### Setup A - Standard GPUs (Ampere / Turing / Volta)

step 1. Create conda environment and install dependencies:
```bash
conda create --name FlashOcc python=3.8.5
conda activate FlashOcc
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.5.3
pip install mmdet==2.25.1
pip install mmsegmentation==0.25.0

sudo apt-get install python3-dev 
sudo apt-get install libevent-dev
sudo apt-get groupinstall 'development tools'

# Set CUDA environment variables (match your system CUDA installation path)
export PATH=/usr/local/cuda-11.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH
export CUDA_ROOT=/usr/local/cuda-11.1
pip install pycuda

pip install lyft_dataset_sdk
pip install networkx==2.2
pip install numba==0.53.0
pip install numpy==1.23.5
pip install nuscenes-devkit
pip install plyfile
pip install scikit-image
pip install tensorboard
pip install trimesh==2.35.39
pip install setuptools==59.5.0
pip install yapf==0.40.1
```

step 2. Clone repos and install:
```bash
cd Path_to_FlashOcc
git clone git@github.com:Yzichen/FlashOCC.git

cd Path_to_FlashOcc/FlashOcc
git clone https://github.com/open-mmlab/mmdetection3d.git

cd Path_to_FlashOcc/FlashOcc/mmdetection3d
git checkout v1.0.0rc4
pip install -v -e . 

cd Path_to_FlashOcc/FlashOcc/projects
pip install -v -e . 
```

---

### Setup B - Ada Lovelace GPUs (RTX 40 series / Ada)

> Ada architecture requires CUDA >= 11.8 runtime. PyTorch 1.13.1+cu117 is the minimum version that works.

step 1. Create conda environment:
```bash
conda create --name FlashOcc python=3.8.5
conda activate FlashOcc
```

step 2. Install PyTorch with CUDA 11.7:
```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```
> If download is slow (e.g. in China), you can use the Tsinghua mirror:
> ```bash
> pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple --index-url https://download.pytorch.org/whl/cu117
> ```

step 3. Install MMDetection family:
```bash
pip install mmdet==2.25.1
pip install mmsegmentation==0.25.0
```

step 4. Install mmcv-full **from source** (pre-built wheels for cu117 may not match your mmdet version):
```bash
sudo apt-get install python3-dev libevent-dev
# If you have devtoolset on CentOS/RHEL:
sudo yum groupinstall 'development tools'

# Set CUDA environment variables (adjust path to match your system CUDA)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_ROOT=/usr/local/cuda

pip install pycuda

# Build mmcv-full 1.6.0 from source (compatible with torch 1.13 + mmdet 2.25.1)
# This takes ~10-15 minutes
pip install mmcv-full==1.6.0
```

step 5. Install other dependencies:
```bash
pip install lyft_dataset_sdk
pip install networkx==2.2
pip install numba==0.53.0
pip install numpy==1.23.5
pip install nuscenes-devkit
pip install plyfile
pip install scikit-image
pip install tensorboard
pip install trimesh==2.35.39
pip install setuptools==59.5.0
pip install yapf==0.40.1
```

step 6. Clone repos and install:
```bash
cd Path_to_FlashOcc
git clone git@github.com:Yzichen/FlashOCC.git

cd Path_to_FlashOcc/FlashOcc
git clone https://github.com/open-mmlab/mmdetection3d.git

cd Path_to_FlashOcc/FlashOcc/mmdetection3d
git checkout v1.0.0rc4
pip install -v -e . 

cd Path_to_FlashOcc/FlashOcc/projects
pip install -v -e . 
```

step 7. Rebuild custom CUDA extensions for the new PyTorch version:

> **Critical for Ada GPUs**: The pre-compiled `.so` files (if any) from the original CUDA 11.1 build are incompatible with PyTorch 1.13. You must clean and rebuild.

```bash
cd Path_to_FlashOcc/FlashOcc

# Remove any pre-compiled .so files from old builds
find projects/mmdet3d_plugin -name "*.so" -delete
rm -rf ~/.cache/torch_extensions/py38_cu117/dvr/

# Rebuild all custom CUDA extensions (bev_pool, bev_pool_v2, nearest_assign)
cd projects
python setup.py build_ext --inplace
```

---

### Verification

Verify your installation works:
```bash
conda activate FlashOcc
cd Path_to_FlashOcc/FlashOcc
python -c "import torch; print('PyTorch:', torch.__version__); m = torch.eye(4, device='cuda', dtype=torch.float64); torch.linalg.inv(m); print('CUDA OK')"
```
Expected output: `CUDA OK` (no errors).

---

step 3. Prepare nuScenes dataset as introduced in [nuscenes_det.md](nuscenes_det.md) and create the pkl for FlashOCC by running:
```shell
python tools/create_data_bevdet.py
```
thus, the folder will be ranged as following:
```shell script
└── Path_to_FlashOcc/
    └── data
        └── nuscenes
            ├── v1.0-trainval (existing)
            ├── sweeps  (existing)
            ├── samples (existing)
            ├── bevdetv2-nuscenes_infos_train.pkl (new)
            └── bevdetv2-nuscenes_infos_val.pkl (new)
```

step 4. For Occupancy Prediction task, download (only) the 'gts' from [CVPR2023-3D-Occupancy-Prediction](https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction) and arrange the folder as:
```shell script
└── Path_to_FlashOcc/
    └── data
        └── nuscenes
            ├── v1.0-trainval (existing)
            ├── sweeps  (existing)
            ├── samples (existing)
            ├── gts (new)
            ├── bevdetv2-nuscenes_infos_train.pkl (new)
            └── bevdetv2-nuscenes_infos_val.pkl (new)
```
(for panoptic occupancy), we follow the data setting in SparseOcc:

(1) Download Occ3D-nuScenes occupancy GT from [gdrive](https://drive.google.com/file/d/1kiXVNSEi3UrNERPMz_CfiJXKkgts_5dY/view?usp=drive_link), unzip it, and save it to `data/nuscenes/occ3d`.

(2) Generate the panoptic occupancy ground truth with `gen_instance_info.py`. The panoptic version of Occ3D will be saved to `data/nuscenes/occ3d_panoptic`.


step 5. CKPTS Preparation
(1) Download flashocc-r50-256x704.pth[https://drive.google.com/file/d/1k9BzXB2nRyvXhqf7GQx3XNSej6Oq6I-B/view] to Path_to_FlashOcc/FlashOcc/ckpts/, then run:
```shell script
# For 4 GPUs:
bash tools/dist_test.sh projects/configs/flashocc/flashocc-r50.py  ckpts/flashocc-r50-256x704.pth 4 --eval map
# For single GPU:
bash tools/dist_test.sh projects/configs/flashocc/flashocc-r50.py  ckpts/flashocc-r50-256x704.pth 1 --eval map
```
Expected output: mIoU ~32.0 (for the 256x704 R50 model).

step 6. (Optional) Install mmdeploy for tensorrt testing
```shell script
conda activate FlashOcc
pip install Cython==0.29.24

### get tensorrt
wget https://developer.download.nvidia.com/compute/machine-learning/tensorrt/secure/8.4.0/tars/TensorRT-8.4.0.6.Linux.x86_64-gnu.cuda-11.6.cudnn8.3.tar.gz
export TENSORRT_DIR=Path_to_TensorRT-8.4.0.6

### get onnxruntime
ONNXRUNTIME_VERSION=1.8.1
pip install onnxruntime-gpu==${ONNXRUNTIME_VERSION}
cd Path_to_your_onnxruntime
wget https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz \
     && tar -zxvf onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz
# export ONNXRUNTIME_DIR=/data01/shuchangyong/pkgs/onnxruntime-linux-x64-1.8.1
export ONNXRUNTIME_DIR=Path_to_your_onnxruntime/onnxruntime-linux-x64-1.8.1
cd Path_to_FlashOcc/FlashOcc/
git clone git@github.com:drilistbox/mmdeploy.git
cd Path_to_FlashOcc/FlashOcc/mmdeploy
git submodule update --init --recursive
mkdir -p build
cd Path_to_FlashOcc/FlashOcc/mmdeploy/build
cmake -DMMDEPLOY_TARGET_BACKENDS="ort;trt" ..
make -j 16
cd Path_to_FlashOcc/FlashOcc/mmdeploy
pip install -e .

### build sdk
cd Path_to_pplcv/
git clone https://github.com/openppl-public/ppl.cv.git
cd Path_to_pplcv/ppl.cv
export PPLCV_VERSION=0.7.0
git checkout tags/v${PPLCV_VERSION} -b v${PPLCV_VERSION}
./build.sh cuda

#pip install nvidia-tensorrt==8.4.0.6
pip install nvidia-tensorrt==8.4.1.5
pip install tensorrt
#pip install h5py
pip install spconv==2.3.6

export PATH=Path_to_TensorRT-8.4.0.6/bin:$PATH
export LD_LIBRARY_PATH=Path_to_TensorRT-8.4.0.6/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=Path_to_TensorRT-8.4.0.6/lib:$LIBRARY_PATH
```

## The finally overall rangement
1. Tensort
```shell script
└── Path_to_TensorRT-8.4.0.6
    └── TensorRT-8.4.0.6
```
2. FlashOcc
```shell script
└── Path_to_FlashOcc/
    └── data
        └── nuscenes
            ├── v1.0-trainval (existing)
            ├── sweeps  (existing)
            ├── samples (existing)
            ├── gts (new)
            ├── bevdetv2-nuscenes_infos_train.pkl (new)
            └── bevdetv2-nuscenes_infos_val.pkl (new)
    └── doc
        ├── install.md
        └── trt_test.md
    ├── figs
    ├── mmdeploy (new)
    ├── mmdetection3d (new)
    ├── projects
    ├── requirements
    ├── tools
    └── README.md
```
3. ppl.cv
```shell script
└── Path_to_pplcv
    └── ppl.cv
```
