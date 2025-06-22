# UCDNet: Multi-UAV Collaborative 3-D Object Detection Network by Reliable Feature Mapping
Welcome to the official PyTorch implementation of "UCDNet: Multi-UAV Collaborative 3-D Object Detection Network by Reliable Feature Mapping." We have open-sourced this repository to foster research and collaboration in the field of multi-UAV perception and related areas.


## Setup Instructions
### Step1: Create basic environment.  
CUDA 11.3  
Python 3.7.16  
Torch 1.10.0   
### Step2: Prepare UCDNet by:
```shell script
git clone https://github.com/Smarome/UCDNet.git
cd UCDNet
pip install mmcv==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
pip install -r requirements.txt
pip install -v -e .
cd nuscenes-devkit-1.1.3/setup
pip install -v -e .
```

## Dataset
### Step1: Download dataset.
Download from https://pan.baidu.com/s/1I9tkdZM-6kMU0Hp3JNuCqg?pwd=p3tp
### Step2: Process dataset.
Create the pkl from AeroCollab3D dataset for UCDNet by running:
```shell script
python tools/create_data_UCDNet.py
```
Or download ready-made pkl from https://pan.baidu.com/s/14o5OBOBbC1yhwfMQ1V5IPw?pwd=hjvw and put it into AeroCollab3D dataset folder.

## Model
### Train model.
```shell script
python tools/train.py $config
```
The config of UCDNet is located at "configs/bevdet/UCDNet-r50-cbgs.py".

### Test model.
```shell script
python tools/test.py $config $checkpoint --eval mAP
```

## Bibtex

If this work is helpful for your research, please consider citing the following BibTeX entries.

```
@article{tian2024ucdnet,
  title={Ucdnet: Multi-uav collaborative 3d object detection network by reliable feature mapping},
  author={Tian, Pengju and Wang, Zhirui and Cheng, Peirui and Wang, Yuchao and Wang, Zhechao and Zhao, Liangjin and Yan, Menglong and Yang, Xue and Sun, Xian},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024},
  publisher={IEEE}
}
```

