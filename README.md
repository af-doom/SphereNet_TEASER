
# SphereNet: Learning a Noise-Robust and General Descriptor for Point Cloud Registration

🔥🔥🔥 Our paper is accepted by IEEE Transactions on Geoscience and Remote Sensing (TGRS) 🔥🔥🔥
[SphereNet: Learning a Noise-Robust and General Descriptor for Point Cloud Registration](https://ieeexplore.ieee.org/document/10356130).

### (1) Overview

<p align="center"> <img src="figs/zongtu.png" width="100%"> </p>

[//]: # (<p align="center"> <img src="figs/FMR.png" width="50%"> </p>)



### (2) Setup
This code has been tested with 
1. Python 3.9, Pytorch 1.11.0, CUDA 10.2 on Arch Linux.
2. Python 3.9, Pytorch 1.11.0, CUDA 11.1 on Ubuntu 20.04.

- Clone the repository 
```
git clone https://github.com/GuiyuZhao/SphereNet && cd SphereNet
```
- Setup conda virtual environment
```
conda create -n spherenet python=3.9
source activate spinnet
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch
conda install -c open3d-admin open3d==0.11.1
pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```
- Prepare the datasets


### (1) Demo: Global point cloud registration using SphereNet and TEASER++
If necessary, you will need to change the radius parameter to fit your data.
```
cd ./ThreeDMatch/Test
python demo.py  
```

## Acknowledgement

In this project, we use parts of the implementations of the following works:

* [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch)
* [PPF-FoldNet](https://github.com/XuyangBai/PPF-FoldNet)
* [D3Feat.pytorch](https://github.com/XuyangBai/D3Feat.pytorch)
* [SpinNet](https://github.com/QingyongHu/SpinNet)
* [Geotransformer](https://github.com/qinzheng93/GeoTransformer)
* [Predator](https://github.com/prs-eth/OverlapPredator)



## Updates
* 03/05/2023: The code is released!
* 12/13/2023: Our paper is accepted by IEEE Transactions on Geoscience and Remote Sensing!


## Citation
```bibtex
@ARTICLE{10356130,
  author={Zhao, Guiyu and Guo, Zhentao and Wang, Xin and Ma, Hongbin},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={SphereNet: Learning a Noise-Robust and General Descriptor for Point Cloud Registration}, 
  year={2024},
  volume={62},
  number={},
  pages={1-16},
  doi={10.1109/TGRS.2023.3342423}
}

```
