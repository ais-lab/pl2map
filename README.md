# Point-Line to Map Regresssion for Camera Relocalization
#### [Project Page](https://thpjp.github.io/pl2map/) | [PL2Map](https://arxiv.org/abs/2402.18011) | [PL2Map++](https://arxiv.org/pdf/2502.20814)(Code for PL2Map++ is coming soon)
## Introduction

![demo_vid](assets/demo.gif)

We introduce a lightweight neural network for visual localization that efficiently represents both 3D points and lines. Specifically, we use a single transformer block to convert line features into distinctive point-like descriptors. These features are then refined through self- and cross-attention in a graph-based framework before 3D map regression using simple MLPs. Our method outperforms [Hloc](https://github.com/cvg/Hierarchical-Localization) and [Limap](https://github.com/cvg/limap) in small-scale indoor localization and achieves the best results in outdoor settings, setting a new benchmark for learning-based approaches. It also operates in real-time at ~16 FPS, compared to [Limap](https://github.com/cvg/limap)’s ~0.03 FPS, while requiring only lightweight network weights of 33MB instead of [Limap](https://github.com/cvg/limap)’s multi-GB memory footprint.

---  
## Papers
**Improved 3D Point-Line Mapping Regression for Camera Relocalization**![new](assets/New.png)  
Bach-Thuan Bui, Huy-Hoang Bui, Yasuyuki Fujii, Dinh-Tuan Tran, and Joo-Ho Lee.   
IEEE/RSJ International Conference on Intelligent Robots and Systems (**IROS**), 2025.
[pdf](https://arxiv.org/pdf/2502.20814)  

**Representing 3D sparse map points and lines for camera relocalization**  
Bach-Thuan Bui, Huy-Hoang Bui, Dinh-Tuan Tran, and Joo-Ho Lee.    
IEEE/RSJ International Conference on Intelligent Robots and Systems (**IROS**), 2024.
[pdf](https://arxiv.org/abs/2402.18011) 


## Installation
Python 3.9 + required packages
```
git clone https://github.com/ais-lab/pl2map.git
cd pl2map
git submodule update --init --recursive
conda create --name pl2map python=3.9
conda activate pl2map
# Refer to https://pytorch.org/get-started/previous-versions/ to install pytorch compatible with your CUDA
python -m pip install torch==1.12.0 torchvision==0.13.0 
python -m pip install -r requirements.txt
```
## Supported datasets
- [Microsoft 7scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)
- [Cambridge Landmarks](https://www.repository.cam.ac.uk/handle/1810/251342/)
- [Indoor-6](https://github.com/microsoft/SceneLandmarkLocalization)

Please run the provided scripts to prepare and download the data which has been preprocessed by running:

7scenes
```
./prepare_scripts/seven_scenes.sh
```
Cambridge Landmarks
```
./prepare_scripts/cambridge.sh 
```
Indoor-6
```
./prepare_scripts/indoor6.sh
```

## Evaluation with pre-trained models
Please download the pre-trained models by running:
```
./prepare_scripts/download_pre_trained_models.sh
```
For example, to evaluate KingsCollege scene:
```
python runners/eval.py --dataset Cambridge --scene KingsCollege -expv pl2map
```

## Training
```
python runners/train.py --dataset Cambridge --scene KingsCollege -expv pl2map_test
```

## Supported detectors
### Lines
- [LSD](https://github.com/iago-suarez/pytlsd)
- [DeepLSD](https://github.com/cvg/DeepLSD)
### Points
- [Superpoint](https://github.com/rpautrat/SuperPoint)


## Citation
If you use this code in your project, please consider citing the following paper:
```bibtex
@article{bui2024pl2map,
  title={Representing 3D sparse map points and lines for camera relocalization},
  author={Bui, Bach-Thuan and Bui, Huy-Hoang and Tran, Dinh-Tuan and Lee, Joo-Ho},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2024}
}
@article{bui2025improved,
  title={Improved 3D Point-Line Mapping Regression for Camera Relocalization},
  author={Bui, Bach-Thuan and Bui, Huy-Hoang and Fujii, Yasuyuki and Tran, Dinh-Tuan and Lee, Joo-Ho},
  journal={arXiv preprint arXiv:2502.20814},
  year={2025}
}
```
This code builds on previous camera relocalization pipeline, namely [D2S](https://github.com/ais-lab/d2s), please consider citing:
```bibtex
@article{bui2024d2s,
  title={D2S: Representing sparse descriptors and 3D coordinates for camera relocalization},
  author={Bui, Bach-Thuan and Bui, Huy-Hoang and Tran, Dinh-Tuan and Lee, Joo-Ho},
  journal={IEEE Robotics and Automation Letters},
  year={2024}
}
```

## Acknowledgement
This code is built based on [Limap](https://github.com/cvg/limap), and [LineTR](https://github.com/yosungho/LineTR). We thank the authors for their useful source code.


