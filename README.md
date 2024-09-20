# Representing 3D sparse map points and lines for camera relocalization
### [Project Page](https://thpjp.github.io/pl2map/) | [Paper](https://arxiv.org/abs/2402.18011)
<br/>

> Representing 3D sparse map points and lines for camera relocalization                                                                                                                                                
> [Bach-Thuan Bui](https://thuanbb.github.io/), [Huy-Hoang Bui](https://github.com/AustrianOakvn), [Dinh-Tuan Tran](https://sites.google.com/view/tuantd), [Joo-Ho Lee](https://research-db.ritsumei.ac.jp/rithp/k03/resid/S000220;jsessionid=8CC0520A8C7C1F3D502596F0A07D64B0?lang=en)                   
> 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 

![demo_vid](assets/demo.gif)

## Installation
Python 3.9 + required packages
```
git submodule update --init --recursive

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
@article{bui2024representing,
  title={Representing 3D sparse map points and lines for camera relocalization},
  author={Bui, Bach-Thuan and Bui, Huy-Hoang and Tran, Dinh-Tuan and Lee, Joo-Ho},
  journal={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2024}
}
```

## Acknowledgement
This code is built based on [Limap](https://github.com/cvg/limap). We thank the authors for their useful source code.


