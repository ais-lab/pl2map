import torch
from torch.utils.data import Dataset
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.data_collection import DataCollection
import numpy as np

class Collection_Loader(Dataset):
    def __init__(self, args, cfg, mode="train"):
        self.DataCol = DataCollection(args, cfg, mode=mode)
        self.mode = mode
        if "train" in mode:
            self.image_list = self.DataCol.train_imgs
            self.augmentation = cfg.train.augmentation.apply if mode == "train" else False
            if self.augmentation: print("[INFOR] Augmentation is applied")
        elif mode == "test":
            self.augmentation = False
            self.image_list = self.DataCol.test_imgs
        else:
            raise ValueError("Error! Mode {0} not supported.".format(mode))
        # # sort image_list
        # self.image_list = sorted(self.image_list)
        # # create new image_list with uniform sample of only 30 images 
        # self.image_list = self.image_list[::int(len(self.image_list)/30)]
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_name = self.image_list[index]
        data, infor = self.DataCol.image_loader(image_name, augmentation=self.augmentation) # dict:{img, ori_img_size}
        target = {}
        if 'train' in self.mode:
            if sum(infor.validPoints) == 0:
                index = 0 
                image_name = self.image_list[index]
                data, infor = self.DataCol.image_loader(image_name, augmentation=self.augmentation)
        if self.mode == "test":
            data['lines'] = self.DataCol.detect_lines2D(image_name)[:,:4] # detect lines2D
            data['keypoints'] = 'None' # to show there is no keypoints
        if "train" in self.mode:
            data['lines'] = infor.line2Ds
            data['keypoints'] = infor.points2Ds
            target['lines3D'] = infor.line3Ds_matrix.T
            target['points3D'] = infor.points3Ds.T
            target['validPoints'] = infor.validPoints
            target['validLines'] = infor.validLines
            
            data['validPoints'] = infor.validPoints
            data['validLines'] = infor.validLines
            assert data['lines'].shape[0] == target['lines3D'].shape[1] == target['validLines'].shape[0]
            assert data['keypoints'].shape[0] == target['points3D'].shape[1] == target['validPoints'].shape[0]
        target['pose'] = infor.pose.get_pose_vector()
        target['camera'] = infor.camera.camera_array
        data['imgname'] = image_name
        data = map_dict_to_torch(data)
        target = map_dict_to_torch(target)
        return data, target

def map_dict_to_torch(data):
    for k, v in data.items():
        if isinstance(v, str):
            continue
        elif isinstance(v, np.ndarray):
            data[k] = torch.from_numpy(v).float()
        else:
            raise ValueError("Error! Type {0} not supported.".format(type(v)))
    return data
