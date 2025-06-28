import numpy as np 
import torch
import torch.nn as nn
from models.base_model import BaseModel
from models.util import get_model
import os
import os.path as osp

class Pipeline(BaseModel):
    default_conf = {
        'trainable': True,
    }
    required_data = ['image', 'original_size', 'keypoints', 'lines']

    def _init(self, conf):
        # get detector model 
        self.detector = get_model(conf.point2d.detector.name, "detector")(conf.point2d.detector.configs)
        assert self.detector.conf.trainable == False, "detector must be fixed, not trainable"
        # get regressor model
        self.regressor = get_model(conf.regressor.name, "regressor")(conf.regressor).train()
        assert self.regressor.conf.trainable == True, "regressor must be trainable"
        print(f'The model regresor {conf.regressor.name} has {count_parameters(self.regressor):,} trainable parameters')

    def _forward(self, data):
        # Pre process data
        # convert lines to line_keypoints | BxLx4 -> BxLx(4+n_line_keypoints*2)
        line_keypoints = get_line_keypoints(data['lines'], self.conf.regressor.n_line_keypoints)
        # sample descriptors using superpoint
        regressor_data = self.detector((data['image'], data['keypoints'], line_keypoints))
        if self.training or 'validPoints' in data.keys():
            regressor_data['validPoints'] = data['validPoints']
            regressor_data['validLines'] = data['validLines']
        # regress descriptors to 3D points and lines
        pred = self.regressor(regressor_data)
        pred['keypoints'] = regressor_data['keypoints']
        pred['lines'] = data['lines']
        return pred

    def loss(self, pred, data):
        pass
    def save_checkpoint(self, path, name, epoch, final = False):
        if os.path.exists(path) == False:
            os.makedirs(path)
        filename = osp.join(path, '{}_final.pth.tar'.format(name)) if final \
            else osp.join(path, '{}.pth.tar'.format(name))
        checkpoint_dict =\
            {'epoch': epoch, 'model_state_dict': self.regressor.state_dict()}
        torch.save(checkpoint_dict, filename)
    def load_checkpoint(self, path, exp_name):
        ''' Load regressor checkpoint from path'''
        filename = osp.join(path, '{}.pth.tar'.format(exp_name))
        if not osp.exists(filename):
            raise FileNotFoundError(f'Cannot find checkpoint at {filename}')
        devide = torch.device(f'cuda:{torch.cuda.current_device()}' \
                                       if torch.cuda.is_available() else 'cpu')
        checkpoint_dict = torch.load(filename, map_location=torch.device(devide))
        self.regressor.load_state_dict(checkpoint_dict['model_state_dict'])
        print(f'[INFOR] Loaded checkpoint from {filename}')
        return checkpoint_dict['epoch']

def get_line_keypoints(lines, n_line_keypoints):
    # convert lines to line_keypoints | BxLx4 -> BxLx(n_line_keypoints+2)x2
    bs,n_line,_ = lines.shape
    total_points = n_line_keypoints + 2 # start point + end point + n_line_keypoints
    line_keypoints = lines.new_zeros((bs,n_line, total_points,2))
    line_keypoints[:,:,0,:] = lines[:,:,:2] # start point
    line_keypoints[:,:,total_points-1,:] = lines[:,:,2:] # end point
    per_distance = (lines[:,:,2:] - lines[:,:,:2])/(n_line_keypoints+2-1) # stop - start point
    for i in range(n_line_keypoints):
        line_keypoints[:,:,i+1,:] = line_keypoints[:,:,0,:] + per_distance*(i+1)
    return line_keypoints

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

