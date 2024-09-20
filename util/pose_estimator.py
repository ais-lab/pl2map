import numpy as np
from omegaconf import OmegaConf
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
from util.help_evaluation import getLine3D_from_modeloutput, getPoint3D_from_modeloutput
import time 
import poselib 

class Pose_Estimator():
    def __init__(self, localize_cfg, eval_cfg, spath):
        self.localize_cfg = localize_cfg # config file for localization
        self.eval_cfg = eval_cfg # local config for evaluation
        self.spath = spath
        self.uncertainty_point = eval_cfg.uncer_threshold_point
        self.uncertainty_line = eval_cfg.uncer_threshold_line
        self.pnppoint = eval_cfg.pnp_point
        self.pnppointline = eval_cfg.pnp_pointline
        if not self.eval_cfg.exist_results:
            self.checkexist()
    def checkexist(self):
        ''' 
        Check if the files exist, if yes, remove them
        '''
        trainfiles_list = ['est_poses_train_pointline.txt', 'est_poses_train_point.txt',
                       'gt_poses_train.txt']
        testfiles_list = ['est_poses_test_pointline.txt', 'est_poses_test_point.txt',
                       'gt_poses_test.txt']
        if self.eval_cfg.eval_train:
            self.rmfiles(trainfiles_list)
        if self.eval_cfg.eval_test:
            self.rmfiles(testfiles_list)

    def rmfiles(self, rm_list):
        for file in rm_list:
            if os.path.exists(os.path.join(self.spath, file)):
                os.remove(os.path.join(self.spath, file))

    def run(self, output, data, target, mode='train'):
        return camera_pose_estimation(self.localize_cfg, output, data, target, self.spath, mode=mode,
                     uncertainty_point=self.uncertainty_point, uncertainty_line=self.uncertainty_line,
                     pnppoint=self.pnppoint, pnppointline=self.pnppointline)

def camera_pose_estimation(localize_cfg, output, data, target, spath, mode='train', 
                           uncertainty_point=0.5, uncertainty_line=0.5, pnppoint=False, pnppointline=True):
    '''
    Creating same inputs for limap library and estimate camera pose
    '''
    p3ds_, point_uncer = getPoint3D_from_modeloutput(output['points3D'], uncertainty_point)
    p3ds = [i for i in p3ds_]
    p2ds = output['keypoints'][0].detach().cpu().numpy() + 0.5 # COLMAP
    p2ds = p2ds[point_uncer,:]
    p2ds = [i for i in p2ds]
    camera = target['camera'][0].detach().cpu().numpy()
    camera_model = "PINHOLE" if camera[0] == 1.0 else "SIMPLE_PINHOLE"
    poselibcamera = {'model': camera_model, 'width': camera[2], 'height': camera[1], 'params': camera[3:]}
    image_name = data['imgname'][0]
    
    if pnppoint:
        start = time.time()
        pose_point, _ = poselib.estimate_absolute_pose(p2ds, p3ds, poselibcamera, {'max_reproj_error': 12.0}, {})
        est_time = time.time() - start
        with open(os.path.join(spath, f"est_poses_{mode}_point.txt"), 'a') as f:
            f.write(f"{pose_point.t[0]} {pose_point.t[1]} {pose_point.t[2]} {pose_point.q[0]} {pose_point.q[1]} {pose_point.q[2]} {pose_point.q[3]} {est_time} {image_name}\n")
    target_pose = target['pose'][0].detach().cpu().numpy()
    with open(os.path.join(spath, f"gt_poses_{mode}.txt"), 'a') as f:
        f.write(f"{target_pose[0]} {target_pose[1]} {target_pose[2]} {target_pose[3]} {target_pose[4]} {target_pose[5]} {target_pose[6]}\n")
    if not pnppointline:
        return None
    # modify the limap pnp to poselib pnp 
    
    
    l3ds, line_uncer = getLine3D_from_modeloutput(output['lines3D'], uncertainty_line)
    l3d_ids = [i for i in range(len(l3ds))]
    l2ds = data['lines'][0].detach().cpu().numpy()
    l2ds = l2ds[line_uncer,:]
    
    localize_cfg = OmegaConf.to_container(localize_cfg, resolve=True)
    
    if pnppointline:
        start = time.time()
        ransac_opt = {"max_reproj_error": 12.0, "max_epipolar_error": 10.0}
        l2d_1 = [i for i in l2ds[:,:2]]
        l2d_2 = [i for i in l2ds[:,2:]]
        l3d_1 = [i for i in l3ds[:,:3]]
        l3d_2 = [i for i in l3ds[:,3:]]
        pose, _ = poselib.estimate_absolute_pose_pnpl(p2ds, p3ds, l2d_1, l2d_2, l3d_1, l3d_2, poselibcamera, ransac_opt)
        est_time = time.time() - start
        with open(os.path.join(spath, f"est_poses_{mode}_pointline.txt"), 'a') as f:
            f.write(f"{pose.t[0]} {pose.t[1]} {pose.t[2]} {pose.q[0]} {pose.q[1]} {pose.q[2]} {pose.q[3]} {est_time} {image_name}\n")
        return [poselibcamera, np.array([pose.t[0], pose.t[1], pose.t[2], pose.q[0], pose.q[1], pose.q[2], pose.q[3]]), target_pose]