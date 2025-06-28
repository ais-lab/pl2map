import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
from util.read_write_model import qvec2rotmat

class Vis_Infor():
    '''
    Store and Merge the 3D lines output from the model (by remove line with high uncertainty)
    lines3D: (N, 6)
    points3D: (N, 3)
    Visualize the 3D lines and 3D points
    '''
    def __init__(self, eval_cfg, highlight_frame=None, limit_n_frames=None, save_list_imgs=False,
                 output_path=None)->None:
        '''
        highlight_frame: "seq-06/frame-000612.color.png", for example
        limit_n_frames: limit the number of frames to visualize the 3D lines and 3D points
        '''
        self.eval_cfg = eval_cfg
        self.highlight_frame = highlight_frame
        self.limit_n_frames = np.inf if limit_n_frames is None else limit_n_frames
        self.save_list_imgs = save_list_imgs
        self.output_path = output_path
        self.lines3D = None
        self.points3D = None
        self.hightlight_lines3D = None
        self.hightlight_points3D = None
        self.threshold_point = eval_cfg.uncer_threshold_point
        self.threshold_line = eval_cfg.uncer_threshold_line
        self.current_num_frames = 0
        self.list_images = [] # list of images to visualize 3D lines / 3D points
        self.cameras = []
        self.prd_poses = []
        self.gt_poses = []
    def update(self, output, data, vis_pose_infor=None):
        '''
        args:
            output: dict of model output
            data: dict of data
            pose_vis_infor: list of camera, est_pose, gt_pose # this is use for line only
        '''
        
        if self.current_num_frames < self.limit_n_frames:
            if self.eval_cfg.vis_line3d:
                lines3D = getLine3D_from_modeloutput(output['lines3D'], self.threshold_line)
                if lines3D is not None:
                    if len(lines3D.shape) != 1:
                        self.lines3D = lines3D if self.lines3D is None else np.concatenate((self.lines3D, lines3D))
                        self.list_images.append(data['imgname'][0])
                if vis_pose_infor is not None:
                    self.cameras.append(vis_pose_infor[0])
                    self.prd_poses.append(vis_pose_infor[1])
                    self.gt_poses.append(vis_pose_infor[2])
            
            if self.eval_cfg.vis_point3d:
                points3D = getPoint3D_from_modeloutput(output['points3D'], self.threshold_point)
                self.points3D = points3D if self.points3D is None else np.concatenate((self.points3D, points3D))
                self.list_images.append(data['imgname'][0])
                
        if self.limit_n_frames is not None and self.highlight_frame is not None:
            # save visualizations for the highlight 3d lines and 3d points
            current_frame = data['imgname'][0]
            if self.highlight_frame == current_frame:
                print("FOUND HIGHLIGHT FRAME")
                if self.eval_cfg.vis_line3d:
                    self.hightlight_lines3D,_ = getLine3D_from_modeloutput(output['lines3D'], self.threshold)
                if self.eval_cfg.vis_point3d:
                    self.hightlight_points3D,_ = getPoint3D_from_modeloutput(output['points3D'], self.threshold)
                if self.current_num_frames >= self.limit_n_frames:
                    self.save_vis_highlights()
        self.current_num_frames += 1
    
    def vis(self):
        if self.eval_cfg.vis_line3d:
            print("[INFOR] Visualizing predicted 3D lines ...")
            from util.visualize import open3d_vis_3d_lines
            open3d_vis_3d_lines(self.lines3D, self.cameras, self.prd_poses, self.gt_poses)
        if self.eval_cfg.vis_point3d:
            print("[INFOR] Visualizing predicted 3D points ...")
            from util.visualize import open3d_vis_3d_points
            open3d_vis_3d_points(self.points3D)
        if self.save_list_imgs:
            print("[INFOR] Saving list of images to visualize 3D lines / 3D points ...")
            with open(os.path.join(self.output_path, "list_vis_imgs.txt"), "w") as f:
                for img in self.list_images:
                    f.write(img + "\n")
            
    def save_vis_highlights(self):
        if self.hightlight_lines3D is not None:
            from util.visualize import open3d_vis_3d_lines_with_hightlightFrame
            open3d_vis_3d_lines_with_hightlightFrame(self.lines3D, self.hightlight_lines3D)
        if self.hightlight_points3D is not None:
            from util.visualize import open3d_vis_3d_points_with_hightlightFrame
            open3d_vis_3d_points_with_hightlightFrame(self.points3D, self.hightlight_points3D)

def getLine3D_from_modeloutput(lines3D, threshold=0.5):
    '''
    get uncertainty and remove line with high uncertainty
    args:
        lines3D: numpy array (1, 7, N)
    return: lines3D (N, 6)
    '''
    if lines3D is None:
        return None
    lines3D = np.squeeze(lines3D.detach().cpu().numpy())
    # uncertainty = 1/(1+100*np.abs(lines3D[6,:]))
    # lines3D = lines3D[:6,:]
    # uncertainty = [True if tmpc >= threshold else False for tmpc in uncertainty]
    # lines3D = lines3D.T[uncertainty,:]
    return lines3D.T #, uncertainty

def getPoint3D_from_modeloutput(points3D, threshold=0.5):
    '''
    get uncertainty and remove point with high uncertainty
    args:
        points3D: numpy array (1, 4, N)
    return: points3D (N, 3)
    '''
    points3D = np.squeeze(points3D.detach().cpu().numpy())
    # uncertainty = 1/(1+100*np.abs(points3D[3,:]))
    # points3D = points3D[:3,:]
    # uncertainty = [True if tmpc >= threshold else False for tmpc in uncertainty]
    # points3D = points3D.T[uncertainty,:]
    return points3D.T #, uncertainty

def pose_evaluator(eval_cfg, spath):
    '''
    Evaluate the estimated poses with ground truth poses
    args:
        eval_cfg: evaluation config
        spath: path to save the estimated poses and ground truth poses
    '''
    def eval(eval_cfg, spath, mode):
        if eval_cfg.pnp_point:
            evaluate_pose_results(spath, mode=mode, pnp='point')
        if eval_cfg.pnp_pointline:
            evaluate_pose_results(spath, mode=mode, pnp='pointline')
        if eval_cfg.pnp_line:
            evaluate_pose_results(spath, mode=mode, pnp='line')
            
    if eval_cfg.eval_train:
        mode = 'train'
        eval(eval_cfg, spath, mode)
    if eval_cfg.eval_test:
        mode = 'test'
        eval(eval_cfg, spath, mode)

def evaluate_pose_results(spath, mode='train', pnp='pointline'):
    '''
    Evaluate the estimated poses with ground truth poses 
    args:
        spath: path to save the estimated poses and ground truth poses
        mode: 'train' or 'test'
        pnp: 'point' or 'pointline'
    '''
    gt_path = os.path.join(spath, f"gt_poses_{mode}.txt")
    prd_path = os.path.join(spath, f"est_poses_{mode}_{pnp}.txt")
    gt = pd.read_csv(gt_path, header=None, sep=" ")
    prd = pd.read_csv(prd_path, header=None, sep =" ")
    # assert len(gt) == len(prd)
    errors_t = []
    errors_R = []
    num_inliers = []
    num_points = []
    num_extracted_points = []
    for i in range(len(prd)):
        num_inliers.append(prd.iloc[i,9])
        num_points.append(prd.iloc[i,10])
        num_extracted_points.append(prd.iloc[i,11])
        R_gt = qvec2rotmat(gt.iloc[i,3:7].to_numpy())
        t_gt = gt.iloc[i,:3].to_numpy()
        t = prd.iloc[i,:3].to_numpy()
        R = qvec2rotmat(prd.iloc[i,3:].to_numpy())
        e_t = np.linalg.norm(-R_gt.T @ t_gt + R.T @ t, axis=0)
        cos = np.clip((np.trace(np.dot(R_gt.T, R)) - 1) / 2, -1., 1.)
        e_R = np.rad2deg(np.abs(np.arccos(cos)))
        errors_t.append(e_t)
        errors_R.append(e_R)
    errors_t = np.array(errors_t)
    errors_R = np.array(errors_R)
    med_t = np.median(errors_t)
    med_R = np.median(errors_R)
    print(f'Evaluation results on {mode} set ({len(gt)}imgs) & PnP {pnp}:')
    print('Median errors: {:.4f}m, {:.4f}deg'.format(med_t, med_R))
    print('Average PnP time: {:.4f}s'.format(np.mean(prd.iloc[:,7].to_numpy())))
    print('Average extracted/fine_points/inliers: {:.1f}/{:.1f}/{:.1f}'.format(np.mean(num_extracted_points),
                                                                        np.mean(num_points), np.mean(num_inliers)))
    print('Percentage of test images localized within:')
    threshs_t = [0.01, 0.02, 0.03, 0.05, 0.10, 0.5]
    threshs_R = [1.0, 2.0, 3.0, 5.0, 10.0, 10]
    for th_t, th_R in zip(threshs_t, threshs_R):
        ratio = np.mean((errors_t < th_t) & (errors_R < th_R))
        print('\t{:.0f}cm, {:.0f}deg : {:.2f}%'.format(th_t*100, th_R, ratio*100))
    return med_t, med_R



