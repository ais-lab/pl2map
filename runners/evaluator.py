import numpy as np
import torch
from tqdm import tqdm
import sys, os
from omegaconf import OmegaConf
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
from models.pipeline import Pipeline
from util.help_evaluation import Vis_Infor, pose_evaluator
from datasets.dataloader import Collection_Loader
from models.util_learner import CriterionPointLine
from trainer import step_fwd, ShowLosses
from util.pose_estimator import Pose_Estimator # require limap library
from util.io import SAVING_MAP

class Evaluator():
    default_cfg = {
        "eval_train": True, # evaluate train_loader
        "eval_test": True, # evaluate test_loader
        "vis_point3d": False, # visualize predicted 3D points, if eval_train/test = True
        "vis_line3d": False, # visualize predicted 3D lines, if eval_train/test = True
        "pnp_point": True, # use point-mode-only for PnP
        "pnp_pointline": True, # use point+line mode for PnP
        "uncer_threshold_point": 0.5, # threshold to remove uncertain points
        "uncer_threshold_line": 0.1, # threshold to remove uncertain lines
        "exist_results":False, # if True, skip running model,then use the existing results in the outputs folder
        "save_3dmap": False, # save predicted 3D map
    }
    def __init__(self, args, cfg, eval_cfg=dict()):
        self.args = args
        self.cfg = cfg
        eval_cfg = eval_cfg if cfg.regressor.name == 'pl2map' \
              else force_onlypoint_cfg(eval_cfg)
        self.eval_cfg = OmegaConf.merge(OmegaConf.create(self.default_cfg), eval_cfg)
        print(f"[INFO] Model: {cfg.regressor.name}")
        print("[INFO] Evaluation Config: ", self.eval_cfg)
        
        if not self.eval_cfg.exist_results:
            self.pipeline = Pipeline(cfg)
            self.criterion = CriterionPointLine(self.cfg.train.loss.reprojection, cfg.train.num_iters) 
            self.device = torch.device(f'cuda:{args.cudaid}' \
                                       if torch.cuda.is_available() else 'cpu')
            self.save_path = None
            # to device
            self.pipeline.to(self.device)
            self.criterion.to(self.device)
            # dataloader
            if self.eval_cfg.eval_train: self.train_collection = Collection_Loader(args, cfg, mode="traintest")
            self.eval_collection = Collection_Loader(args, cfg, mode="test")
            print("[INFO] Loaded data collection")
            if self.eval_cfg.eval_train: self.train_loader = torch.utils.data.DataLoader(self.train_collection, batch_size=1,
                                                            shuffle=True)
            self.eval_loader = torch.utils.data.DataLoader(self.eval_collection, batch_size=1,
                                                            shuffle=True)
            self.train_loss = ShowLosses()
            self.exp_name = str(args.dataset) + "_" + str(args.scene) + "_" + str(cfg.regressor.name)
            self.vis_infor_train = Vis_Infor(self.eval_cfg)
            self.vis_infor_test = Vis_Infor(self.eval_cfg)
            # self.vis_infor_test = Vis_Infor(self.eval_cfg, "seq-06/frame-000780.color.png", 20)
            if self.eval_cfg.save_3dmap: self.saving_map = SAVING_MAP(self.args.outputs)
            self.pose_estimator = Pose_Estimator(self.cfg.localization, self.eval_cfg, 
                                                self.args.outputs)
        else:
            print("[INFO] Skip running model, then use the existing results in the outputs folder")

    def eval(self):
        if not self.eval_cfg.exist_results:
            epoch = self.pipeline.load_checkpoint(self.args.outputs, self.exp_name)
            self.pipeline.eval()
            print("[INFO] Start evaluating ...")
            if self.eval_cfg.eval_train:
                print("[INFO] Evaluating train_loader ...")
                for _, (data, target) in enumerate(tqdm(self.train_loader)):
                    loss, output = step_fwd(self.pipeline, self.device, data,target,
                                             iteration=self.cfg.train.num_iters, 
                                             criterion=self.criterion, train=True)
                    self.train_loss.update(loss)
                    self.vis_infor_train.update(output, data)
                    self.pose_estimator.run(output, data, target, mode='train')
                self.train_loss.show(epoch)
                self.vis_infor_train.vis()
            if self.eval_cfg.eval_test:
                i = 0
                print("[INFO] Evaluating test_loader ...")
                for _, (data, target) in enumerate(tqdm(self.eval_loader)):
                    _, output = step_fwd(self.pipeline, self.device, data, 
                                        target, train=False)
                    if self.eval_cfg.save_3dmap: self.saving_map.save(output, data)
                    # if data['imgname'][0] == self.vis_infor_test.highlight_frame:
                    pose_vis_infor = self.pose_estimator.run(output, data, target, mode='test')
                    self.vis_infor_test.update(output, data, pose_vis_infor)
                    # i += 1
                    # if i > 20: break
                self.vis_infor_test.vis()
        else:
            print("[INFO] Skip evaluating and use the existing results")
        pose_evaluator(self.eval_cfg, self.args.outputs)
        print("[INFO] DONE evaluation")

def force_onlypoint_cfg(cfg):
    '''
    Force the evaluation config to be only point mode
    '''
    if cfg["pnp_pointline"] or cfg["vis_line3d"]: # turn off line mode, if it is on
        print("[Warning] Force the evaluation config to be only point mode")
        cfg["vis_line3d"] = False
        cfg["pnp_pointline"] = False
    return cfg