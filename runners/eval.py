from pathlib import Path
import argparse
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
import util.config as utilcfg
from omegaconf import OmegaConf
from evaluator import Evaluator

def parse_config():
    arg_parser = argparse.ArgumentParser(description='pre-processing for PL2Map dataset')
    arg_parser.add_argument('-d', '--dataset_dir', type=Path, default='../datasets/imgs_datasets/', help='')
    arg_parser.add_argument('--sfm_dir', type=Path, default='../datasets/gt_3Dmodels/', help='sfm ground truth directory')
    arg_parser.add_argument('--dataset', type=str, default="7scenes", help='dataset name')
    arg_parser.add_argument('-s', '--scene', type=str, default="stairs", help='scene name(s)')
    arg_parser.add_argument('-c','--cudaid', type=int, default=0, help='specify cuda device id')
    arg_parser.add_argument('-o','--outputs', type=Path, default='logs/',
                        help='Path to the output directory, default: %(default)s')
    arg_parser.add_argument('-expv', '--experiment_version', type=str, default="00_00_00", help='experiment version folder')
    args, _ = arg_parser.parse_known_args()
    args.outputs = os.path.join(args.outputs, args.scene + "_" + args.experiment_version)
    path_to_eval_cfg = f'{args.outputs}/config.yaml'
    cfg = utilcfg.load_config(path_to_eval_cfg, default_path='cfgs/default.yaml')
    cfg = OmegaConf.create(cfg)
    return args, cfg

def main():
    eval_cfg = {
        "eval_train": False, # evaluate train_loader
        "eval_test": True, # evaluate test_loader
        "vis_point3d": False, # visualize predicted 3D points, if eval_train/test = True
        "vis_line3d": False, # visualize predicted 3D lines, if eval_train/test = True
        "pnp_point": True, # use point-mode-only for PnP
        "pnp_pointline": True, # use point+line mode for PnP
        "uncer_threshold_point": 0.5, # threshold to remove uncertain points
        "uncer_threshold_line": 0.02, # threshold to remove uncertain lines
        "exist_results":False, # if True, skip running model,then use the existing results in the outputs folder
        "save_3dmap": True, # save predicted 3D map
    }
    args, cfg = parse_config()
    evaler = Evaluator(args, cfg, eval_cfg)
    evaler.eval()

if __name__ == "__main__":
    main()