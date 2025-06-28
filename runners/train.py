from pathlib import Path
import argparse
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
import util.config as utilcfg
from omegaconf import OmegaConf
from trainer import Trainer
from util.logger import DualLogger
import time

def parse_config():
    arg_parser = argparse.ArgumentParser(description='pre-processing for PL2Map dataset')
    arg_parser.add_argument('-d', '--dataset_dir', type=Path, default='train_test_datasets/imgs_datasets/', help='')
    arg_parser.add_argument('--sfm_dir', type=Path, default='train_test_datasets/gt_3Dmodels/', help='sfm ground truth directory')
    arg_parser.add_argument('--dataset', type=str, default="7scenes", help='dataset name')
    arg_parser.add_argument('-s', '--scene', type=str, default="pumpkin", help='scene name(s)')
    arg_parser.add_argument('-cp','--checkpoint', action= 'store_true', help='use pre-trained model')
    arg_parser.add_argument('--visdom', action= 'store_true', help='visualize loss using visdom')
    arg_parser.add_argument('-c','--cudaid', type=int, default=0, help='specify cuda device id')
    arg_parser.add_argument('-o','--outputs', type=Path, default='logs/',
                        help='Path to the output directory, default: %(default)s')
    arg_parser.add_argument('-expv', '--experiment_version', type=str, default="pl2mapplus", help='experiment version folder')
    args, _ = arg_parser.parse_known_args()
    args.outputs = os.path.join(args.outputs, args.scene + "_" + args.experiment_version)
    print("Dataset: {} | Scene: {}".format(args.dataset, args.scene))
    cfg = utilcfg.load_config(f'cfgs/{args.dataset}.yaml', default_path='cfgs/default.yaml')
    cfg = OmegaConf.create(cfg)
    utilcfg.mkdir(args.outputs)

    # Save the config file for evaluation purposes
    config_file_path = os.path.join(args.outputs, 'config.yaml')
    OmegaConf.save(cfg, config_file_path)

    return args, cfg

def main():
    args, cfg = parse_config()
    sys.stdout = DualLogger(f'{args.outputs}/train_log.txt')
    trainer = Trainer(args, cfg)
    start_time = time.time()
    trainer.train()
    print("Training time: {:.2f} hours".format((time.time() - start_time) / (60*60)))
    sys.stdout.log.close() 

if __name__ == "__main__":
    main()











