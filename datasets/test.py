from pathlib import Path
import argparse
from data_collection import DataCollection 
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import util.config as utilcfg
import util.visualize as u_vis
from omegaconf import OmegaConf

def parse_config():
    arg_parser = argparse.ArgumentParser(description='pre-processing for PL2Map dataset')
    arg_parser.add_argument('-d', '--dataset_dir', type=Path, default='datasets/imgs_datasets/', help='')
    arg_parser.add_argument('--dataset', type=str, default="7scenes", help='dataset name')
    arg_parser.add_argument('-s', '--scene', type=str, default="office", help='scene name(s)')
    arg_parser.add_argument('-cp','--checkpoint', type=int, default=0, choices=[0,1], help='use pre-trained model')
    arg_parser.add_argument('--visdom', type=int, default=1,  choices=[0,1], help='visualize loss using visdom')
    arg_parser.add_argument('-c','--cudaid', type=int, default=0, help='specify cuda device id')
    arg_parser.add_argument('--use_depth', type=int, default=0, choices=[0,1], help='use SfM corrected by depth or not')
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
    dataset = DataCollection(args, cfg, mode="train")
    # img_name = "seq-06/frame-000780.color.png"
    
    # print(dataset.imgname2imgclass[img_name].camera.camera_array)
    # print(dataset.imgname2imgclass[img_name].pose.get_pose_vector())
    
    # u_vis.visualize_2d_points_lines_from_collection(dataset, img_name, mode="online")
    # u_vis.visualize_2d_lines_from_collection(dataset, img_name, mode="online")
    # u_vis.visualize_2d_lines_from_collection(dataset, img_name, mode="offline")
    # u_vis.open3d_vis_3d_points_from_datacollection(dataset)
    # u_vis.open3d_vis_3d_lines_from_single_imgandcollection(dataset, img_name)
    u_vis.open3d_vis_3d_lines_from_datacollection(dataset)
    # u_vis.visualize_2d_points_from_collection(dataset, img_name, mode="online")
    # u_vis.visualize_2d_points_from_collection(dataset, img_name, mode="offline")
    # dataset.image_loader(img_name, cfg.train.augmentation.apply, debug=True)
    # img_name = "seq-06/frame-000499.color.png"
    # train_img_list = dataset.train_imgs
    # i = 0
    # for img_name in train_img_list:
    #     i+=1
    #     if i%5 == 0:
    #         continue
    #     print(img_name)
    #     # u_vis.visualize_2d_points_from_collection(dataset, img_name, mode="offline")
    #     # u_vis.visualize_2d_points_from_collection(dataset, img_name, mode="online")
    #     u_vis.visualize_2d_lines_from_collection(dataset, img_name, mode="offline")
    #     # u_vis.visualize_2d_lines_from_collection(dataset, img_name, mode="online")
    #     # visualize 3D train lines
    #     # u_vis.open3d_vis_3d_lines_from_datacollection(dataset)
    #     if i > 2000:
    #         break
if __name__ == "__main__":
    main()











