import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util.read_write_model import read_model
import numpy as np
from pathlib import Path
from datasets._base import (Image_Class, Base_Collection, Line3D, Pose,
                            Camera)

strlist2floatlist = lambda strlist: [float(s) for s in strlist]
strlist2intlist = lambda strlist: [int(s) for s in strlist]

class DataCollection(Base_Collection):
    def __init__(self, args:dict, cfg:dict, mode="train")->None:
        super(DataCollection, self).__init__(args, cfg, mode)
        self.gt_3Dmodels_path = self.args.sfm_dir / f"{self.args.dataset}/{self.args.scene}"
        self.SfM_with_depth = self.cfg.train.use_depth # use SfM labels which has been corrected by depth or not
        self.train_imgs = [] # list of train image names
        self.test_imgs = [] # list of test image names
        self.imgname2limapID = {} # map from image name to limap image id
        self.limapID2imgname = {} # map from limap image id to image name
        # load all images 2D & 3D points and create Image_Class objects
        self.imgname2imgclass = {} # map from image name to Image_Class object
        self.load_all_2Dpoints_by_dataset(self.args.dataset)
        self.load_imgname2limapID()
        # load lines data from Limap output
        self.load_all_2Dlines_data()
        # load alltracks data from Limap output
        self.load_alltracks_limap_3Dlines()
    
    def load_all_2Dpoints_by_dataset(self, dataset):
        if dataset == "7scenes":
            self.gt_3Dmodels_path = self.args.sfm_dir / f"{self.args.dataset}_{self.cfg.train.ground_truth}/{self.args.scene}"
            self.load_all_2Dpoints_7scenes()
        elif dataset == "Cambridge":
            self.load_all_2Dpoints_Cambridge()
        elif dataset == "indoor6_UnDis": 
            self.load_all_2Dpoints_indoor6_UnDis()
        else:
            raise NotImplemented
        

    def load_all_2Dpoints_7scenes(self):
        # currently used for 7scenes. 
        # load all 2d & 3d points from colmap output
        path_gt_3Dmodels_full = self.gt_3Dmodels_path/"sfm_sift_full"
        if self.SfM_with_depth:
            print("[INFOR] Using SfM labels corrected by depth.")
            path_gt_3Dmodels_train = self.gt_3Dmodels_path/"sfm_superpoint+superglue+depth"
        else:
            path_gt_3Dmodels_train = self.gt_3Dmodels_path/"sfm_superpoint+superglue"
        testlist_path = path_gt_3Dmodels_full/"list_test.txt"
        cameras_all, images_all, _ = read_model(path=path_gt_3Dmodels_full, ext=".bin")
        _, images_train, points3D_train = read_model(path=path_gt_3Dmodels_train, ext=".bin")
        name2id_train = {image.name: i for i, image in images_train.items()}

        if os.path.exists(testlist_path):
            with open(testlist_path, 'r') as f:
                testlist = f.read().rstrip().split('\n')
        else:
            raise ValueError("Error! Input file/directory {0} not found.".format(testlist_path))
        for id_, image in images_all.items():
            img_name = image.name
            self.imgname2imgclass[img_name] = Image_Class(img_name)
            if image.name in testlist:
                # fill data to TEST img classes 
                self.test_imgs.append(img_name)
                self.imgname2imgclass[img_name].pose = Pose(image.qvec, image.tvec)
                self.imgname2imgclass[img_name].camera = Camera(cameras_all[image.camera_id],
                                                                          iscolmap=True)
            else:
                # fill data to TRAIN img classes
                self.train_imgs.append(img_name)
                self.imgname2imgclass[img_name].pose = Pose(image.qvec, image.tvec)
                image_train = images_train[name2id_train[img_name]]
                self.imgname2imgclass[img_name].points2Ds = image_train.xys
                self.imgname2imgclass[img_name].points3Ds = np.stack([points3D_train[ii].xyz if ii != -1 else 
                                np.array([0,0,0]) for ii in image_train.point3D_ids], 0)
                self.imgname2imgclass[img_name].validPoints = np.stack([1 if ii != -1 else 
                                0 for ii in image_train.point3D_ids], 0)
                self.imgname2imgclass[img_name].camera = Camera(cameras_all[image.camera_id],
                                                                          iscolmap=True)
    
    def load_all_2Dpoints_Cambridge(self):
        # load all 2d & 3d points from colmap output
        path_gt_3Dmodels_full = self.gt_3Dmodels_path/"sfm_sift_full"
        
        # load query_list_with_intrinsics.txt 
        query_list_with_intrinsics = self.gt_3Dmodels_path/"query_list_with_intrinsics.txt"
        if not os.path.exists(query_list_with_intrinsics):
            raise ValueError("Error! Input file/directory {0} not found.".format(query_list_with_intrinsics))
        query_list_with_intrinsics = pd.read_csv(query_list_with_intrinsics, sep=" ", header=None)
        # get test dictionary with its intrinsic 
        testimgname2intrinsic = {query_list_with_intrinsics.iloc[i,0]:list(query_list_with_intrinsics.iloc[i,1:]) 
                                   for i in range(len(query_list_with_intrinsics))}
        
        # load id_to_origin_name.txt 
        import json 
        id_to_origin_name = self.gt_3Dmodels_path / "id_to_origin_name.txt"
        with open(id_to_origin_name, 'r') as f:
            id_to_origin_name = json.load(f)

        originalname2newimgname = {}
        for id, originalname in id_to_origin_name.items():
            id = int(id)
            originalname2newimgname[originalname] = "image{0:08d}.png".format(id)
        
        
        # load the camera model from colmap output
        _, images_all, _ = read_model(path=path_gt_3Dmodels_full, ext=".bin")
        path_gt_3Dmodels_train = self.gt_3Dmodels_path/"sfm_superpoint+superglue"
        cameras_train, images_train, points3D_train = read_model(path=path_gt_3Dmodels_train, ext=".bin")
        name2id_train = {image.name: i for i, image in images_train.items()}
        
        for _, image in images_all.items():
            img_name = image.name
            new_img_name = originalname2newimgname[img_name]
            self.imgname2imgclass[new_img_name] = Image_Class(new_img_name)
            if new_img_name in testimgname2intrinsic:
                # fill data to TEST img classes 
                self.test_imgs.append(new_img_name)
                self.imgname2imgclass[new_img_name].pose = Pose(image.qvec, image.tvec)
                self.imgname2imgclass[new_img_name].camera = Camera(testimgname2intrinsic[new_img_name],
                                                                          iscolmap=False)
            else:
                # fill data to TRAIN img classes
                if new_img_name not in name2id_train:
                    continue
                self.train_imgs.append(new_img_name)
                self.imgname2imgclass[new_img_name].pose = Pose(image.qvec, image.tvec)
                image_train = images_train[name2id_train[new_img_name]]
                self.imgname2imgclass[new_img_name].points2Ds = image_train.xys
                self.imgname2imgclass[new_img_name].points3Ds = np.stack([points3D_train[ii].xyz if ii != -1 else 
                                np.array([0,0,0]) for ii in image_train.point3D_ids], 0)
                self.imgname2imgclass[new_img_name].validPoints = np.stack([1 if ii != -1 else 
                                0 for ii in image_train.point3D_ids], 0)
                self.imgname2imgclass[new_img_name].camera = Camera(cameras_train[image_train.camera_id],
                                                                          iscolmap=True)
    
    def load_all_2Dpoints_indoor6_UnDis(self):
        # load all 2d & 3d points from colmap output
        path_gt_3Dmodels_full = self.gt_3Dmodels_path/"sfm_sift_full"
        
        # load query_list_with_intrinsics.txt 
        query_list_with_intrinsics = self.gt_3Dmodels_path/"query_list_with_intrinsics.txt"
        if not os.path.exists(query_list_with_intrinsics):
            raise ValueError("Error! Input file/directory {0} not found.".format(query_list_with_intrinsics))
        query_list_with_intrinsics = pd.read_csv(query_list_with_intrinsics, sep=" ", header=None)
        # get test dictionary with its intrinsic 
        testimgname2intrinsic = {query_list_with_intrinsics.iloc[i,0]:list(query_list_with_intrinsics.iloc[i,1:]) 
                                   for i in range(len(query_list_with_intrinsics))}
        
        # load id_to_origin_name.txt 
        import json 
        id_to_origin_name = self.gt_3Dmodels_path / "id_to_origin_name.txt"
        with open(id_to_origin_name, 'r') as f:
            id_to_origin_name = json.load(f)

        originalname2newimgname = {}
        for id, originalname in id_to_origin_name.items():
            id = int(id)
            originalname2newimgname[originalname] = "image{0:08d}.png".format(id)
        
        
        # load the camera model from colmap output
        _, images_all, _ = read_model(path=path_gt_3Dmodels_full, ext=".bin")
        path_gt_3Dmodels_train = self.gt_3Dmodels_path/"sfm_superpoint+superglue"
        cameras_train, images_train, points3D_train = read_model(path=path_gt_3Dmodels_train, ext=".bin")
        name2id_train = {image.name: i for i, image in images_train.items()}
        
        for _, image in images_all.items():
            img_name = image.name
            new_img_name = originalname2newimgname[img_name]
            self.imgname2imgclass[new_img_name] = Image_Class(new_img_name)
            if new_img_name in testimgname2intrinsic:
                # fill data to TEST img classes 
                self.test_imgs.append(new_img_name)
                self.imgname2imgclass[new_img_name].pose = Pose(image.qvec, image.tvec)
                self.imgname2imgclass[new_img_name].camera = Camera(testimgname2intrinsic[new_img_name],
                                                                          iscolmap=False)
            else:
                # fill data to TRAIN img classes
                if new_img_name not in name2id_train:
                    continue
                image_train = images_train[name2id_train[new_img_name]]
                if len(image_train.point3D_ids) == 0:
                    continue
                self.train_imgs.append(new_img_name)
                self.imgname2imgclass[new_img_name].pose = Pose(image.qvec, image.tvec)
                self.imgname2imgclass[new_img_name].points2Ds = image_train.xys
                
                self.imgname2imgclass[new_img_name].points3Ds = np.stack([points3D_train[ii].xyz if ii != -1 else 
                                np.array([0,0,0]) for ii in image_train.point3D_ids], 0)
                self.imgname2imgclass[new_img_name].validPoints = np.stack([1 if ii != -1 else 
                                0 for ii in image_train.point3D_ids], 0)
                self.imgname2imgclass[new_img_name].camera = Camera(cameras_train[image_train.camera_id],
                                                                          iscolmap=True)
    
    def load_imgname2limapID(self):
        # load path image list from limap output
        img_list_path = self.gt_3Dmodels_path/f"limap/{self.cfg.line2d.detector.name}/image_list.txt"
        if not os.path.exists(img_list_path):
            raise ValueError("Error! Input file/directory {0} not found.".format(img_list_path))
        with open(img_list_path, 'r') as f:
            lines = f.readlines()[1:]  # read all lines except the first one
            for line in lines:
                img_id, img_name = line.strip().split(',')  # assuming two columns separated by comma
                self.imgname2limapID[img_name] = int(img_id)
                self.limapID2imgname[int(img_id)] = img_name
            

    
    def load_all_2Dlines_data(self):
        # load train all lines data from limap output (all exixting lines in all images)
        # then create line3D objects for each image
        segments_path = self.gt_3Dmodels_path /f"limap/{self.cfg.line2d.detector.name}/segments"
        def read_segments_file(img_id, segments_path):
            segments_file = segments_path / f"segments_{img_id}.txt"
            if not os.path.exists(segments_file):
                raise ValueError("Error! Input file/directory {0} not found.".format(segments_file))
            segments_matrix = pd.read_csv(segments_file, sep=' ', skiprows=1, header=None).to_numpy()
            return segments_matrix
        for img_name in self.train_imgs:
            img_id = self.imgname2limapID[img_name]
            segments_matrix = read_segments_file(img_id, segments_path)
            length = segments_matrix.shape[0]
            self.imgname2imgclass[img_name].line2Ds = segments_matrix
            self.imgname2imgclass[img_name].line3Ds = [None for _ in range(length)]
            # if length < 80:
            #     print(length, img_name)
                
    
    
    def load_alltracks_limap_3Dlines(self):
        # load all tracks data from limap output (training data only)
        track_file = "fitnmerge_alltracks.txt" if self.SfM_with_depth else "alltracks.txt"
        tracks_path = self.gt_3Dmodels_path/ f"limap/{self.cfg.line2d.detector.name}/{track_file}"
        if not os.path.exists(tracks_path):
            raise ValueError("Error! Input file/directory {0} not found.".format(tracks_path))
        with open(tracks_path, 'r') as f:
            lines = f.readlines()
            number3Dlines = int(lines[0].strip())
            i = 1
            length = len(lines)
            while i < length:
                i += 1 # skip the first line (3dline id, #2dlines, #imgs)
                start_3d = strlist2floatlist(lines[i].strip().split(' '))
                i += 1
                end_3d = strlist2floatlist(lines[i].strip().split(' '))
                i += 1
                # load img ids 
                img_ids = strlist2intlist(lines[i].strip().split(' '))
                i += 1
                # load 2d line ids 
                line2d_ids = strlist2intlist(lines[i].strip().split(' '))
                # fill data to Image_Class objects
                for img_id, line2d_id in zip(img_ids, line2d_ids):
                    img_name = self.limapID2imgname[img_id]
                    if img_name not in self.train_imgs:
                        continue
                    if self.imgname2imgclass[img_name].line3Ds[line2d_id] is not None:
                        raise ValueError("Error! 3D line {0} in image {1} is already filled.".format(line2d_id, img_id))
                    self.imgname2imgclass[img_name].line3Ds[line2d_id] = Line3D(start_3d, end_3d)
                i += 1
        self.load_all_lines3D_matrix()
        
    def load_all_lines3D_matrix(self):
        # load all lines3D matrix from colmap output
        for img_name in self.train_imgs:
            self.imgname2imgclass[img_name].get_line3d_matrix()



