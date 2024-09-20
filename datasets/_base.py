from typing import NamedTuple
import numpy as np
import os
import sys
import torch
import math
from scipy.spatial.transform import Rotation as R
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detectors.line2d.register_linedetector import get_linedetector
from detectors.point2d.register_pointdetector import get_pointdetector
from util.io import read_image
import copy
import datasets.augmentation as aug


def frame2tensor(frame, device):
    return torch.from_numpy(frame/255.).float()[None, None].to(device)

class Camera():
    '''
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    '''
    model_name2id = {"SIMPLE_PINHOLE": 0, "PINHOLE": 1,
                     "SIMPLE_RADIAL": 2, "RADIAL": 3}
    
    def __init__(self, camera, iscolmap=True) -> None:
        if iscolmap:
            self.name = camera.model
            self.get_camera_vector_colmap(camera)
        else: # list type 
            self.name = camera[0]
            self.camera_array = np.array([self.model_name2id[self.name]] + camera[1:])
    def get_camera_vector_colmap(self, camera):
        '''
        Return a camera vector from a colmap camera object
        return: numpy array of camera vector
                [modelid, width, height, focal,..., cx, cy,...]
        '''
        id = self.model_name2id[camera.model]
        array = [id, camera.width, camera.height]
        array.extend(camera.params)
        self.camera_array =  np.array(array)
        
    def update_scale(self, scale_factor):
        self.camera_array[1:] = self.camera_array[1:]*scale_factor
    
    def get_dict_camera(self):
        '''
        Return a dictionary of camera
        '''
        return {"model": self.name, "width": self.camera_array[1], "height": self.camera_array[2],
                "params": self.camera_array[3:].tolist()}

class Line3D():
    def __init__(self, start, end) -> None:
        self.start = np.asarray(start)
        self.end = np.asarray(end)
    def get_line3d_vector(self):
        return np.hstack([self.start, self.end])
    
class Pose():
    def __init__(self, qvec, tvec) -> None:
        self.qvec = qvec # quaternion, [w,x,y,z]
        self.tvec = tvec # translation,  [x,y,z]
    def get_pose_vector(self):
        """
        Return a pose vector [tvec, qvec]
        """
        return np.hstack([self.tvec, self.qvec])
    def get_pose_Tmatrix(self):
        """
        Return a pose matrix [R|t]
        """
        # Convert the quaternion to a rotation matrix
        qvec = np.zeros(4)
        qvec[:3] = self.qvec[1:] # convert quaternion from [w,x,y,z] (colmap) to [x,y,z,w] (scipy)
        qvec[3] = self.qvec[0]
        rotation = R.from_quat(qvec)
        rotation_matrix = rotation.as_matrix()
        # Create a 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = self.tvec
        return T
    
    def rotate(self, angle):
        pose = self.get_pose_Tmatrix()
        angle = -angle * math.pi / 180 # convert to radian, and reverse direction != opencv
        pose_rot = np.eye(4) 
        pose_rot[0, 0] = math.cos(angle)
        pose_rot[0, 1] = -math.sin(angle)
        pose_rot[1, 0] = math.sin(angle)
        pose_rot[1, 1] = math.cos(angle)
        pose = np.matmul(pose, pose_rot)
        self.tvec = pose[:3, 3]
        rotation = R.from_matrix(pose[:3, :3])
        qvec = rotation.as_quat()
        self.qvec = np.hstack([qvec[3], qvec[:3]]) # convert quaternion from [x,y,z,w] to [w,x,y,z] colmap

class Image_Class():
    def __init__(self,imgname:str) -> None:
        '''
        - image class for storing 2D & 3D points, 2D & 3D lines, camera vector, pose vector
        - comments with ### can be changed if augmenting data, otherwise must be fixed
                        #@ no change, but can be reduced if augmenting data
        '''
        self.points2Ds = None  ### numpy matrix of 2D points (Nx2)
        self.points3Ds = None #@ numpy matrix of 3D points, including np.array[0,0,0] if not available
        self.validPoints = None # numpy array of valid 2D points (have 2D-3D points correspondence)
        self.line2Ds = None ### numpy matrix of 2D line segments (Nx4)
        self.line3Ds = None #@ list of 3D line segment objects, including None if not available
        self.line3Ds_matrix = None # numpy matrix of 3D line segments, including np.array[0,0,0,0,0,0] if not available
        self.validLines = None # numpy array of valid 3D lines (have 2D-3D lines correspondence)
        self.camera = None # camera class
        self.id = None
        self.imgname = imgname # string: image name
        self.pose = None  ### Pose object
    def get_line3d_matrix(self):
        '''
        Return a matrix of line3D vectors
        '''
        self.line3Ds_matrix = np.stack([ii.get_line3d_vector() if ii is not None else 
                                np.array([0,0,0,0,0,0]) for ii in self.line3Ds], 0)
        self.validLines = np.stack([1 if ii is not None else 
                                0 for ii in self.line3Ds], 0)
    

class Base_Collection():
    def __init__(self, args, cfg, mode) -> None:
        self.args = args
        self.cfg = cfg
        self.device = f'cuda:{args.cudaid}' if torch.cuda.is_available() else 'cpu'
        if mode == "test":
            self.get_detector_models()
    
    def get_point_detector_model(self):
        '''
        Return a point detector model
        '''
        configs = self.cfg.point2d.detector.configs
        method = self.cfg.point2d.detector.name
        return get_pointdetector(method = method, configs=configs)
    
    def get_line_detector_model(self):
        '''
        Return a line detector model
        '''
        max_num_2d_segs = self.cfg.line2d.max_num_2d_segs
        do_merge_lines = self.cfg.line2d.do_merge_lines
        visualize = self.cfg.line2d.visualize
        method = self.cfg.line2d.detector.name_test_model
        return get_linedetector(method= method, max_num_2d_segs=max_num_2d_segs,
                 do_merge_lines=do_merge_lines, visualize=visualize, cudaid=self.args.cudaid)
    
    def get_detector_models(self):
        self.line_detector = self.get_line_detector_model()
        # self.point_detector = self.get_point_detector_model().eval().to(self.device)
    
    def do_augmentation(self, image, image_infor_class, debug = False):
        if not aug.is_apply_augment(self.cfg.train.augmentation.on_rate):
            # No apply augmentation
            return image, image_infor_class
        # Apply the brightness and contrast
        transf_image = aug.random_brightness_contrast(image, self.cfg.train.augmentation.brightness,
                                                      self.cfg.train.augmentation.contrast)
        points2Ds = image_infor_class.points2Ds
        lines2Ds = image_infor_class.line2Ds
        camera = image_infor_class.camera
        pose = image_infor_class.pose
        if self.cfg.train.augmentation.homography.apply:
            # camera and pose are not correct after applying homography
            H,W = image.shape
            shape = np.array([H,W])
            h_matrix = aug.sample_homography(shape, self.cfg.train.augmentation.homography) # sample homography matrix
            transf_image = aug.warpPerspective_forimage(transf_image, h_matrix)
            points2Ds = aug.perspectiveTransform_forpoints(image_infor_class.points2Ds, h_matrix)
            lines2Ds = aug.perspectiveTransform_forlines(image_infor_class.line2Ds, h_matrix)            
        
        # dsacstar-like augmentation method.
        if self.cfg.train.augmentation.dsacstar.apply:
            # camera and pose will be corrected in this augmentation
            assert not self.cfg.train.augmentation.homography.apply, "dsacstar augmentation cannot be applied with homography augmentation"
            transf_image, points2Ds, lines2Ds, camera, pose = aug.dsacstar_augmentation(
                transf_image, self.cfg.train.augmentation.dsacstar, points2Ds, lines2Ds, camera, pose)

        if debug:
            from util.visualize import visualize_img_withlinesandpoints
            visualize_img_withlinesandpoints(image, image_infor_class.points2Ds,image_infor_class.line2Ds)
        image_infor_class.points2Ds = points2Ds
        image_infor_class.line2Ds = lines2Ds
        image_infor_class.camera = camera
        image_infor_class.pose = pose
        # correct points and lines inside image
        if self.cfg.train.augmentation.dsacstar.apply:
            image_infor_class = aug.correct_points_lines_inside_image(transf_image.shape, image_infor_class)
        if debug:
            visualize_img_withlinesandpoints(transf_image, image_infor_class.points2Ds,image_infor_class.line2Ds, True)
        return transf_image, image_infor_class
    
    def image_loader(self, image_name, augmentation=False, debug = False):
        '''
        (use only for point2d detector model)
        Read an image, do augmentation if needed, preprocess it, and 
        return a dictionary of image data and a Image_Class object
        '''
        resize_max = self.cfg.point2d.detector.preprocessing.resize_max
        resize_force = self.cfg.point2d.detector.preprocessing.resize_force
        interpolation = self.cfg.point2d.detector.preprocessing.interpolation
        grayscale = self.cfg.point2d.detector.preprocessing.grayscale
        path_to_image = self.get_image_path(image_name)
        image = read_image(path_to_image, grayscale=grayscale)
        
        size = image.shape[:2][::-1]
        if resize_force and (max(size) > resize_max):
            scale = resize_max / max(size)
            size_new = tuple(int(round(x*scale)) for x in size)
            image = aug.resize_image(image, size_new, interpolation)
            # rescale 2D points and lines, camera focal length
            raise NotImplementedError
        
        image_infor_class = copy.deepcopy(self.imgname2imgclass[image_name])
        if augmentation:
            image, image_infor_class = self.do_augmentation(image, image_infor_class, debug)
            if debug:
                print("Debugging image_loader")
                return None
            
        image = image.astype(np.float32)
        if grayscale:
            image = image[None]
        else:
            image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
        image = image / 255.
        original_size = np.array(size)
        data = {
            'image': image,
            'original_size': original_size,
        }
        return data, image_infor_class
    
    def detect_points2D(self, image_name):
        '''
        Read an image, preprocess it, and  
        Return a keypoints from that image using
        loaded point detector model. 
        '''
        point_detector = self.get_point_detector_model().eval().to(self.device)
        resize_force = self.cfg.point2d.detector.preprocessing.resize_force
        resize_max = self.cfg.point2d.detector.preprocessing.resize_max
        data,_ = self.image_loader(image_name, False)
        data['image'] = torch.from_numpy(data['image'][None]).float().to(self.device)
        keypointsdict = point_detector._forward_default(data)
        scale = resize_max / max(data['original_size'])
        if resize_force and (max(data['original_size']) > resize_max):
            keypointsdict['keypoints'][0] = (keypointsdict['keypoints'][0] + .5)/scale - .5
        else:
            keypointsdict['keypoints'][0] += .5
        return keypointsdict
        
    def detect_lines2D(self, image_name):
        '''
        Return a list of lines2D in the image
        '''
        grayscale = self.cfg.line2d.preprocessing.grayscale
        image_path = self.get_image_path(image_name)
        image = read_image(image_path, grayscale=grayscale)
        if self.line_detector.get_module_name() == "deeplsd":
            image = frame2tensor(image, self.device)
        segs = self.line_detector.detect(image)
        return segs
    def get_2dpoints_lines_for_testing(self, image_name):
        '''
        Return a list of points2D and a list of lines2D in the image
        '''
        raise NotImplementedError
    
    def get_image_path(self, image_name):
        '''
        Return a path to image
        '''
        img_path = os.path.join(self.args.dataset_dir, self.args.dataset, self.args.scene, image_name)
        return img_path
        

