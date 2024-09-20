import os
import numpy as np
from tqdm import tqdm
import cv2
from torch import nn

#import limap.util.io as limapio
#import limap.visualize as limapvis

import collections
from typing import NamedTuple
class BaseDetectorOptions(NamedTuple):
    """
    Base options for the line detector

    :param set_gray: whether to set the image to gray scale (sometimes depending on the detector)
    :param max_num_2d_segs: maximum number of detected line segments (default = 3000)
    :param do_merge_lines: whether to merge close similar lines at post-processing (default = False)
    :param visualize: whether to output visualizations into output folder along with the detections (default = False)
    :param weight_path: specify path to load weights (at default, weights will be downloaded to ~/.local)
    """
    max_num_2d_segs: int = 3000
    do_merge_lines: bool = False
    visualize: bool = False
    weight_path: str = None
    cudaid: int = 0

class LineBaseDetector():
    """
    Virtual class for line detector
    """
    def __init__(self, options = BaseDetectorOptions()):
        self.max_num_2d_segs = options.max_num_2d_segs
        self.do_merge_lines = options.do_merge_lines
        self.visualize = options.visualize
        self.weight_path = options.weight_path
        self.cudaid = options.cudaid

    # Module name needs to be set
    def get_module_name(self):
        """
        Virtual method (need to be implemented) - return the name of the module
        """
        raise NotImplementedError
    # The functions below are required for detectors
    def detect(self, data):
        """
        Virtual method (for detector) - detect 2D line segments

        Args:
            view (:class:`limap.base.CameraView`): The `limap.base.CameraView` instance corresponding to the image
        Returns:
            :class:`np.array` of shape (N, 5): line detections. Each row corresponds to x1, y1, x2, y2 and score.
        """
        raise NotImplementedError
    # The functions below are required for extractors
    def extract(self, camview, segs):
        """
        Virtual method (for extractor) - extract the features for the detected segments

        Args:
            view (:class:`limap.base.CameraView`): The `limap.base.CameraView` instance corresponding to the image
            segs: :class:`np.array` of shape (N, 5), line detections. Each row corresponds to x1, y1, x2, y2 and score. Computed from the `detect` method.
        Returns:
            The extracted feature
        """
        raise NotImplementedError
    def get_descinfo_fname(self, descinfo_folder, img_id):
        """
        Virtual method (for extractor) - Get the target filename of the extracted feature

        Args:
            descinfo_folder (str): The output folder
            img_id (int): The image id
        Returns:
            str: target filename
        """
        raise NotImplementedError
    def save_descinfo(self, descinfo_folder, img_id, descinfo):
        """
        Virtual method (for extractor) - Save the extracted feature to the target folder

        Args:
            descinfo_folder (str): The output folder
            img_id (int): The image id
            descinfo: The features extracted from the function `extract`
        """
        raise NotImplementedError
    def read_descinfo(self, descinfo_folder, img_id):
        """
        Virtual method (for extractor) - Read in the extracted feature. Dual function for `save_descinfo`.

        Args:
            descinfo_folder (str): The output folder
            img_id (int): The image id
        Returns:
            The extracted feature
        """
        raise NotImplementedError
    # The functions below are required for double-functioning objects
    def detect_and_extract(self, camview):
        """
        Virtual method (for dual-functional class that can perform both detection and extraction) - Detect and extract on a single image

        Args:
            view (:class:`limap.base.CameraView`): The `limap.base.CameraView` instance corresponding to the image
        Returns:
            segs (:class:`np.array`): of shape (N, 5), line detections. Each row corresponds to x1, y1, x2, y2 and score. Computed from the `detect` method.
            descinfo: The features extracted from the function `extract`
        """
        raise NotImplementedError
    def sample_descinfo_by_indexes(self, descinfo, indexes):
        """
        Virtual method (for dual-functional class that can perform both detection and extraction) -  sample descriptors for a subset of images

        Args:
            descinfo: The features extracted from the function `extract`.
            indexes (list[int]): List of image ids for the subset.
        """
        raise NotImplementedError

    def get_segments_folder(self, output_folder):
        """
        Return the folder path to the detected segments

        Args:
            output_folder (str): The output folder
        Returns:
            path_to_segments (str): The path to the saved segments
        """
        return os.path.join(output_folder, "segments")

    def merge_lines(self, segs):
        from limap.line2d.line_utils import merge_lines
        segs = segs[:, :4].reshape(-1, 2, 2)
        segs = merge_lines(segs)
        segs = segs.reshape(-1, 4)
        return segs

    def take_longest_k(self, segs, max_num_2d_segs=3000):
        indexes = np.arange(0, segs.shape[0])
        if max_num_2d_segs is None or max_num_2d_segs == -1:
            pass
        elif segs.shape[0] > max_num_2d_segs:
            lengths_squared = (segs[:,2] - segs[:,0]) ** 2 + (segs[:,3] - segs[:,1]) ** 2
            indexes = np.argsort(lengths_squared)[::-1][:max_num_2d_segs]
            segs = segs[indexes,:]
        return segs, indexes

    def visualize_segs(self, output_folder, imagecols, first_k=10):
        seg_folder = self.get_segments_folder(output_folder)
        n_vis_images = min(first_k, imagecols.NumImages())
        vis_folder = os.path.join(output_folder, "visualize")
        limapio.check_makedirs(vis_folder)
        image_ids = imagecols.get_img_ids()[:n_vis_images]
        for img_id in image_ids:
            img = imagecols.read_image(img_id)
            segs = limapio.read_txt_segments(seg_folder, img_id)
            img = limapvis.draw_segments(img, segs, (0, 255, 0))
            fname = os.path.join(vis_folder, "img_{0}_det.png".format(img_id))
            cv2.imwrite(fname, img)

    def detect_all_images(self, output_folder, imagecols, skip_exists=False):
        """
        Perform line detection on all images and save the line segments

        Args:
            output_folder (str): The output folder
            imagecols (:class:`limap.base.ImageCollection`): The input image collection
            skip_exists (bool): Whether to skip already processed images
        Returns:
            dict[int -> :class:`np.array`]: The line detection for each image indexed by the image id. Each segment is with shape (N, 5). Each row corresponds to x1, y1, x2, y2 and score.
        """
        seg_folder = self.get_segments_folder(output_folder)
        if not skip_exists:
            limapio.delete_folder(seg_folder)
        limapio.check_makedirs(seg_folder)
        if self.visualize:
            vis_folder = os.path.join(output_folder, "visualize")
            limapio.check_makedirs(vis_folder)
        for img_id in tqdm(imagecols.get_img_ids()):
            if skip_exists and limapio.exists_txt_segments(seg_folder, img_id):
                if self.visualize:
                    segs = limapio.read_txt_segments(seg_folder, img_id)
            else:
                segs = self.detect(imagecols.camview(img_id))
                if self.do_merge_lines:
                    segs = self.merge_lines(segs)
                segs, _ = self.take_longest_k(segs, max_num_2d_segs=self.max_num_2d_segs)
                limapio.save_txt_segments(seg_folder, img_id, segs)
            if self.visualize:
                img = imagecols.read_image(img_id)
                img = limapvis.draw_segments(img, segs, (0, 255, 0))
                fname = os.path.join(vis_folder, "img_{0}_det.png".format(img_id))
                cv2.imwrite(fname, img)
        all_2d_segs = limapio.read_all_segments_from_folder(seg_folder)
        all_2d_segs = {id: all_2d_segs[id] for id in imagecols.get_img_ids()}
        return all_2d_segs