import os
import pytlsd
import numpy as np
from ..linebase_detector import LineBaseDetector, BaseDetectorOptions

class LSDDetector(LineBaseDetector):
    def __init__(self, options = BaseDetectorOptions()):
        super(LSDDetector, self).__init__(options)

    def get_module_name(self):
        return "lsd"

    def detect(self, image):
        max_n_lines = None # 80
        min_length = 15
        lines, scores, valid_lines = [], [], []
        if max_n_lines is None:
            b_segs = pytlsd.lsd(image)
        else:
            for s in [0.3, 0.4, 0.5, 0.7, 0.8, 1.0]:
                b_segs = pytlsd.lsd(image, scale=s)
                # print(len(b_segs))
                if len(b_segs) >= max_n_lines:
                    break
        # print(len(b_segs))
        segs_length = np.linalg.norm(b_segs[:, 2:4] - b_segs[:, 0:2], axis=1)
        # Remove short lines
        # b_segs = b_segs[segs_length >= min_length]
        # segs_length = segs_length[segs_length >= min_length]
        b_scores = b_segs[:, -1] * np.sqrt(segs_length)
        # Take the most relevant segments with
        indices = np.argsort(-b_scores)
        if max_n_lines is not None:
            indices = indices[:max_n_lines]
        b_segs = b_segs[indices, :]
        # print(b_segs.shape)
        # segs = pytlsd.lsd(image)
        return b_segs

