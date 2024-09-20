from .linebase_detector import BaseDetectorOptions

def get_linedetector(method="lsd", max_num_2d_segs=3000,
                 do_merge_lines=False, visualize=False, weight_path=None,
                 cudaid=0):
    """
    Get a line detector
    """
    options = BaseDetectorOptions()
    options = options._replace(max_num_2d_segs=max_num_2d_segs,
        do_merge_lines=do_merge_lines, visualize=visualize, weight_path=weight_path,
        cudaid=cudaid)

    if method == "lsd":
        from .LSD.lsd import LSDDetector
        return LSDDetector(options)
    elif method == "deeplsd":
        from .DeepLSD.deeplsd import DeepLSDDetector
        return DeepLSDDetector(options)
    else:
        raise NotImplementedError