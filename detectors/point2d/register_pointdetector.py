def get_pointdetector(method="superpoint", configs=dict()):
    """
    Get a point detector
    """
    if method == "superpoint":
        from .SuperPoint.superpoint import SuperPoint
        return SuperPoint(configs)
    else:
        raise NotImplementedError