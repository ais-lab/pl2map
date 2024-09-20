import cv2
import numpy as np
import math
import random
import PIL.Image

def resize_image(image, size, interp):
    if interp.startswith('cv2_'):
        interp = getattr(cv2, 'INTER_'+interp[len('cv2_'):].upper())
        h, w = image.shape[:2]
        if interp == cv2.INTER_AREA and (w < size[0] or h < size[1]):
            interp = cv2.INTER_LINEAR
        resized = cv2.resize(image, size, interpolation=interp)
    elif interp.startswith('pil_'):
        interp = getattr(PIL.Image, interp[len('pil_'):].upper())
        resized = PIL.Image.fromarray(image.astype(np.uint8))
        resized = resized.resize(size, resample=interp)
        resized = np.asarray(resized, dtype=image.dtype)
    else:
        raise ValueError(
            f'Unknown interpolation {interp}.')
    return resized


def sample_homography(shape, cfg):
    """Sample a random valid homography.

    Computes the homography transformation between a random patch in the original image
    and a warped projection with the same image size.
    As in `tf.contrib.image.transform`, it maps the output point (warped patch) to a
    transformed input point (original patch).
    The original patch, which is initialized with a simple half-size centered crop, is
    iteratively projected, scaled, rotated and translated.

    Arguments:
        shape: A numpy array [H,W] specifying the height and width of the original image.
        perspective: A boolean that enables the perspective and affine transformations.
        scaling: A boolean that enables the random scaling of the patch.
        rotation: A boolean that enables the random rotation of the patch.
        translation: A boolean that enables the random translation of the patch.
        n_scales: The number of tentative scales that are sampled when scaling.
        n_angles: The number of tentatives angles that are sampled when rotating.
        scaling_amplitude: Controls the amount of scale.
        perspective_amplitude_x: Controls the perspective effect in x direction.
        perspective_amplitude_y: Controls the perspective effect in y direction.
        patch_ratio: Controls the size of the patches used to create the homography.
        max_angle: Maximum angle used in rotations.
        allow_artifacts: A boolean that enables artifacts when applying the homography.
        translation_overflow: Amount of border artifacts caused by translation.

    Returns:
        A numpy of shape 3x3 corresponding to the homography transform.
    """
    shift=0
    perspective=cfg.perspective
    scaling=cfg.scaling
    rotation=cfg.rotation
    translation=cfg.translation
    n_scales=cfg.n_scales
    n_angles=cfg.n_angles
    scaling_amplitude=cfg.scaling_amplitude
    perspective_amplitude_x=cfg.perspective_amplitude_x
    perspective_amplitude_y=cfg.perspective_amplitude_y
    patch_ratio=cfg.patch_ratio
    max_angle=math.pi*(cfg.max_angle/180)
    allow_artifacts=cfg.allow_artifacts
    translation_overflow=0.

    # Corners of the output image
    pts1 = np.stack([[0., 0.], [0., 1.], [1., 1.], [1., 0.]], axis=0)
    # Corners of the input patch
    margin = (1 - patch_ratio) / 2
    pts2 = margin + np.array([[0, 0], [0, patch_ratio],
                                 [patch_ratio, patch_ratio], [patch_ratio, 0]])

    from numpy.random import normal
    from numpy.random import uniform
    from scipy.stats import truncnorm

    # Random perspective and affine perturbations
    # lower, upper = 0, 2
    std_trunc = 2

    if perspective:
        if not allow_artifacts:
            perspective_amplitude_x = min(perspective_amplitude_x, margin)
            perspective_amplitude_y = min(perspective_amplitude_y, margin)

        perspective_displacement = truncnorm(-1*std_trunc, std_trunc, loc=0, scale=perspective_amplitude_y/2).rvs(1)
        h_displacement_left = truncnorm(-1*std_trunc, std_trunc, loc=0, scale=perspective_amplitude_x/2).rvs(1)
        h_displacement_right = truncnorm(-1*std_trunc, std_trunc, loc=0, scale=perspective_amplitude_x/2).rvs(1)
        pts2 += np.array([[h_displacement_left, perspective_displacement],
                          [h_displacement_left, -perspective_displacement],
                          [h_displacement_right, perspective_displacement],
                          [h_displacement_right, -perspective_displacement]]).squeeze()

    # Random scaling
    # sample several scales, check collision with borders, randomly pick a valid one
    if scaling:
        scales = truncnorm(-1*std_trunc, std_trunc, loc=1, scale=scaling_amplitude/2).rvs(n_scales)
        scales = np.concatenate((np.array([1]), scales), axis=0)

        center = np.mean(pts2, axis=0, keepdims=True)
        scaled = (pts2 - center)[np.newaxis, :, :] * scales[:, np.newaxis, np.newaxis] + center
        if allow_artifacts:
            valid = np.arange(n_scales)  # all scales are valid except scale=1
        else:
            valid = (scaled >= 0.) * (scaled < 1.)
            valid = valid.prod(axis=1).prod(axis=1)
            valid = np.where(valid)[0]
        idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
        pts2 = scaled[idx,:,:]

    # Random translation
    if translation:
        t_min, t_max = np.min(pts2, axis=0), np.min(1 - pts2, axis=0)
        if allow_artifacts:
            t_min += translation_overflow
            t_max += translation_overflow
        pts2 += np.array([uniform(-t_min[0], t_max[0],1), uniform(-t_min[1], t_max[1], 1)]).T

    # Random rotation
    # sample several rotations, check collision with borders, randomly pick a valid one
    if rotation:
        angles = np.linspace(-max_angle, max_angle, num=n_angles)
        angles = np.concatenate((angles, np.array([0.])), axis=0)  # in case no rotation is valid
        center = np.mean(pts2, axis=0, keepdims=True)
        rot_mat = np.reshape(np.stack([np.cos(angles), -np.sin(angles), np.sin(angles),
                                       np.cos(angles)], axis=1), [-1, 2, 2])
        rotated = np.matmul( (pts2 - center)[np.newaxis,:,:], rot_mat) + center
        if allow_artifacts:
            valid = np.arange(n_angles)  # all scales are valid except scale=1
        else:
            valid = (rotated >= 0.) * (rotated < 1.)
            valid = valid.prod(axis=1).prod(axis=1)
            valid = np.where(valid)[0]
        idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
        pts2 = rotated[idx,:,:]


    # Rescale to actual size
    shape = shape[::-1]  # different convention [y, x]
    pts1 *= shape[np.newaxis,:]
    pts2 *= shape[np.newaxis,:]

    def ax(p, q): return [p[0], p[1], 1, 0, 0, 0, -p[0] * q[0], -p[1] * q[0]]
    def ay(p, q): return [0, 0, 0, p[0], p[1], 1, -p[0] * q[1], -p[1] * q[1]]

    homography = cv2.getPerspectiveTransform(np.float32(pts1+shift), np.float32(pts2+shift))
    return homography

def warpPerspective_forimage(ori_img, h_matrix):
    # Apply the homography transformation to the image
    transformed_image = cv2.warpPerspective(ori_img, h_matrix, (ori_img.shape[1], ori_img.shape[0]))
    return transformed_image

def perspectiveTransform_forpoints(positions, h_matrix):
    # Apply the homography transformation to the list of positions
    transformed_positions = cv2.perspectiveTransform(np.array([positions]), h_matrix)
    return transformed_positions[0,:,:]

def perspectiveTransform_forlines(lines, h_matrix):
    # Apply the homography transformation to the list of 2D lines
    start_points = lines[:,:2]  
    end_points = lines[:,2:]
    transformed_start_points = cv2.perspectiveTransform(np.array([start_points]), h_matrix)[0,:,:]
    transformed_end_points = cv2.perspectiveTransform(np.array([end_points]), h_matrix)[0,:,:]
    transformed_lines = np.concatenate((transformed_start_points, transformed_end_points), axis=1)
    return transformed_lines

def random_brightness_contrast(image, b_rate, c_rate):
    # Random the brightness and contrast values
    contrast = [1.0, 1.0+2.0*c_rate]
    brightness = [-100*b_rate, 100*b_rate]
    alpha = random.uniform(contrast[0], contrast[1])
    beta = random.uniform(brightness[0], brightness[1])
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def is_apply_augment(rate):
    '''
    Return True if the augmentation is applied by select a random option
    '''
    # Define the options and their probabilities
    options = [True, False]
    probabilities = [rate, 1-rate] 
    # Choose an option to turn on augmentation or not
    return random.choices(options, weights=probabilities, k=1)[0]

def dsacstar_augmentation(image, cfg, points2d, lines2d, camera, pose, interpolation='cv2_area'):
    '''
    Apply the augmentation to the input image, points, lines, camera, and pose
    args:
        image: input image np.array WxH 
        cfg: configuration file .yaml
        points2d: 2D points np.array Nx2
        lines2d: 2D lines np.array Nx4
        camera: camera parameters np.array[w, h, f, cx, cy,...]
        pose: camera pose 'class _base.Pose'
    '''
    # Random the scale factor and rotation angle
    scale_factor = random.uniform(cfg.aug_scale_min, cfg.aug_scale_max)
    angle = random.uniform(-cfg.aug_rotation, cfg.aug_rotation)

    # Apply the scale factor and rotation angle to the image
    new_shape = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor)) # height, width 
    image = resize_image(image, new_shape, interpolation)

    # ajust the points and lines coordinates
    points2d = points2d * scale_factor
    lines2d = lines2d * scale_factor
    
    # ajust the camera parameters
    camera.update_scale(scale_factor)

    # rotate input image
    # Get the rotation matrix
    M = cv2.getRotationMatrix2D((new_shape[0] / 2, new_shape[1] / 2), angle, 1)
    # Rotate the image
    image = cv2.warpAffine(image, M, new_shape)
    points2d = rotate_points_dsacstar(points2d, M)
    lines2d = rotate_lines_dsacstar(lines2d, M)
    # rotate ground truth camera pose
    pose.rotate(angle)
    
    return image, points2d, lines2d, camera, pose

def rotate_points_dsacstar(points, M):
    # Convert the points to homogeneous coordinates
    points_hom = np.hstack((points, np.ones((points.shape[0], 1))))
    # Rotate the points
    rotated_points_hom = np.dot(M, points_hom.T).T
    # Convert the points back to 2D
    rotated_points = rotated_points_hom[:, :2]
    return rotated_points

def rotate_lines_dsacstar(lines, M):
    start_points = lines[:,:2]  
    end_points = lines[:,2:]
    start_points = rotate_points_dsacstar(start_points, M)
    end_points = rotate_points_dsacstar(end_points, M)
    rotated_lines = np.concatenate((start_points, end_points), axis=1)
    return rotated_lines

def is_inside_img(points, img_shape):
    h, w = img_shape[0], img_shape[1]
    return (points[:, 0] >= 0) & (points[:, 0] < w) & (points[:, 1] >= 0) & (points[:, 1] < h)

def correct_points_lines_inside_image(shape, image_infor_class):
    '''
    Correct the points and lines coordinates to be inside the image
    if the points/lines are outside the image, remove them 
    if lines have half inside and half outside, shrink the line to be inside the image
    Then, correct the 3D ground truth points coordinates
    Args:
        shape: image shape (height, width)
        image_infor_class: class _base.ImageInfor
    '''
    H, W = shape[0], shape[1]
    # correct 2d points
    points2d = image_infor_class.points2Ds
    valid_points = is_inside_img(points2d, shape)
    points2d = points2d[valid_points]
    image_infor_class.points2Ds = points2d
    # correct 3d points
    image_infor_class.points3Ds = image_infor_class.points3Ds[valid_points]
    # correct id of valids 
    image_infor_class.validPoints = image_infor_class.validPoints[valid_points]
    assert len(image_infor_class.points2Ds) == len(image_infor_class.points3Ds) == len(image_infor_class.validPoints)

    # correct 2d lines
    lines2d = image_infor_class.line2Ds
    lines3d = image_infor_class.line3Ds_matrix 
    valids_lines2d = image_infor_class.validLines
    start_points = lines2d[:,:2]
    end_points = lines2d[:,2:]
    valid_start_points = is_inside_img(start_points, shape)
    valid_end_points = is_inside_img(end_points, shape)
    # remove lines that are outside the image
    valid_lines = valid_start_points | valid_end_points

    start_points = start_points[valid_lines]
    end_points = end_points[valid_lines]
    lines3d = lines3d[valid_lines]
    valids_lines2d = valids_lines2d[valid_lines]

    valid_start_points = valid_start_points[valid_lines]
    valid_end_points = valid_end_points[valid_lines]
    # shrink lines that are half inside and half outside the image
    indices = np.where(~valid_start_points)[0]
    for idx in indices:
        start = start_points[idx,:] # outside points 
        end = end_points[idx,:]
        m, c = line_equation(start, end) # y = mx + c
        if start[0] < 0:
            start[0] = 0+1
            start[1] = compute_y(m, c, 0)
        elif start[0] > W:
            start[0] = W - 1 
            start[1] = compute_y(m, c, W)
        if start[1] < 0:
            start[0] = compute_x(m, c, 0)
            start[1] = 0 + 1
        elif start[1] > H:
            start[0] = compute_x(m, c, H)
            start[1] = H - 1
        start_points[idx] = start 
    indices = np.where(~valid_end_points)[0]
    for idx in indices:
        start = start_points[idx] 
        end = end_points[idx] # outside points 
        m, c = line_equation(start, end) # y = mx + c
        if end[0] < 0:
            end[0] = 0+1
            end[1] = compute_y(m, c, 0)
        elif end[0] > W:
            end[0] = W - 1 
            end[1] = compute_y(m, c, W)
        if end[1] < 0:
            end[0] = compute_x(m, c, 0)
            end[1] = 0 + 1
        elif end[1] > H:
            end[0] = compute_x(m, c, H)
            end[1] = H - 1
        end_points[idx] = end

    assert np.all(is_inside_img(start_points, shape))
    assert np.all(is_inside_img(end_points, shape))
    lines2d = np.concatenate((start_points, end_points), axis=1)
    assert len(lines2d) == len(lines3d) == len(valids_lines2d)
    image_infor_class.line2Ds  = lines2d
    image_infor_class.line3Ds_matrix = lines3d
    image_infor_class.validLines = valids_lines2d
    return image_infor_class

def line_equation(start, end):
    # Calculate the slope
    m = (end[1] - start[1]) / (end[0] - start[0])
    # Calculate the y-intercept
    c = start[1] - m * start[0]
    return m, c # y = mx + c
def compute_x(m, c, y):
    # Calculate the x value that corresponds to the given y value
    # and the line equation y = mx + c
    return (y - c) / m
def compute_y(m, c, x):
    # Calculate the y value that corresponds to the given x value
    # and the line equation y = mx + c
    return m * x + c