import numpy as np
import open3d as o3d
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets._base import Line3D
from util.io import read_image
import matplotlib.pyplot as plt

def test_point_inside_ranges(point, ranges):
    point = np.array(point)
    if ~np.all(point > ranges[0]) or ~np.all(point < ranges[1]):
        return False
    return True

def test_line_inside_ranges(line, ranges):
    if not test_point_inside_ranges(line.start, ranges):
        return False
    if not test_point_inside_ranges(line.end, ranges):
        return False
    return True

def open3d_get_line_set(lines, color=[0.0, 0.0, 0.0], width=2, ranges=None, scale=1.0):
    """
    convert a list of line3D objects to an Open3D lines set
    Args:
        lines (list[:class:`datasets._base.Line3D`] or numpy array of Nx6): The 3D line map
        color (list[float]): The color of the lines
        width (float, optional): width of the line
    """
    o3d_points, o3d_lines, o3d_colors = [], [], []
    counter = 0
    for line in lines:
        if isinstance(line, np.ndarray):
            line = Line3D(line[:3], line[3:])
        if ranges is not None:
            if not test_line_inside_ranges(line, ranges):
                continue
        o3d_points.append(line.start * scale)
        o3d_points.append(line.end * scale)
        o3d_lines.append([2*counter, 2*counter+1])
        counter += 1
        o3d_colors.append(color)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(o3d_points)
    line_set.lines = o3d.utility.Vector2iVector(o3d_lines)
    line_set.colors = o3d.utility.Vector3dVector(o3d_colors)
    return line_set


def open3d_vis_3d_lines_with_hightlightFrame(lines3D, hightlight_lines3D, width=2, ranges=None, scale=1.0):
    """
    Save 3D line map with `Open3D <http://www.open3d.org/>`_

    Args:
        lines3D: numpy array of Nx6
        hightlight_lines3D: numpy array of Nx6
        width (float, optional): width of the line
    """

    line_set = open3d_get_line_set(lines3D, width=width, ranges=ranges, scale=scale)
    line_set_highlight = open3d_get_line_set(hightlight_lines3D, color=[0.0, 1.0, 0.0], width=width*2, ranges=ranges, scale=scale)
    
    # Save the line_set
    o3d.io.write_line_set("visualization/line_set.ply", line_set)
    o3d.io.write_line_set("visualization/line_set_highlight.ply", line_set_highlight)
    
    '''
    vis = o3d.visualization.Visualizer()
    vis.create_window(height=1080, width=1920)
    vis.add_geometry(line_set)
    vis.add_geometry(line_set_highlight)
    vis.run()
    vis.destroy_window()
    '''
        
    


def open3d_vis_3d_lines(lines3D, cameras=None, poses=None, gt_pose=None, width=2, ranges=None, scale=1.0):
    """
    Visualize a 3D line map with `Open3D <http://www.open3d.org/>`_

    Args:
        lines (list[:class:`datasets._base.Line3D` and/or None]): The 3D line map
        width (float, optional): width of the line
    """
    if isinstance(lines3D, list):
        lines = []
        for line in lines3D:
            if line is not None:
                lines.append(line)
    elif isinstance(lines3D, np.ndarray):
        lines = lines3D
    else:
        raise ValueError("lines3D must be either a list or a numpy array")

    vis = o3d.visualization.Visualizer()
    vis.create_window(height=1080, width=1920)
    prune = len(lines)
    # prune = int(0.8*len(lines))
    line_set = open3d_get_line_set(lines[:prune,:], width=width, ranges=ranges, scale=scale)
    o3d.io.write_line_set("visualization/line_set.ply", line_set)
    # return
    vis.add_geometry(line_set)
    if poses is not None:
        assert cameras is not None
        assert gt_pose is not None
        def get_t(pose):
            R = qvec2rotmat(pose[3:])
            # translation
            t = pose[:3]
            # invert
            t = -R.T @ t
            return t
        connect_poses_lines = []
        is_draws = []
        for i in range(len(poses)):
            est_t = get_t(poses[i])
            gt_t = get_t(gt_pose[i])
            # calculate distance between two points
            is_draw = True if np.linalg.norm(est_t - gt_t) < 100 else False
            is_draws.append(is_draw)
            if is_draw:
                tmp_line = np.array([est_t[0], est_t[1], est_t[2], gt_t[0], gt_t[1], gt_t[2]])
                connect_poses_lines.append(tmp_line)
        connect_line_set = open3d_get_line_set(connect_poses_lines, width=width, ranges=ranges, scale=scale, color=[0,1,0])
        vis.add_geometry(connect_line_set)
        i = 0 
        for pose, camera in zip(poses, cameras):
            if is_draws[i]: add_camera(vis, pose, camera, scale=0.2, gt = False)
            i+=1
        i = 0 
        for pose, camera in zip(gt_pose, cameras):
            if is_draws[i]: add_camera(vis, pose, camera, scale=0.2, gt = True)
            i+=1
    
    vis.run()
    vis.destroy_window()

def open3d_vis_3d_lines_from_datacollection(datacollection, train_or_test="train"):
    '''
    Visualize 3D lines from datasetcollection
    Args:
        datacollection: DataCollection object
        train_or_test: string, "train" or "test"'''
    if train_or_test != "train":
        raise ValueError("Currently only support 'train' mode.")
    vis_lines = []
    imgs_list = datacollection.train_imgs if train_or_test=="train" else datacollection.test_imgs
    import random
    random.shuffle(imgs_list)
    cameras = []
    poses = []
    i = 0
    for img in imgs_list:
        vis_lines += datacollection.imgname2imgclass[img].line3Ds
        poses.append(datacollection.imgname2imgclass[img].pose.get_pose_vector())
        cameras.append(datacollection.imgname2imgclass[img].camera.get_dict_camera())
        # i += 1
        # if i > 20:
        #     break
    open3d_vis_3d_lines(vis_lines, cameras=cameras, poses=poses)

def open3d_vis_3d_lines_from_single_imgandcollection(datacollection, img_name):
    '''
    Visualize 3D lines from datasetcollection
    Args:
        datacollection: DataCollection object
        img_name: string, image name
    '''
    if img_name in datacollection.test_imgs:
        raise ValueError("Only train images have 3D labeled lines.")
    vis_lines = datacollection.imgname2imgclass[img_name].line3Ds
    open3d_vis_3d_lines(vis_lines)

def visualize_2d_lines(img_path, savename,lines2D, lines3D, save_path="visualization/"):
    """ Plot lines for existing images.
    Args:
        img_path: string, path to the image.
        lines2D: list of ndarrays of size (N, 4).
        lines3D: list of objects with size of (N, 1).
        save_path: string, path to save the image.
    """
    save_path = os.path.join(save_path,savename)
    img = read_image(img_path)
    plt.figure()
    plt.imshow(img)
    length = lines2D.shape[0]
    for i in range(length):
        k = lines2D[i,:]
        x = [k[0], k[2]]
        y = [k[1], k[3]]
        if lines3D is not None:
            c = 'lime' if lines3D[i] is None else 'red'
        else:
            c = 'lime'
        plt.plot(x, y, color=c)
    plt.savefig(save_path)
    # Close the figure to free up memory
    plt.close()

def visualize_2d_lines_from_collection(datacollection, img_name, mode="offline"):
    """
    Visualize 2D lines from datasetcollection
    Args:
        datacollection: DataCollection object
        img_name: string, image name
        mode: string, "offline" (take from exiting labels) or "online" (use detector model to get 2D points)
    """
    if mode == "offline":
        line2Ds = datacollection.imgname2imgclass[img_name].line2Ds
        line3Ds = datacollection.imgname2imgclass[img_name].line3Ds
    elif mode == "online":
        line2Ds = datacollection.detect_lines2D(img_name)
        line3Ds = None
    else:
        raise ValueError("mode must be either 'offline' or 'online'")
    img_path = datacollection.get_image_path(img_name)
    save_name = img_name.replace("/","_") + "_lines_" + mode +".png"
    visualize_2d_lines(img_path, save_name, line2Ds, line3Ds)

# -------------------------------- end line visualization --------------------------------
##########################################################################################
# -------------------------------- start point visualization -----------------------------


def visualize_2d_points(img_path, points2D, savename, colors='lime', ps=4, save_path="visualization/"):
    """Plot keypoints for existing images.
    Args:
        img_path: string, path to the image.
        points2D: list of ndarrays of size (N, 2).
        colors: string, or list of list of tuples (one for each keypoints).
        ps: size of the keypoints as float.
        save_path: string, path to save the image.
    """
    save_path = os.path.join(save_path,savename)
    img = read_image(img_path)
    plt.figure()
    plt.imshow(img)
    if not isinstance(colors, list):
        colors = [colors] * len(points2D)
    for k, c in zip(points2D, colors):
        plt.scatter(k[0], k[1], c=c, s=ps, linewidths=0)
    plt.savefig(save_path)
    # Close the figure to free up memory
    plt.close()

def visualize_2d_points_from_collection(datacollection, img_name, mode="offline"):
    """
    Visualize 2D points from datasetcollection
    Args:
        datacollection: DataCollection object
        img_name: string, image name
        mode: string, "offline" (take from exiting labels) or "online" (use detector model to get 2D points)
    """
    if mode == "offline":
        if img_name in datacollection.test_imgs:
            raise ValueError("Only train images have 2D labeled points.")
        points2D = datacollection.imgname2imgclass[img_name].points2Ds
    elif mode == "online":
        data = datacollection.detect_points2D(img_name)
        points2D = data["keypoints"][0].detach().cpu().numpy()
    else:
        raise ValueError("mode must be either 'offline' or 'online'")
    img_path = datacollection.get_image_path(img_name)
    save_name = img_name.replace("/","_") + "_points_" + mode +".png"
    visualize_2d_points(img_path, points2D, save_name)

def open3d_get_point_set(points, color=[0.0, 0.0, 0.0], width=2, scale=1.0):
    """
    convert a numpy array of points3D to an Open3D lines set
    Args:
        points (numpy array of Nx3): The 3D point map
        color (list[float]): The color of the lines
        width (float, optional): width of the line
    """
    o3d_points, o3d_colors = [], []
    for point in points:
        if np.sum(point) == 0:
            continue
        o3d_points.append(point)
        o3d_colors.append(color)
    point_set = o3d.geometry.PointCloud()
    point_set.points = o3d.utility.Vector3dVector(o3d_points)
    point_set.colors = o3d.utility.Vector3dVector(o3d_colors)
    return point_set

def open3d_vis_3d_points(points3D:np.asanyarray, width=2, ranges=None, scale=1.0):
    """
    Visualize a 3D point map with `Open3D <http://www.open3d.org/>`_

    Args:
        points3D (list[:class:`datasets._base.Line3D` and/or None]): The 3D line map
        width (float, optional): width of the line
    """

    vis = o3d.visualization.Visualizer()
    vis.create_window(height=1080, width=1920)
    point_set = open3d_get_point_set(points3D, width=width, scale=scale)
    o3d.io.write_point_cloud("visualization/point_set.ply", point_set)
    # return
    vis.add_geometry(point_set)
    vis.run()
    vis.destroy_window()

def open3d_vis_3d_points_from_datacollection(datacollection, mode="train"):
    '''
    Visualize 3D points from datasetcollection
    Args:
        datacollection: DataCollection object
        mode: string, "train" or "test"
    '''
    if mode != "train":
        raise ValueError("Currently only support 'train' mode.")
    vis_points = np.array([[0,0,0]])
    imgs_list = datacollection.train_imgs if mode=="train" else datacollection.test_imgs
    for img in imgs_list:
        vis_points = np.concatenate((vis_points, datacollection.imgname2imgclass[img].points3Ds))
    open3d_vis_3d_points(vis_points)


def open3d_vis_3d_points_with_hightlightFrame(points3D, hightlight_points3D, width=2, ranges=None, scale=1.0):
    """
    Save 3D point map with `Open3D <http://www.open3d.org/>`_

    Args:
        points3D (list[:class:`datasets._base.Line3D` and/or None]): The 3D line map
        width (float, optional): width of the line
    """

    point_set = open3d_get_point_set(points3D, width=width, scale=scale)
    highlight_point_set = open3d_get_point_set(hightlight_points3D, color=[0.0, 1.0, 0.0], width=width*2, scale=scale)
    
    # save the point_set
    o3d.io.write_point_cloud("visualization/point_set.ply", point_set)
    o3d.io.write_point_cloud("visualization/highlight_point_set.ply", highlight_point_set)
    '''
    vis = o3d.visualization.Visualizer()
    vis.create_window(height=1080, width=1920)
    vis.add_geometry(point_set)
    vis.add_geometry(highlight_point_set)
    vis.run()
    vis.destroy_window()
    '''


##########################################################################################
# -------------------- merging points and lines for visualization together --------------
def visualize_2d_points_lines(img_path, points2D, lines2D, lines3D, savename,
                              colors='lime', ps=4, save_path="visualization/"):
    """Plot keypoints for existing images.
    Args:
        img_path: string, path to the image.
        points2D: list of ndarrays of size (N, 2).
        lines2D: list of ndarrays of size (N, 4).
        lines3D: list of objects with size of (N, 1).
        colors: string, or list of list of tuples (one for each keypoints).
        ps: size of the keypoints as float.
        save_path: string, path to save the image.
    """
    save_path = os.path.join(save_path,savename)
    img = read_image(img_path)
    plt.figure()
    plt.imshow(img)
    if not isinstance(colors, list):
        colors = [colors] * len(points2D)
    # visualize points
    for k, c in zip(points2D, colors):
        plt.scatter(k[0], k[1], c=c, s=ps, linewidths=0)
        
    # visualize lines
    length = lines2D.shape[0]
    for i in range(length):
        k = lines2D[i,:]
        x = [k[0], k[2]]
        y = [k[1], k[3]]
        if lines3D is not None:
            c = 'lime' if lines3D[i] is None else 'red'
        else:
            c = 'lime'
        plt.plot(x, y, color=c)
    plt.savefig(save_path)
    # Close the figure to free up memory
    plt.close()


def visualize_2d_points_lines_from_collection(datacollection, img_name, mode="offline"):
    """
    Visualize 2D points and lines from datasetcollection
    Args:
        datacollection: DataCollection object
        img_name: string, image name
        mode: string, "offline" (take from exiting labels) or "online" (use detector model to get 2D points)
    """
    if mode == "offline":
        if img_name in datacollection.test_imgs:
            raise ValueError("Only train images have 2D labeled points.")
        points2D = datacollection.imgname2imgclass[img_name].points2Ds
        
        line2Ds = datacollection.imgname2imgclass[img_name].line2Ds
        line3Ds = datacollection.imgname2imgclass[img_name].line3Ds
        
    elif mode == "online":
        data = datacollection.detect_points2D(img_name)
        points2D = data["keypoints"][0].detach().cpu().numpy()
        
        line2Ds = datacollection.detect_lines2D(img_name)
        line3Ds = None
    else:
        raise ValueError("mode must be either 'offline' or 'online'")
    img_path = datacollection.get_image_path(img_name)
    save_name = img_name.replace("/","_") + "_points_lines_" + mode +".svg"
    
    visualize_2d_points_lines(img_path, points2D, line2Ds, line3Ds, save_name)


##########################################################################################
# -------------------------------- Augmentation Visualization Debug ----------------------

import cv2

def visualize_img_withlinesandpoints(image, points, lines, augmented=False):
    
    save_path = "visualization/"
    point_size = 1

    # Draw the original positions on the original image
    for position in points:
        cv2.circle(image, (int(position[0]), int(position[1])), point_size, (0, 0, 255), -1)
    # Draw the original lines on the original image
    for line in lines:
        cv2.line(image, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), (255, 0, 0), 1)
        cv2.circle(image, (int(line[0]), int(line[1])), point_size*3, (0, 0, 255), -1)
        cv2.circle(image, (int(line[2]), int(line[3])), point_size*3, (0, 0, 255), -1)

    if augmented:
        cv2.imwrite(save_path+'Transformed_Image.jpg', image)
    else:
        cv2.imwrite(save_path+'Original_Image.jpg', image)
        
##########################################################################################
# -------------------------------- draw camera poses -------------------------------------
from util.read_write_model import qvec2rotmat
def draw_camera(K, R, t, w, h,
                scale=1, color=[1, 0, 0]):
    """Create axis, plane and pyramed geometries in Open3D format.
    :param K: calibration matrix (camera intrinsics)
    :param R: rotation matrix
    :param t: translation
    :param w: image width
    :param h: image height
    :param scale: camera model scale
    :param color: color of the image plane and pyramid lines
    :return: camera model geometries (axis, plane and pyramid)
    """

    # intrinsics
    K = K.copy() / scale
    Kinv = np.linalg.inv(K)

    # 4x4 transformation
    T = np.column_stack((R, t))
    T = np.vstack((T, (0, 0, 0, 1)))

    # axis
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5 * scale)
    axis.transform(T)

    # points in pixel
    points_pixel = [
        [0, 0, 0],
        [0, 0, 1],
        [w, 0, 1],
        [0, h, 1],
        [w, h, 1],
    ]

    # pixel to camera coordinate system
    points = [Kinv @ p for p in points_pixel]

    # image plane
    width = abs(points[1][0]) + abs(points[3][0])
    height = abs(points[1][1]) + abs(points[3][1])
    plane = o3d.geometry.TriangleMesh.create_box(width, height, depth=1e-6)
    # plane.paint_uniform_color([0.5,0,0])
    # plane.paint_uniform_color(color)
    plane.translate([points[1][0], points[1][1], scale])
    plane.transform(T)

    # pyramid
    points_in_world = [(R @ p + t) for p in points]
    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [1, 2], 
        [2, 4],
        [4, 3],
        [3, 1],
    ]
    colors = [color for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points_in_world),
        lines=o3d.utility.Vector2iVector(lines))
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # return as list in Open3D format
    # return [axis, plane, line_set]
    # return [plane, line_set]
    return [line_set]



def add_camera(vis, pose, camera, scale=0.1, gt = False, othermethod = False):
        plane_scale = 1
        # rotation
        R = qvec2rotmat(pose[3:])
        # translation
        t = pose[:3]
        # invert
        t = -R.T @ t
        R = R.T
        # intrinsics

        if camera['model'] in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            fx = fy = camera['params'][0]
            cx = camera['params'][1]
            cy = camera['params'][2]
        elif camera['model'] in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"):
            fx = camera['params'][0]
            fy = camera['params'][1]
            cx = camera['params'][2]
            cy = camera['params'][3]
        else:
            raise Exception("Camera model not supported")

        # intrinsics
        K = np.identity(3)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy
        if othermethod:
            color = [0,1,0]
        else:
            color = [1, 0, 0] if gt else [0, 0, 1]
        # create axis, plane and pyramed geometries that will be drawn
        cam_model = draw_camera(K, R, t, camera['width']*plane_scale, camera['height']*plane_scale, scale, color)
        for i in cam_model:
            vis.add_geometry(i)
    
