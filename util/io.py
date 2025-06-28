import cv2 
import numpy as np 
import os
import torch

def read_image(path, grayscale=False):
    if grayscale:
        mode = cv2.IMREAD_GRAYSCALE
    else:
        mode = cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise ValueError(f'Cannot read image {path}.')
    if not grayscale and len(image.shape) == 3:
        image = image[:, :, ::-1]  # BGR to RGB
    return image


class SAVING_MAP():
    def __init__(self, save_path) -> None:
        print("[INFOR] Saving prediction 3D map")
        self.save_path = os.path.join(save_path, "Map_Prediction")
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.image_list = []
        self.idx = 0
    def save(self, output, data):
        image_name = data['imgname'][0]
        self.image_list.append(image_name)
        
        p2ds = output['keypoints'][0].detach().cpu().numpy()
        # save 2D points
        np.savetxt(os.path.join(self.save_path, str(self.idx) + "_p2d.txt"), p2ds)
        
        # save 3D points
        points3D = np.squeeze(output['points3D'].detach().cpu().numpy())
        np.savetxt(os.path.join(self.save_path, str(self.idx) + "_p3d.txt"), points3D)
        
        # save 2D lines
        l2ds = data['lines'][0].detach().cpu().numpy()
        np.savetxt(os.path.join(self.save_path, str(self.idx) + "_l2d.txt"), l2ds)

        # save 3D lines
        lines3D = np.squeeze(output['lines3D'].detach().cpu().numpy())
        np.savetxt(os.path.join(self.save_path, str(self.idx) + "_l3d.txt"), lines3D)
        
        # save prd_mask_points_coarse
        prd_mask_points_coarse = torch.sigmoid(output['prd_mask_points_coarse'][0]).detach().cpu().numpy()
        # prd_mask_points_coarse = output['prd_mask_points_coarse'][0].detach().cpu().numpy()
        np.savetxt(os.path.join(self.save_path, str(self.idx) + "_prd_mask_points_coarse.txt"), prd_mask_points_coarse)
        
        # save prd_mask_lines_coarse
        prd_mask_lines_coarse = torch.sigmoid(output['prd_mask_lines_coarse'][0]).detach().cpu().numpy()
        # prd_mask_lines_coarse = output['prd_mask_lines_coarse'][0].detach().cpu().numpy()
        np.savetxt(os.path.join(self.save_path, str(self.idx) + "_prd_mask_lines_coarse.txt"), prd_mask_lines_coarse)
        
        with open(os.path.join(self.save_path, "images.txt"), "a") as f:
            f.write(str(self.idx) + " " + image_name + "\n")
            
        self.idx += 1
        

        
    
    
    
    
    
    