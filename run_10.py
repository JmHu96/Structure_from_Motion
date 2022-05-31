from pathlib import Path
import json
import numpy as np
from numpy.linalg import inv, svd, det, norm
import matplotlib.pyplot as plt
from types import SimpleNamespace
from PIL import Image

import open3d as o3d
from vis import vis_3d, o3d_pc, draw_camera
from mpl_interactive import Visualizer as TwoViewVis

from sfm_10 import (
    t_and_R_from_pose_pair,
    essential_from_t_and_R,
    F_from_K_and_E,
    E_from_K_and_F,
    t_and_R_from_essential,
    disambiguate_four_chirality_by_triangulation,
    triangulate,
    triangulate_multi_camera,
    normalized_eight_point_algorithm,
    eight_point_algorithm,
    bundle_adjustment,
    align_B_to_A,
    as_cross
)

from render import (
    compute_intrinsics, compute_extrinsics, as_homogeneous, homogenize
)

DATA_ROOT = Path("./data")


def load_sfm_data():
    root = DATA_ROOT / "sfm"
    poses = np.load(root / "camera_poses.npy")
    visibility = np.load(root / "multiview_visibility.npy")
    #print(visibility[0,:])
    pts_3d = np.load(root / "pts_3d.npy")
    with open(root / "camera_intrinsics.json", "r") as f:
        intr = json.load(f)
    img_w, img_h, fov = intr['img_w'], intr['img_h'], intr['vertical_fov']

    data = SimpleNamespace(
        img_w=img_w, img_h=img_h, fov=fov,
        poses=poses, visibility=visibility, pts_3d=pts_3d
    )
    return data


def read_view_image(i):
    fname = DATA_ROOT / "sfm" / f"view_{i}.png"
    img = np.array(Image.open(fname))
    return img


def common_visible_points(data, view_indices):
    """view_indices: a list of view indices e.g. [0, 1, 4]
    """
    mask = [] # your code here: use data.visibility
    for i in range(data.visibility.shape[1]):
        if(np.sum(data.visibility[view_indices,i]) >= 2):
            mask.append(i)
    common_pts = data.pts_3d[mask]
    return common_pts


def project(pts_3d, K, pose):
    P = K @ inv(pose)
    x1 = homogenize(pts_3d @ P.T)
    return x1


class Engine():
    def __init__(self):
        data = load_sfm_data()
        self.view1, self.view2, self.view3, self.view4, self.view5 = 0, 1, 2, 3, 4
        self.view6, self.view7, self.view8, self.view9, self.view10 = 5, 6, 7, 8, 9
        pose1,pose2,pose3,pose4,pose5 = data.poses[self.view1],data.poses[self.view2],data.poses[self.view3],data.poses[self.view4],data.poses[self.view5]
        pose6,pose7,pose8,pose9,pose10 = data.poses[self.view6],data.poses[self.view7],data.poses[self.view8],data.poses[self.view9],data.poses[self.view10]

        visibility_mat_pre = data.visibility
        visibility_mat = np.array([])
        index_valid = []
        for i in range(visibility_mat_pre.shape[1]):
            if np.sum(visibility_mat_pre[:,i]) >= 2:
                visibility_mat = np.concatenate((visibility_mat,visibility_mat_pre[:,i]))
                index_valid.append(i)
    
        visibility_mat = np.reshape(visibility_mat,(10,-1),order='F')
        self.visivility_mat = visibility_mat

        self.pts_3d = data.pts_3d[index_valid,:]

        pts_3d = common_visible_points(data, [self.view1, self.view2, self.view3, self.view4, self.view5, self.view6, self.view7, self.view8, self.view9, self.view10])

        img_w, img_h = data.img_w, data.img_h
        fov = data.fov
        K = compute_intrinsics(img_w / img_h, fov, img_h)

        self.data = data
        self.K = K
        self.pose1,self.pose2,self.pose3,self.pose4,self.pose5,self.pose6,self.pose7,self.pose8,self.pose9,self.pose10 = pose1,pose2,pose3,pose4,pose5,pose6,pose7,pose8,pose9,pose10
        
        self.pts_3d = pts_3d
        self.img_w, self.img_h = img_w, img_h
        

    def q10(self):
        self.sfm_pipeline(draw_config=True)

    def q12(self):
        self.sfm_pipeline(use_noise=True, use_BA=True, final_vis=True)

    def sfm_pipeline(self, use_noise=False, use_BA=False, draw_config=False, final_vis=True):     
        pts_3d = self.pts_3d
        K = self.K
        img_w, img_h = self.img_w, self.img_h
        pose1, pose2 = self.pose1, self.pose2
        pose3, pose4, pose5, pose6, pose7, pose8, pose9, pose10 = self.pose3, self.pose4, self.pose5, self.pose6, self.pose7, self.pose8, self.pose9, self.pose10
        visibility_mat = self.visivility_mat

        x1s = project(pts_3d, K, pose1)
        xs = np.zeros((x1s.shape[0],x1s.shape[1],10))
        xs[...,0] = x1s
        xs[...,1] = project(pts_3d, K, pose2)
        xs[...,2] = project(pts_3d, K, pose3)
        xs[...,3] = project(pts_3d, K, pose4)
        xs[...,4] = project(pts_3d, K, pose5)
        xs[...,5] = project(pts_3d, K, pose6)
        xs[...,6] = project(pts_3d, K, pose7)
        xs[...,7] = project(pts_3d, K, pose8)
        xs[...,8]= project(pts_3d, K, pose9)
        xs[...,9]= project(pts_3d, K, pose10)

        K = K[:3,:3]

        T = np.array([[2/img_w, 0, -1],
                      [0, 2/img_h, -1],
                      [0,0,1]])

        Fs = np.zeros((3,3,9,10))

        for i in range(10):
            for j in range(9):
                if j != i:
                    visibility_mat_temp = visibility_mat[[i,j],:]
                    index = []
                    for k in range(visibility_mat_temp.shape[1]):
                        if np.sum(visibility_mat_temp[:,k]) == 2:
                            index.append(k)
                    x1s = xs[index,:,i]
                    x2s = xs[index,:,j]
                    Fs[:,:,j,i] = T.T@normalized_eight_point_algorithm(x1s,x2s,img_w,img_h,T)@T

        Es = np.zeros_like(Fs)
        for i in range(10):
            for j in range(9):
                Es[:,:,j,i] = K.T @ Fs[:,:,j,i] @ K

        p = np.zeros((4,4,10,10))
        for i in range(10):
            for j in range(10):
                if i == j:
                    p[:,:,j,i] = np.eye(4)
                else:
                    if j < i:
                        jj = j
                    else:
                        jj = j - 1
                    four = t_and_R_from_essential(Es[:,:,jj,i])
                    visibility_mat_temp = visibility_mat[[i,j],:]
                    index = []
                    for k in range(visibility_mat_temp.shape[1]):
                        if np.sum(visibility_mat_temp[:,k]) == 2:
                            index.append(k)
                    x1s = xs[index,:,i]
                    x2s = xs[index,:,jj]
                    _,p[:,:,j,i],_,_ = disambiguate_four_chirality_by_triangulation(four,x1s,x2s,K,draw_config=False)
        
        P = np.zeros((3,4,10,10))
        for i in range(10):
            for j in range(10):
                P[:,:,j,i] = K@p[:3,:,j,i]

        pred_pts = np.zeros((xs.shape[0],4))

        for i in range(xs.shape[0]):
            cameras = []
            for j in range(10):
                if visibility_mat[j,i] == 1:
                    cameras.append(j)

            P_temp = P[:,:,cameras[1:],cameras[0]]
            xs_now = xs[i,:,cameras]
            pred_pts[i,:] = triangulate_multi_camera(P_temp,xs_now)

        if final_vis:
            red = (1, 0, 0)
            green = (0, 1, 0)
            blue = (0, 0, 1)

            vis_3d(
                1500, 1500,
                o3d_pc(pts_3d, red),
                #o3d_pc(pred_pts, green),
                
                #draw_camera(K, pose1, img_w, img_h, 10, red),
                #draw_camera(K, pose2, img_w, img_h, 10, red),
                #draw_camera(K, pose3, img_w, img_h, 10, red),
                #draw_camera(K, pose4, img_w, img_h, 10, red),
                #draw_camera(K, pose5, img_w, img_h, 10, red),
                #draw_camera(K, pose6, img_w, img_h, 10, red),
                #draw_camera(K, pose7, img_w, img_h, 10, red),
                #draw_camera(K, pose8, img_w, img_h, 10, red),
                #draw_camera(K, pose9, img_w, img_h, 10, red),
                #draw_camera(K, pose10, img_w, img_h, 10, red),
                
                
                #draw_camera(K, p[...,0,0], img_w, img_h, 10, blue),
                #draw_camera(K, p[...,1,0], img_w, img_h, 10, blue),
                #draw_camera(K, p[...,2,0], img_w, img_h, 10, blue),
                #draw_camera(K, p[...,3,0], img_w, img_h, 10, blue),
                #draw_camera(K, p[...,4,0], img_w, img_h, 10, blue),
                #draw_camera(K, p[...,5,0], img_w, img_h, 10, blue),
                #draw_camera(K, p[...,6,0], img_w, img_h, 10, blue),
                #draw_camera(K, p[...,7,0], img_w, img_h, 10, blue),
                #draw_camera(K, p[...,8,0], img_w, img_h, 10, blue),
                #draw_camera(K, p[...,9,0], img_w, img_h, 10, blue),
                
                
            )

    def show_visib(self):
        plt.imshow(self.data.visibility, aspect="auto", interpolation='nearest')
        plt.xlabel("3D points")
        plt.ylabel("Camera view")
        plt.yticks(np.arange(10, dtype=int))
        plt.show()


def corruption_pipeline(x1s, x2s):
    s = 1
    noise = s * np.random.randn(*x1s.shape)
    noise[:, -1] = 0  # cannot add noise to the 1! fatal error

    x1s = x1s + noise
    x2s = x2s + noise

    x1s = flip_correspondence(x1s, 0.02)
    # round to integer
    x1s = np.rint(x1s)
    x2s = np.rint(x2s)
    return x1s, x2s


def flip_correspondence(x1s, perc):
    n = x1s.shape[0]
    num_wrong = int(n * perc)
    chosen = np.random.choice(n, size=num_wrong, replace=False)
    x1s[chosen] = x1s[np.random.permutation(chosen)]
    return x1s


def main():
    engine = Engine()

    engine.q10()
    # engine.q12()


if __name__ == "__main__":
    main()
