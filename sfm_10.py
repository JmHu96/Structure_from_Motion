import numpy as np
from numpy.linalg import inv, svd, det, norm
import scipy as sci
from scipy import optimize

from render import (
    as_homogeneous,
    homogenize
)

import torch
import torch.optim as optim
from lietorch import SE3
from scipy.spatial.transform import Rotation as rot
from vis import vis_3d, o3d_pc, draw_camera


def as_cross(t):
    assert len(t) == 3
    t1,t2,t3 = t[0],t[1],t[2]
    t_cross = np.zeros((3,3))
    t_cross[0,1] = -t3
    t_cross[1,0] = t3
    t_cross[0,2] = t2
    t_cross[2,0] = -t2
    t_cross[1,2] = -t1
    t_cross[2,1] = t1
    return t_cross

def project(pts_3d, K, pose):
    P = K @ inv(pose)
    x1 = homogenize(pts_3d @ P.T)
    return x1


def pose_to_se3_embed(pose):
    R, t = pose[:3, :3], pose[:3, -1]
    tau = t
    phi = rot.from_matrix(R).as_quat()  # convert to quaternion
    embed = np.concatenate([tau, phi], axis=0)
    embed = torch.as_tensor(embed)
    return embed


def as_torch_tensor(*args):
    return [torch.as_tensor(elem) for elem in args]


def torch_project(pts_3d, K, se3_pose):
    P = K @ se3_pose.inv().matrix()
    x1 = pts_3d @ P.T
    x1 = homogenize(x1)
    return x1


def bundle_adjustment(x1s, x2s, full_K, p1, p2, pred_pts):
    embed1 = pose_to_se3_embed(p1)
    embed2 = pose_to_se3_embed(p2)

    embed1.requires_grad_(True)
    embed2.requires_grad_(True)

    x1s, x2s, full_K, pred_pts = \
        as_torch_tensor(x1s, x2s, full_K, pred_pts)

    pred_pts.requires_grad_(True)

    lr = 1e-3
    # optimizer = optim.SGD([embed1, embed2, pred_pts], lr=lr, momentum=0.9)
    optimizer = optim.Adam([embed1, embed2, pred_pts], lr=lr)

    n_steps = 10000
    for i in range(n_steps):
        optimizer.zero_grad()

        p1 = SE3.InitFromVec(embed1)
        p2 = SE3.InitFromVec(embed2)

        x1_hat = torch_project(pred_pts, full_K, p1)
        x2_hat = torch_project(pred_pts, full_K, p2)
        err1 = torch.norm((x1_hat - x1s), dim=1)
        err1 = err1.mean()
        err2 = torch.norm((x2_hat - x2s), dim=1)
        err2 = err2.mean()
        err = (err1 + err2) / 2

        err.backward()
        optimizer.step()

        if (i % (n_steps // 10)) == 0:
            print(f"step {i}, err: {err.item()}")

    p1 = SE3.InitFromVec(embed1).matrix().detach().numpy()
    p2 = SE3.InitFromVec(embed2).matrix().detach().numpy()
    pred_pts = pred_pts.detach().numpy()
    return p1, p2, pred_pts


def eight_point_algorithm(x1s, x2s):
    # estimate the fundamental matrix
    # your code here
    n = len(x1s)
    A = np.zeros((n,9))

    for i in range(n):
        x1 = x1s[i,:]
        x2 = x2s[i,:]
        A[i,:] = np.einsum('i,j -> ij', x2, x1).flatten()
    
    _, _, V_t = svd(A,full_matrices=False)

    # cond number is the ratio of top rank / last rank singular values
    # when you solve Ax = b, you take s[0] / s[-1]. But in vision,
    # we convert the problem above into the form Ax = 0. The nullspace is reserved for the solution.
    # This is called a "homogeneous system of equations" in linear algebra. This might be the reason
    # why homogeneous coordiantes are called homogeneous.
    # hence s[0] / s[-2].
    #cond = s[0] / s[-2]
    #print(f"condition number {cond}")

    least_sigular_vec = V_t[8,:]
    F = least_sigular_vec.reshape((3,3))
    F = enforce_rank_2(F)
    return F


def enforce_rank_2(F):
    # your code here
    U,s,V_t = np.linalg.svd(F)
    s[2] = 0
    F = U@np.diag(s)@V_t
    return F


def normalized_eight_point_algorithm(x1s, x2s, img_w, img_h, T):
    half_w = img_w / 2
    half_h = img_h / 2

    x1s = T@x1s.T
    x2s = T@x2s.T

    F = eight_point_algorithm(x1s.T, x2s.T)
    return F


def triangulate(P1, x1s, P2, x2s):
    # x1s: [n, 3]
    assert x1s.shape == x2s.shape
    n = len(x1s)
    pts = np.zeros((n,4))
    # you can follow this and write it in a vectorized way, or you can do it
    # row by row, entry by entry

    if P1.shape[0] == 4:
        P1,P2 = P1[:3,:], P2[:3,:]
    
    for i in range(x1s.shape[0]):
        x1 = as_cross(x1s[i,:])
        x2 = as_cross(x2s[i,:])
        G = np.concatenate((x1@P1,x2@P2))
        _,_,V_t = np.linalg.svd(G,full_matrices=False)
        pts[i,:] = V_t[3,:]/V_t[3,-1]
    
    
    return pts

def triangulate_multi_camera(P,x):
    n = x.shape[0]
    n_P = P.shape[2]

    def fun(X):
        sum = 0
        for i in range(n):
            u = x[i,0]
            v = x[i,1]
            for j in range(n_P+1):
                if j == 0:
                    P_now = np.eye(4)
                else:
                    P_now = P[:,:,j-1]
                sum += (u-(P_now[0,:]@X)/P_now[2,:]@X)**2 + (v-(P_now[1,:]@X/P_now[2,:]@X)**2)
        return sum

    pts = optimize.minimize(fun,x0=np.ones(4))
    return pts.x


def t_and_R_from_pose_pair(p1, p2):
    """the R and t that transforms points from pose 1's local frame to pose 2's local frame
    """
    trans_mat = inv(p2)@p1
    return trans_mat[:3,3], trans_mat[:3,:3]


def pose_pair_from_t_and_R(t, R):
    """since we only have their relative orientation, the first pose
    is fixed to be identity
    """
    p11 = np.eye(4)
    # your code here
    p22 = np.zeros((4,4))
    p22[:3,:3] = R
    p22[:3,3] = t
    p22[3,3] = 1
    return p11, inv(p22)


def essential_from_t_and_R(t, R):
    # your code here
    E = as_cross(t)@R
    return E


def t_and_R_from_essential(E):
    """
    this has even more ambiguity. there are 4 compatible (t, R) configurations
    out of which only 1 places all points in front of both cameras

    That the rank-deficiency in E induces 2 valid R is subtle...
    """
    # your code here; get t
    # get t from left-null space of E
    t = sci.linalg.null_space(E.T)
    t = t[:,0]

    # now solve procrustes to get back R
    t_cross = as_cross(t)
    U, s, V_t = svd(t_cross.T@E)
    R = U@V_t

    # makes sure R has det 1, and that we have 2 possible Rs

    R1 = R * det(R)
    U[:, 2] = -U[:, 2]
    R = U @ V_t
    R2 = R * det(R)

    four_hypothesis = [
        [ t, R1],
        [-t, R1],
        [ t, R2],
        [-t, R2],
    ]
    return four_hypothesis


# incorrect
def disambiguate_four_chirality_by_triangulation(four, x1s, x2s, full_K, draw_config=False):
    # note that our camera is pointing towards its negative z axis
    num_infront = np.array([0, 0, 0, 0])
    four_pose_pairs = []
    K = full_K[:3,:3]

    for i, (t, R) in enumerate(four):
        # error 1
        # review session in HW2
        p1, p2 = pose_pair_from_t_and_R(t, R)
        p1 = inv(p1)
        p2 = inv(p2)
        P1 = K@p1[:3,:]
        P2 = K@p2[:3,:]
        
        pts = triangulate(P1,x1s,P2,x2s)
        nv1 = 0 # how many points in front of camera 1?
        nv2 = 0 # how many points in front of camera 2?
        
        p1x1 = inv(p1)@pts.T
        p2x2 = inv(p2)@pts.T
        nv1 = np.sum(p1x1.T[:,-2] < 0)
        nv2 = np.sum(p2x2.T[:,-2] < 0)

        num_infront[i] = nv1 + nv2

        p1, p2 = inv(p1), inv(p2)

        four_pose_pairs.append((p1, p2))
        if draw_config:
            vis_3d(
                1500, 1500, o3d_pc(_throw_outliers(pts)),
                draw_camera(full_K, p1, 1600, 1200),
                draw_camera(full_K, p2, 1600, 1200),
            )
    i = np.argmax(num_infront)
    t, R = four[i]
    p1, p2 = four_pose_pairs[i]
    return p1, p2, t, R


def F_from_K_and_E(K, E):
    # your code
    F = inv(K.T) @ E @ inv(K)
    return F


def E_from_K_and_F(K, F):
    # your code
    E = K.T @ F @ K
    return E


def _throw_outliers(pts):
    pts = pts[:, :3]
    mask = (np.abs(pts) > 10).any(axis=1)
    return pts[~mask]


def align_B_to_A(B, p1, p2, A):
    # B, A: [n, 3]
    assert B.shape == A.shape
    A = A[:, :3]
    B = B[:, :3]
    p1 = p1.copy()
    p2 = p2.copy()

    a_centroid = A.mean(axis=0)
    b_centroid = B.mean(axis=0)

    A = A - a_centroid
    B = B - b_centroid
    p1[:3, -1] -= b_centroid
    p2[:3, -1] -= b_centroid

    centroid = np.array([0, 0, 0])
    # root mean squre from centroid
    scale_a = (norm((A - centroid), axis=1) ** 2).mean(axis=0) ** 0.5
    scale_b = (norm((B - centroid), axis=1) ** 2).mean(axis=0) ** 0.5
    rms_ratio = scale_a / scale_b

    B = B * rms_ratio
    p1[:3, -1] *= rms_ratio
    p2[:3, -1] *= rms_ratio

    U, s, V_t = svd(B.T @ A)
    R = U @ V_t
    #assert np.allclose(det(R), 1), "not special orthogonal matrix"
    new_B = B @ R  # note that here there's no need to transpose R... lol... this is subtle
    p1[:3] = R.T @ p1[:3]
    p2[:3] = R.T @ p2[:3]

    new_B = new_B + a_centroid
    new_B = as_homogeneous(new_B)
    p1[:3, -1] += a_centroid
    p2[:3, -1] += a_centroid
    return new_B, p1, p2
