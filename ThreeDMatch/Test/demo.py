import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import numpy as np
import torch
import shutil
import torch.nn as nn
import sys
import teaserpp_python
sys.path.append('../../')
import script.common as cm
from ThreeDMatch.Test.tools import get_pcd, get_keypts
from sklearn.neighbors import KDTree
import importlib
import open3d
import copy
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
def make_open3d_point_cloud(xyz, color=None):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(xyz)
    if color is not None:
        pcd.paint_uniform_color(color)
    return pcd


vicinity=3

def build_patch_input(pcd, keypts, vicinity=0.8, num_points_per_patch=2048):
    refer_pts = keypts.astype(np.float32)
    pts = np.array(pcd.points).astype(np.float32)
    num_patches = refer_pts.shape[0]
    tree = KDTree(pts[:, 0:3])
    ind_local = tree.query_radius(refer_pts[:, 0:3], r=vicinity)   # vicinity 会不会是这个导致不准的？？ 尤其是对于我们训练集的数据！！！！ 0.3和1.0差别
    local_patches = np.zeros([num_patches, num_points_per_patch, 3], dtype=float)
    for i in range(num_patches):
        local_neighbors = pts[ind_local[i], :]
        if local_neighbors.shape[0] >= num_points_per_patch:
            temp = np.random.choice(range(local_neighbors.shape[0]), num_points_per_patch, replace=False)
            local_neighbors = local_neighbors[temp]
            local_neighbors[-1, :] = refer_pts[i, :]
        else:
            fix_idx = np.asarray(range(local_neighbors.shape[0]))   # 可能会筛选出重复的点，是否会有影响？
            while local_neighbors.shape[0] + fix_idx.shape[0] < num_points_per_patch:
                fix_idx = np.concatenate((fix_idx, np.asarray(range(local_neighbors.shape[0]))), axis=0)
            random_idx = np.random.choice(local_neighbors.shape[0], num_points_per_patch - fix_idx.shape[0],
                                          replace=False)
            choice_idx = np.concatenate((fix_idx, random_idx), axis=0)
            local_neighbors = local_neighbors[choice_idx]
            local_neighbors[-1, :] = refer_pts[i, :]
        local_patches[i] = local_neighbors

    return local_patches


def noise_Gaussian(points, std):
    noise = np.random.normal(0, std, points.shape)
    noise[noise>0.05]=0.05
    noise[noise < -0.05] = -0.05
    out = points + noise
    return out

def noise_ramdom(points):
    noise = np.random.rand([points.shape[0],points.shape[1]])*0.1-0.05
    out = points + noise
    return out

def noise_Gaussian_proportion(points, std, proportion):
    noise = np.random.normal(0, std, points.shape)
    noise[noise>0.05]=0.05
    noise[noise < -0.05] = -0.05
    select=np.random.rand(points.shape[0])
    out = points
    out[select<proportion]=points[select<proportion]+noise[select<proportion]
    return out

def unsampling_points(points,num_key=2000):
    n = np.random.choice(len(points),num_key , replace=False)
    keypoints= points[n]
    return keypoints

def noise_Gaussian_replace(points, std, proportion):
    noise = np.random.normal(0, std, points.shape)
    noise[noise>0.05]=0.05
    noise[noise < -0.05] = -0.05
    select=np.random.rand(points.shape[0])
    out = points
    out[select<proportion]=points[select<proportion]+noise[select<proportion]
    return out

def prepare_patch(pcdpath,numkey=2000):

    pcd = open3d.io.read_point_cloud(pcdpath)

    voxel_size = 0.05  # 根据需求设置体素大小
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_np = np.array(pcd_down.points).astype(np.float32)

    # pcd_np = np.array(pcd.points).astype(np.float32)
    keypoints = unsampling_points(pcd_np,numkey)
    local_patches = build_patch_input(pcd_down, keypoints)  # [num_keypts, 1024, 4]
    return local_patches, keypoints


def generate_descriptor(model, pcdpath,numkey=2000):
    model.eval()
    with torch.no_grad():
        local_patches, keypoints = prepare_patch(pcdpath,numkey)
        input_ = torch.tensor(local_patches.astype(np.float32))
        B = input_.shape[0]
        input_ = input_.cuda()
        model = model.cuda()
        # calculate descriptors
        desc_list = []
        start_time = time.time()
        desc_len = 64
        step_size =8
        iter_num = np.int32(np.ceil(B / step_size))
        for k in range(iter_num):
            if k == iter_num - 1:
                desc = model(input_[k * step_size:, :, :])
            else:
                desc = model(input_[k * step_size: (k + 1) * step_size, :, :])
            desc_list.append(desc.view(desc.shape[0], desc_len).detach().cpu().numpy())
            del desc
        step_time = time.time() - start_time
        print(f'Finish {B} descriptors spend {step_time:.4f}s')
        desc = np.concatenate(desc_list, 0).reshape([B, desc_len])

    return desc, keypoints

def calculate_M(source_desc, target_desc):
    """
    Find the mutually closest point pairs in feature space.
    source and target are descriptor for 2 point cloud key points. [5000, 512]
    """

    kdtree_s = KDTree(target_desc)
    sourceNNdis, sourceNNidx = kdtree_s.query(source_desc, 1)
    kdtree_t = KDTree(source_desc)
    targetNNdis, targetNNidx = kdtree_t.query(target_desc, 1)
    result = []
    for i in range(len(sourceNNidx)):
        if targetNNidx[sourceNNidx[i]] == i:
            result.append([i, sourceNNidx[i][0]])
    return np.array(result)

def find_correspondences( feats0, feats1, mutual_filter=True):
        nns01 = find_knn_cpu(feats0, feats1, knn=1, return_distance=False)
        corres01_idx0 = np.arange(len(nns01))
        corres01_idx1 = nns01

        if not mutual_filter:
            return corres01_idx0, corres01_idx1

        nns10 = find_knn_cpu(feats1, feats0, knn=1, return_distance=False)
        corres10_idx1 = np.arange(len(nns10))
        corres10_idx0 = nns10

        mutual_filter = (corres10_idx0[corres01_idx1] == corres01_idx0)
        corres_idx0 = corres01_idx0[mutual_filter]
        corres_idx1 = corres01_idx1[mutual_filter]

        return corres_idx0, corres_idx1

def find_knn_cpu(feat0, feat1, knn=1, return_distance=False):
        feat1tree = cKDTree(feat1)
        dists, nn_inds = feat1tree.query(feat0, k=knn)
        if return_distance:
            return nn_inds, dists
        else:
            return nn_inds
def register2Fragments(keypoints1, keypoints2, descriptor1, descriptor2, gtTrans = None):

    source_keypts = keypoints1
    target_keypts = keypoints2
    source_desc = np.nan_to_num(descriptor1)
    target_desc = np.nan_to_num(descriptor2)
    if source_desc.shape[0] > num_keypoints:
        rand_ind = np.random.choice(source_desc.shape[0], num_keypoints, replace=False)
        source_keypts = source_keypts[rand_ind]
        target_keypts = target_keypts[rand_ind]
        source_desc = source_desc[rand_ind]
        target_desc = target_desc[rand_ind]

    if gtTrans is not None:
        # find mutually cloest point.
        corr = calculate_M(source_desc, target_desc)
        frag1 = source_keypts[corr[:, 0]]
        frag2_pc = open3d.geometry.PointCloud()
        frag2_pc.points = open3d.utility.Vector3dVector(target_keypts[corr[:, 1]])
        frag2_pc.transform(gtTrans)
        frag2 = np.asarray(frag2_pc.points)
        distance = np.sqrt(np.sum(np.power(frag1 - frag2, 2), axis=1))
        num_inliers = np.sum(distance < 0.10)
        inlier_ratio = num_inliers / len(distance)
        gt_flag = 1
        print(f"num_inliers:{num_inliers}")
        print(f"inlier_ratio:{inlier_ratio}")

    # calculate the transformation matrix using RANSAC, this is for Registration Recall.
    source_pcd = open3d.geometry.PointCloud()
    source_pcd.points = open3d.utility.Vector3dVector(source_keypts)
    target_pcd = open3d.geometry.PointCloud()
    target_pcd.points = open3d.utility.Vector3dVector(target_keypts)
    s_desc = open3d.pipelines.registration.Feature()
    s_desc.data = source_desc.T
    t_desc = open3d.pipelines.registration.Feature()
    t_desc.data = target_desc.T

    # registration method: registration_ransac_based_on_feature_matching
    result = open3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_pcd,
        target_pcd,
        s_desc,
        t_desc,
        mutual_filter=True,
        max_correspondence_distance=0.5,
        estimation_method=open3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            open3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            open3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.2),
        ],
        criteria=open3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000),
    )

    return result.transformation, result.correspondence_set


def draw_registration_result(source, target):
    # vis=open3d.visualization.Visualizer()
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    open3d.geometry.PointCloud.estimate_normals(source_temp)
    open3d.geometry.PointCloud.estimate_normals(target_temp)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    open3d.visualization.draw_geometries([source_temp, target_temp])

def draw_registration_corr(source, target,source_keypoint, target_keypoint, transformation, correspondence_set):
    # vis=open3d.visualization.Visualizer()
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_keypoint_transform = copy.deepcopy(source_keypoint)
    open3d.geometry.PointCloud.estimate_normals(source_temp)
    open3d.geometry.PointCloud.estimate_normals(target_temp)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])

    source_keypoint_transform.transform(transformation)
    source_transform_np = np.array(source_keypoint_transform.points).astype(np.float32)
    target_np = np.array(target_keypoint.points).astype(np.float32)
    correspondence = np.array(correspondence_set).astype(np.int32)
    distance = np.sqrt(np.sum(np.power(source_transform_np[correspondence[:,0]] - target_np[correspondence[:,1]], 2), axis=1))
    correspondence_inlier = correspondence[distance<0.1]
    correspondence_inlier = open3d.utility.Vector2iVector(correspondence_inlier)

    source_temp.translate((5, 0, 0), relative=True)
    source_keypoint.translate((5, 0, 0), relative=True)
    # source_temp.transform(transformation)
    inlier_corr_line = open3d.geometry.LineSet.create_from_point_cloud_correspondences(source_keypoint, target_keypoint, correspondence_inlier)
    inlier_corr_line.paint_uniform_color([0, 1, 0])
    corr_line = open3d.geometry.LineSet.create_from_point_cloud_correspondences(source_keypoint, target_keypoint, correspondence)
    corr_line.paint_uniform_color([1, 0, 0])
    open3d.visualization.draw_geometries([source_temp, target_temp, inlier_corr_line, corr_line])

def get_teaser_solver(noise_bound):
        solver_params = teaserpp_python.RobustRegistrationSolver.Params()
        solver_params.cbar2 = 1.0
        solver_params.noise_bound = noise_bound
        solver_params.estimate_scaling = False
        solver_params.inlier_selection_mode = \
            teaserpp_python.RobustRegistrationSolver.INLIER_SELECTION_MODE.PMC_EXACT
        solver_params.rotation_tim_graph = \
            teaserpp_python.RobustRegistrationSolver.INLIER_GRAPH_FORMULATION.CHAIN
        solver_params.rotation_estimation_algorithm = \
            teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
        solver_params.rotation_gnc_factor = 1.4
        solver_params.rotation_max_iterations = 10000
        solver_params.rotation_cost_threshold = 1e-16
        solver = teaserpp_python.RobustRegistrationSolver(solver_params)
        return solver
if __name__ == '__main__':

    # dynamically load the model
    module_file_path = '../model.py'
    shutil.copy2(os.path.join('.', '../../network/SphereNet.py'), module_file_path)
    module_name = ''
    module_spec = importlib.util.spec_from_file_location(module_name, module_file_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    model = module.SphereNet(1.5, 15, 40, 20, '3DMatch', True, True)
    model = nn.DataParallel(model, device_ids=[0])
    model.load_state_dict(torch.load('../../pretrain/3DMatch_generalization.pkl'))
    num_keypoints =2000

    pcdpath1 = f"/home/wyw/ROS1_PROJECT/BD/2023/Multi_lidar/MicroG-main/data/6.pcd"
    pcdpath2 = f"/home/wyw/ROS1_PROJECT/BD/2023/Multi_lidar/MicroG-main/data/target20.pcd"


    pcd1 = open3d.io.read_point_cloud(pcdpath1)
    pcd2 = open3d.io.read_point_cloud(pcdpath2)
    # 查看点云数量
    num_points_pcd1 = len(pcd1.points)
    num_points_pcd2 = len(pcd2.points)
    print(num_points_pcd2/num_points_pcd1)

    print(f"点云1的点数量: {num_points_pcd1}")
    print(f"点云2的点数量: {num_points_pcd2}")
    descriptor1, keypoints1 = generate_descriptor(model, pcdpath1,1000)
    descriptor2, keypoints2 = generate_descriptor(model, pcdpath2,5000)
    print(type(keypoints1))  # <class 'int'>
    print(type(keypoints1))  # <class 'str'>
    corrs_A, corrs_B = find_correspondences(
        descriptor1, descriptor2, mutual_filter=True)
    corrs_A = corrs_A.T
    corrs_B = corrs_B.T
    keypoints1=keypoints1.T
    keypoints2 = keypoints2.T
    A_corr = keypoints1[:, corrs_A]
    B_corr = keypoints2[:, corrs_B]
    # # transformation, correspondence_set = register2Fragments(keypoints1, keypoints2, descriptor1, descriptor2)
    # numpy_data = np.asarray(correspondence_set)
    # print(numpy_data)
    teaser_solver =get_teaser_solver(0.1)
    teaser_solver.solve(A_corr, B_corr)
    solution = teaser_solver.getSolution()
    transformation = np.eye(4)
    transformation[:3, :3] = solution.rotation
    transformation[:3, 3] = solution.translation
    print(transformation)
    source_pcd = open3d.geometry.PointCloud()
    source_pcd.points = open3d.utility.Vector3dVector(keypoints1.T)
    target_pcd = open3d.geometry.PointCloud()
    target_pcd.points = open3d.utility.Vector3dVector(keypoints2.T)
    # draw_registration_corr(pcd1, pcd2, source_pcd, target_pcd, transformation, correspondence_set)
    draw_registration_result(source_pcd, target_pcd)

    draw_registration_result(pcd1, pcd2)
