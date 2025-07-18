import open3d as o3d
import numpy as np

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 4
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria())
    return result.inlier_rmse,result.transformation

def feature_matching(src_path,ref_path,out_dir,progress=None,reg=None):
    import time
    import os
    start=time.time()
    voxel_size = 1  # means 5cm for this dataset
    source = o3d.io.read_point_cloud(src_path)
    target = o3d.io.read_point_cloud(ref_path)
    if progress:
        progress.advance(reg,30)
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    if progress:
        progress.advance(reg,10)
    inlier_rmse, transformation = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    
    if progress:
        progress.advance(reg,30)
    #print(inlier_rmse)
    #print(transformation)

    inlier_rmse_icp, transformation_icp = refine_registration(source_down, target_down,transformation,voxel_size)
    #print(inlier_rmse_icp)
    #print(transformation_icp)
    source_reg=source.transform(transformation_icp)
    o3d.io.write_point_cloud(os.path.join(out_dir,'registration_result.ply'),source_reg)
    print("Registration finished! The result is written to {}".format(os.path.join(out_dir,'registration_result.ply')))
    end=time.time()
    if progress:
        progress.advance(reg,30)
    #print("{} s".format(end-start))
    


def refine_registration(source, target, init_trans, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result.inlier_rmse, result.transformation


if __name__=='__main__':
    feature_matching(r'E:\data\tmp\uav-lidar\uav.ply',r'E:\data\tmp\uav-lidar\als.ply',r'E:\data\tmp\uav-lidar')