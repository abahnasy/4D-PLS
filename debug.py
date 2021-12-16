"""
Understand the formation of 4D volumes
load two point cloud frames
get their respective world transformation matrices
convert them into the world coordinates
project the rest except for the first one into the first frame
see the difference between leaving them in the world coordinates or projecting back to the first one
"""

import os

import numpy as np
import open3d as o3d



BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data')
SAVE_PATH = os.path.join(BASE_DIR, "../data/output")

frames_list = ['000001', '000005', '000010']

def parse_calibration(filename):
    """ read calibration file with given filename

        Returns
        -------
        dict
            Calibration matrices as 4x4 numpy arrays.
    """
    calib = {}

    calib_file = open(filename)
    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        calib[key] = pose

    calib_file.close()

    return calib



def parse_poses(filename, calibration):
    """ read poses file with per-scan poses from given filename

        Returns
        -------
        list
            list of poses as 4x4 numpy arrays.
    """
    file = open(filename)

    poses = []

    Tr = calibration["Tr"]
    Tr_inv = np.linalg.inv(Tr)

    for line in file:
        values = [float(v) for v in line.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

    return poses

poses = {}
for frame in frames_list:
    
    velo_file_path = os.path.join(DATA_PATH, 'velodyne', frame+'.bin')
    calib_path = os.path.join(DATA_PATH, 'calib.txt')
    pose_path = os.path.join(DATA_PATH, 'poses.txt')
    print(velo_file_path)
    # load velo path
    frame_points = np.fromfile(velo_file_path, dtype=np.float32)
    points = frame_points.reshape((-1, 4))
    # print(frame_points.shape)
    # print(points.shape)
    
    # Read Calib
    calibration = parse_calibration(calib_path)
    # Read poses
    poses_f64 = parse_poses(pose_path, calibration)
    poses[frame] = poses_f64[int(frame)]
    if frame == '000001':
        continue # skip the first frame, already exported
    
    #============================================================#
    # write the point cloud on desk
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points[:, 0:3])
    # o3d.io.write_point_cloud("{}/{}.ply".format(SAVE_PATH, frame), pcd)
    # print("written !")
    
    #============================================================#
    # write transformed point cloud on desk
    # Apply pose (without np.dot to avoid multi-threading)
    hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
    new_points = np.sum(np.expand_dims(hpoints, 2) * poses[frame].T, axis=1)
    new_points = new_points[:, 3:]
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(new_points[:, 0:3])
    # o3d.io.write_point_cloud("{}/{}_{}.ply".format(SAVE_PATH, frame, "world"), pcd)
    # print("written !")

    #============================================================#
    # We have to project in the first frame coordinates
    new_coords = new_points - poses['000001'][:3, 3]
    # new_coords = new_coords.dot(pose0[:3, :3])
    new_coords = np.sum(np.expand_dims(new_coords, 2) * poses['000001'][:3, :3], axis=1)
    print(">>> debugn", new_coords.shape)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(new_coords)
    o3d.io.write_point_cloud("{}/{}_{}.ply".format(SAVE_PATH, frame, "projected"), pcd)
    # print("written !")
    print(new_coords)
    exit()
    

    


