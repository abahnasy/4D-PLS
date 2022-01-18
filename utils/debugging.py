import os
import numpy as np
import torch
import open3d as o3d
from prettytable import PrettyTable

from utils.config import bcolors

def d_print(str, color = bcolors.WARNING):
    color = color
    print("{:}{}{:}".format(color, str, bcolors.ENDC))

def write_pc(pc: np.ndarray, abs_path: str):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    o3d.io.write_point_cloud("{}.ply".format(abs_path), pcd)
    if os.path.exists("{}.ply".format(abs_path)):
        d_print("successful write on desk", bcolors.OKGREEN)
    else:
        d_print("Invalid Point CLoud write", bcolors.FAIL)

def seed_torch(seed=0):
    #random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def rotate_pc(pc: np.ndarray, angle):
    """
    Parameters:
        pc: 
        angle: angle in degrees
    Returns:

    """
    # Initialize rotation matrix
    R = np.eye(pc.shape[1])
    assert pc.shape[1] == 3, "check pc dimensions"
     # Create random rotations
    theta = angle*np.pi/180.
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    R = R.astype(np.float32)
    augmented_points = np.sum(np.expand_dims(pc, 2) * R, axis=1)

    return augmented_points, R


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params