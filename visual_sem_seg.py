import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
import open3d as o3d

from datasets.semantic_kitti_dataset import SemanticKittiDataSet
from models.vnDGCNN_sem_seg import vnDGCNN
from utils.trainer_vnDGCNN import ModelTrainervnDGCNN
from utils.config import Config
from utils.config import bcolors
from models.dgcnn_utils import get_model_parameters
from utils.debugging import d_print



def visualize_semantic_acc(pc, probs, labels, label_values, ignored_labels, abs_path):
    """ Write colored point cloud into val_preds folder
    Args:
        outputs: predicted semantic labels
        labels: gt data
    Return:
        None
    """
    assert pc.shape[1] == 3
    d_print(pc.shape)
    # get correctly classified labels
    # plot them in green
    # plot wrongly classified in red
    probs = probs.view(-1,19)
    # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
    for l_ind, label_value in enumerate(label_values):       
        if label_value in ignored_labels:
            probs = np.insert(probs, l_ind, 0, axis=1)
    
    values, counts = np.unique(labels, return_counts=True)
    d_print('count labels in gt labels:\n labels:{0}\n counts:{1}\n'.format(values, counts))
    
    preds = label_values[np.argmax(probs, axis=1)]
    
    values, counts = np.unique(preds, return_counts=True)
    d_print('count labels in preds: \n preds:{0}\n counts:{1}\n'.format(values, counts))
    
    preds = torch.from_numpy(preds)
    # preds.to(outputs.device)

    # predicted = torch.argmax(outputs, dim=1)
    total = preds.size(0)
    colors = np.zeros((total, 3))
    colors[:,0] = 1 # mark all as incorrect
    correct = np.where(preds == labels)
    d_print(correct)
    # d_print(correct.shape)
    colors[correct] = np.array([0,1,0]) # mark as green

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud("{}.ply".format(abs_path), pcd)
    if os.path.exists("{}.ply".format(abs_path)):
        d_print("successful write on desk", bcolors.OKGREEN)
    else:
        d_print("Invalid Point CLoud write", bcolors.FAIL)

    # write another ply file for semantics of the things class
    things_idx = np.where(np.logical_and(preds > 0,preds < 9))
    pc = pc[things_idx]
    d_print("things shape: {}".format(pc.shape))
    colors = np.zeros((pc.shape[0], 3))
    colors[:] = np.array([0,1,1])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.colors = o3d.utility.Vector3dVector(colors)


    o3d.io.write_point_cloud("{}_{}.ply".format(abs_path, "things"), pcd)
    if os.path.exists("{}_{}.ply".format(abs_path, "things")):
        d_print("successful write on desk", bcolors.OKGREEN)
    else:
        d_print("Invalid Point CLoud write", bcolors.FAIL)



if __name__ == '__main__':

    DATASET_PATH = './data'
    train_set = SemanticKittiDataSet(path=DATASET_PATH, set='train', balance_classes= True, num_samples=1, augmentation='aligned',verbose=False)
    val_set = SemanticKittiDataSet(path=DATASET_PATH, set='val', num_samples=1, augmentation='aligned',verbose=False)
    train_loader = DataLoader(train_set, batch_size= 1, num_workers=1, shuffle=False, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size= 1, num_workers=1, shuffle=False, pin_memory=True)
    
    net=vnDGCNN(val_set.label_values, val_set.ignored_labels)

    chkp_path = './results/vndgcnn/Experiments0129/I_250_balance_class/checkpoints/current_chkp.tar'
    pretrained_model = torch.load(chkp_path, map_location='cpu')
    net.load_state_dict(pretrained_model['model_state_dict'], strict=True)

    config = Config()
    config.saving_path = './visuals/sem_seg_acc'
    
    with torch.no_grad():
        # for batch in val_loader:
        #     sample_gpu = batch
        #     outputs, _, _, _ = net(sample_gpu['in_fts'][:,:,:3])
        #     viz_pc = sample_gpu['in_pts'].squeeze()
        #     break
        # s_ind = sample_gpu['s_ind']
        # f_ind = sample_gpu['f_ind']

        sample_gpu = train_set[0]
        viz_pc = sample_gpu['in_pts'].squeeze()
        outputs, _, _, _ = net(torch.tensor(sample_gpu['in_pts']).unsqueeze(0))
        s_ind = 0
        f_ind = 0
        filename = '{:s}_{:07d}'.format(val_loader.dataset.sequences[s_ind], f_ind)
        visualize_semantic_acc(
            viz_pc,
            outputs.cpu(), 
            sample_gpu['in_lbls'], #sample_gpu['in_lbls'].squeeze(0)
            val_loader.dataset.label_values,
            val_loader.dataset.ignored_labels,
            os.path.join(config.saving_path, 'val_preds', filename)
        )

