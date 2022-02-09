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
    
    # d_print(correct[1])
    # d_print(correct.shape)
    colors[correct[1]] = np.array([0,1,0]) # mark as green

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

    config = Config()
    config.saving_path = './visuals/sem_seg_acc'

    DATASET_PATH = './data'
    train_set = SemanticKittiDataSet(path=DATASET_PATH, set='train', balance_classes= False, num_samples=10, augmentation='aligned',verbose=False)
    val_set = SemanticKittiDataSet(path=DATASET_PATH, set='val', balance_classes= True, num_samples=10, in_R=51., saving_path = config.saving_path, augmentation='aligned',verbose=False)
    train_loader = DataLoader(train_set, batch_size= 1, num_workers=1, shuffle=False, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size= 1, num_workers=1, shuffle=False, pin_memory=True)
    
    # # checkpoint of DGCNN on 250 frames in sequence 4, balanced sampling, old gaussian pdf
    # net=DGCNN_semseg(train_set.label_values, train_set.ignored_labels, input_feature_dims=4)
    # chkp_path = './results/dgcnn/Experiments0203/I_250_balanced_sampling_old_gaussian/checkpoints/best_chkp.tar'

    # # checkpoint of VN-DGCNN on 250 frames in sequence 4, balanced sampling, old gaussian pdf    
    # net=vnDGCNN(val_set.label_values, val_set.ignored_labels)
    # chkp_path = './results/vndgcnn/Experiments0202/I_250_balanced_sampling_old_gaussian/checkpoints/best_chkp.tar'
    
    # checkpoint of VN-DGCNN on 250 frames in sequence 4, unbalanced sampling, old gaussian pdf 
    net=vnDGCNN(val_set.label_values, val_set.ignored_labels)
    chkp_path = './results/vndgcnn/I_250_wrong_gaussian_unbalanced/resume/checkpoints/best_chkp.tar'
    pretrained_model = torch.load(chkp_path, map_location='cpu')
    net.load_state_dict(pretrained_model['model_state_dict'], strict=True)

    
    with torch.no_grad():
        for batch in val_loader:
            sample_gpu = batch
            outputs, _, _, _ = net(sample_gpu['in_fts'][:,:,:3])
            viz_pc = sample_gpu['in_pts'].squeeze()
            break
        s_ind = sample_gpu['s_ind']
        f_ind = sample_gpu['f_ind']
        
        # s_ind = 0
        # f_ind = 0
        # sample_gpu = train_set[f_ind]
        # viz_pc = sample_gpu['in_fts'][:,:3]
        # outputs, _, _, _ = net(torch.tensor(sample_gpu['in_fts'][:,:3]).unsqueeze(0))
        d_print(val_loader.dataset.sequences[s_ind])
        d_print(f_ind)
        filename = '{:s}_{:07d}'.format(val_loader.dataset.sequences[s_ind], int(f_ind))
        visualize_semantic_acc(
            viz_pc,
            outputs.cpu(), 
            sample_gpu['in_lbls'], #sample_gpu['in_lbls'].squeeze(0)
            val_loader.dataset.label_values,
            val_loader.dataset.ignored_labels,
            os.path.join(config.saving_path, 'val_preds', filename)
        )

