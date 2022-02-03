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
from models.losses_pointnet import *
from utils.config import bcolors
from models.dgcnn_utils import get_model_parameters
from utils.debugging import d_print


if __name__ == '__main__':

    config = Config()
    config.on_gpu = True
    config.saving_path = './results/vndgcnn/Experiments0202/instance_seg_metrics'

    DATASET_PATH = './data'
    train_set = SemanticKittiDataSet(path=DATASET_PATH, set='train', balance_classes= True, num_samples=4, augmentation='aligned',verbose=False)
    val_set = SemanticKittiDataSet(path=DATASET_PATH, set='val', balance_classes= True, num_samples=16, in_R=51., saving_path = config.saving_path, augmentation='aligned',verbose=False)
    train_loader = DataLoader(train_set, batch_size= 4, num_workers=4, shuffle=False, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size= 4, num_workers=4, shuffle=False, pin_memory=True)
    
    net=vnDGCNN(val_set.label_values, val_set.ignored_labels)
    
    # chkp_path = './results/vndgcnn/Experiments0127/I_250_fixed_gaussian/lr_search/0.1/0.1_resume/checkpoints/chkp_0100.tar'
    # chkp_path = './results/vndgcnn/Experiments0127/I_250_wrong_gaussian/checkpoints/chkp_0100.tar'
    # chkp_path = './results/vndgcnn/Experiments0129/I_250_balance_class/resume/checkpoints/best_chkp.tar'
    chkp_path = './results/vndgcnn/Experiments0202/I_250_balanced_sampling_old_gaussian/checkpoints/chkp_0100.tar'
    pretrained_model = torch.load(chkp_path, map_location='cpu')
    net.load_state_dict(pretrained_model['model_state_dict'], strict=True)

    
    with torch.no_grad():
        for batch in train_loader:
            sample_gpu ={}
            if config.on_gpu and torch.cuda.is_available():
                device = torch.device("cuda")
                for k, v in batch.items():
                    sample_gpu[k] = v.to(device)
                net.to(device)
            else:
                device = torch.device("cpu")
                sample_gpu = batch

            outputs, centers_output, var_output, embeddings = net(sample_gpu['in_fts'][:,:,:3])
            
            points = sample_gpu['in_fts'][:,:,:3]
            times = sample_gpu['in_fts'][:,:,8]
            ins_labels = sample_gpu['in_slbls']
            centers_gt = sample_gpu['in_fts'][:,:,4:8]
            
            batch_size = outputs.shape[0]
            num_points = outputs.shape[1]

            centers_gt = centers_gt.view(num_points*batch_size, -1)
            centers_p = centers_output.view(num_points*batch_size)
            ins_labels = ins_labels.squeeze().view(batch_size*num_points) # [B,N,1] -> [B*N,] ground truth
            variances = var_output.view(batch_size*num_points, -1) # [B,N,260] -> [B*N,260]
            points = points.view(batch_size*num_points, -1)
            times = times.view(batch_size*num_points, 1)
            embeddings = embeddings.view(batch_size*num_points, -1)

            instance_loss = iou_instance_loss(
                    centers_p, embeddings, variances, ins_labels, 
                    points,
                    times)
            
            d_print('instance loss:{}'.format(instance_loss/batch_size))