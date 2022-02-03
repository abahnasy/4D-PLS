import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np


from datasets.semantic_kitti_dataset import SemanticKittiDataSet
from models.dgcnn_sem_seg import DGCNN_semseg
from utils.trainer_dgcnn import ModelTrainerDGCNN
from utils.config import Config
from models.dgcnn_utils import get_model_parameters



def seed_torch(seed=0):
    #random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    seed_torch(seed=0)

    config = Config()
    config.on_gpu = True
    config.checkpoint_gap = 100
    config.grad_clip_norm = -100.0
    
    config.val_sem = False 
    config.max_epoch = 500
    config.learning_rate = 0.1 
    config.lr_scheduler = False      
    config.saving_path = './results/dgcnn/Experiments0203/'+'I_250_balanced_sampling_old_gaussian'

    DATASET_PATH = './data'
    train_set = SemanticKittiDataSet(path=DATASET_PATH, set='train', balance_classes= True, num_samples=250, augmentation='aligned',verbose=False)
    val_set = SemanticKittiDataSet(path=DATASET_PATH, set='val', balance_classes= True, num_samples=20, in_R=51., saving_path = config.saving_path, augmentation='z',verbose=False)
    train_loader = DataLoader(train_set, batch_size= 4, num_workers=4, shuffle=False, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size= 4, num_workers=4, shuffle=False, pin_memory=True)
    
    net=DGCNN_semseg(train_set.label_values, train_set.ignored_labels, input_feature_dims=4)
    num_parameters = get_model_parameters(net)
    print('Number of model parameters:', num_parameters)

    config.resume_training = False
    

    # Training with pretrained weights
    chkp_path = './results/dgcnn_semseg_pretrained/model_1.t7'
    trainer = ModelTrainerDGCNN(net, config, chkp_path=chkp_path, resume_training=config.resume_training, on_gpu=config.on_gpu)
    trainer.train(config, net, train_loader, val_loader, loss_type='4DPLSloss')
