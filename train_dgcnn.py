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


def seed_torch(seed=0):
    #random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    seed_torch(seed=0)

    DATASET_PATH = './data'
    train_set = SemanticKittiDataSet(path=DATASET_PATH, set='train')
    val_set = SemanticKittiDataSet(path=DATASET_PATH, set='val')
    train_loader = DataLoader(train_set, batch_size= 4, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size= 4, num_workers=1, pin_memory=True)
    
    net=DGCNN_semseg(train_set.label_values, train_set.ignored_labels, input_feature_dims=9)
    
    config = Config()
    config.learning_rate = 0.1
    #config.saving_path = './results/dgcnn'
    config.checkpoint_gap = 50


    # Training from scratch
    trainer = ModelTrainerDGCNN(net, config, on_gpu=True)
    trainer.train(net, train_loader, val_loader, config)


    # Training with pretrained weights
    # chkp_path = './results/dgcnn_semseg_pretrained/model_1.t7'
    # trainer = ModelTrainerDGCNN(net, config, chkp_path=chkp_path, finetune=True, on_gpu=True)
    # trainer.train(net, train_loader, val_loader, config)
