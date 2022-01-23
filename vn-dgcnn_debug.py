import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np

from datasets.semantic_kitti_dataset import SemanticKittiDataSet
from models.vnDGCNN_sem_seg import vnDGCNN
from utils.trainer_vnDGCNN import ModelTrainervnDGCNN
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
    
    DATASET_PATH = './data'

 
    train_set = SemanticKittiDataSet(path=DATASET_PATH, set='train',num_samples=80, augmentation='aligned',verbose=False)
    val_set = SemanticKittiDataSet(path=DATASET_PATH, set='val',num_samples=16, augmentation='z',verbose=False)
    train_loader = DataLoader(train_set, batch_size= 4, num_workers=4, shuffle=False, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size= 4, num_workers=4, shuffle=False, pin_memory=True)

    net=vnDGCNN(train_set.label_values, train_set.ignored_labels, input_feature_dims=3)
    num_parameters = get_model_parameters(net)
    print('Number of model parameters:', num_parameters)

    config = Config()
    config.on_gpu = True
    config.max_epoch = 1000
    config.checkpoint_gap = 100
    config.lr_scheduler = False      # multistep scheduler: milestones=[200, 400, 600], gamma=0.45
    for lr in [0.1, 0.01, 0.001]:
        config.learning_rate = lr   
        config.saving_path = './results/vndgcnn/Expriments0119-'+'4DPLSloss-80/I-z/'+str(config.learning_rate)
        
        trainer = ModelTrainervnDGCNN(net, config, finetune=True, on_gpu=config.on_gpu)
        trainer.train_overfit_4D(config, net, train_loader, val_loader, loss_type='4DPLSloss')#4DPLSloss, CEloss


