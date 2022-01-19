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
    
    DATASET_PATH = './data'

 
    train_set = SemanticKittiDataSet(path=DATASET_PATH, set='train',num_samples=16, augmentation='z',verbose=False)
    val_set = SemanticKittiDataSet(path=DATASET_PATH, set='val',num_samples=16, augmentation='z',verbose=False)
    train_loader = DataLoader(train_set, batch_size= 4, num_workers=4, shuffle=False, pin_memory=True)
    val_loader = DataLoader(train_set, batch_size= 4, num_workers=4, shuffle=False, pin_memory=True)

    net=DGCNN_semseg(train_set.label_values, train_set.ignored_labels, input_feature_dims=4)
    num_parameters = get_model_parameters(net)
    print('Number of model parameters:', num_parameters)

    config = Config()
    config.on_gpu = True
    config.learning_rate = 0.1
    config.max_epoch = 50000
    config.checkpoint_gap = 50
    config.lr_scheduler = False      # multistep scheduler: milestones=[200, 400, 600], gamma=0.45
    config.saving_path = './results/dgcnn/Expriments0119/'+'z-z'

    # Pretrained weights of both dgcnn and loss heads
    chkp_path = './results/dgcnn_semseg_pretrained/model_1.t7'
    trainer = ModelTrainerDGCNN(net, config, chkp_path=chkp_path, finetune=True, on_gpu=config.on_gpu)
    trainer.train_overfit_4D(config, net, train_loader, val_loader, loss_type='4DPLSloss')
