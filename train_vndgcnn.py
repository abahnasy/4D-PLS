import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np

from datasets.semantic_kitti_dataset import SemanticKittiDataSet
from models.vnDGCNN_sem_seg import vnDGCNN
from models.vnDGCNN_ref import VNDGCNN
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

    net=vnDGCNN(train_set.label_values, train_set.ignored_labels)
    # net = VNDGCNN(train_set.label_values,train_set.ignored_labels)
    num_parameters = get_model_parameters(net)
    print('Number of model parameters:', num_parameters)

    config = Config()
    config.on_gpu = True
    config.max_epoch = 3000
    config.checkpoint_gap = 100
    config.val_pls = False
    config.learning_rate = 0.1   
    config.lr_scheduler = False      
    config.saving_path = './results/vndgcnn/Experiments0126_debug/'+'I'+'_grad_clip'
    # chkp_path = './results/vndgcnn/Experiments0125_80/old_checkpoints/checkpoints/chkp_0200.tar'
    # chkp_path = './results/vndgcnn/Experiments0125_80/I_cos/checkpoints/best_chkp.tar'
    chkp_path = None
    config.resume_training = False
    config.grad_clip_norm = -100.0
    
    trainer = ModelTrainervnDGCNN(net, config, chkp_path=chkp_path, resume_training=config.resume_training, on_gpu=config.on_gpu)
    trainer.train_overfit_4D(config, net, train_loader, val_loader, loss_type='4DPLSloss')#4DPLSloss, CEloss
    
    # # Learning rate search
    # for lr in [100, 10, 1, 1e-1, 1e-2, 1e-3]:
    #     config.learning_rate = lr   
    #     config.saving_path = './results/vndgcnn/Expriments0124-'+'lr_search-16/'+str(config.learning_rate)
        
    #     trainer = ModelTrainervnDGCNN(net, config, chkp_path=None, resume_training=None, on_gpu=config.on_gpu)
    #     trainer.train_overfit_4D(config, net, train_loader, val_loader, loss_type='4DPLSloss')#4DPLSloss, CEloss

