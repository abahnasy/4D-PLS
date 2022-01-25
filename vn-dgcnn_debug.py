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

 
    train_set = SemanticKittiDataSet(path=DATASET_PATH, set='train',num_samples=16, augmentation='aligned',verbose=False)
    val_set = SemanticKittiDataSet(path=DATASET_PATH, set='val',num_samples=16, augmentation='z',verbose=False)
    train_loader = DataLoader(train_set, batch_size= 4, num_workers=4, shuffle=False, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size= 4, num_workers=1, shuffle=False, pin_memory=True)

    net=vnDGCNN(train_set.label_values, train_set.ignored_labels, input_feature_dims=3)
    # net = VNDGCNN(train_set.label_values,train_set.ignored_labels)
    num_parameters = get_model_parameters(net)
    print('Number of model parameters:', num_parameters)

    config = Config()
    config.on_gpu = True
    config.max_epoch = 1000
    config.checkpoint_gap = 100
    config.val_pls = False
    config.lr_scheduler = True      # lr search: 1e-4 - 5 LinearLR / multistep scheduler: milestones=[200, 400, 600], gamma=0.45
    config.learning_rate = 5   
    # config.saving_path = './results/vndgcnn/Expriments0124/'+'vndgcnn_lr_search'
    
    for batch in train_loader:
        sample_gpu = batch
        centers = sample_gpu['in_fts'][:,:,4:8]
        times = sample_gpu['in_fts'][:,:,8]

        outputs, centers_output, var_output, embedding = net(sample_gpu['in_fts'][:,:,:3])

        loss = net.loss(outputs, centers_output, var_output, embedding, 
                        sample_gpu['in_lbls'], sample_gpu['in_slbls'], centers, sample_gpu['in_pts'], times)
        break



