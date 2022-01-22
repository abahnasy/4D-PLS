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

 
    train_set = SemanticKittiDataSet(path=DATASET_PATH, set='train',num_samples=4, augmentation='aligned',verbose=False)
    val_set = SemanticKittiDataSet(path=DATASET_PATH, set='val',num_samples=4, augmentation='z',verbose=False)
    train_loader = DataLoader(train_set, batch_size= 4, num_workers=4, shuffle=False, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size= 4, num_workers=4, shuffle=False, pin_memory=True)

    net=vnDGCNN(train_set.label_values, train_set.ignored_labels, input_feature_dims=3)
    num_parameters = get_model_parameters(net)
    print('Number of model parameters:', num_parameters)

    config = Config()
    config.on_gpu = True
    config.learning_rate = 0.1
    config.max_epoch = 50000
    config.checkpoint_gap = 50
    config.lr_scheduler = False      # multistep scheduler: milestones=[200, 400, 600], gamma=0.45
    config.saving_path = './results/vndgcnn/Expriments0119-CEloss/'+'I-z'
    
    trainer = ModelTrainervnDGCNN(net, config, finetune=True, on_gpu=config.on_gpu)
    trainer.train_overfit_4D(config, net, train_loader, val_loader, loss_type='CEloss')#4DPLSloss, CEloss

    # for batch in train_loader:
    #     # move to device (GPU)
    #     sample_gpu ={}
    #     if config.on_gpu and torch.cuda.is_available():
    #         print('On GPU')
    #         device = torch.device("cuda:0")
    #     else:
    #         print('On CPU')
    #         device = torch.device("cpu")
    #     if 'cuda' in device.type:
    #         for k, v in batch.items():
    #             sample_gpu[k] = v.to(device)
    #     else:
    #         sample_gpu = batch

    #     outputs, centers_output, var_output, embedding = net(sample_gpu['in_fts'][:,:,:3])
        
    #     print(outputs.size())

    #     break
