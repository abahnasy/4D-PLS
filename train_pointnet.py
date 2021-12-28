import importlib
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf

from models.pointnet_sem_seg import PointNet
from models.pointnet2_sem_seg import PointNet2
from utils.trainer_pointnet import ModelTrainerPointNet
from datasets.semantic_kitti_dataset import SemanticKittiDataSet
from utils.debugging import d_print, write_pc

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')


@hydra.main(config_path="conf", config_name="config")
def my_app(cfg : DictConfig) -> None:

    # ----------------------------------------------------------------- #
    # choose previous checkpoint
    previous_training_path =''
    # Choose index of checkpoint to start from. If None, uses the latest chkp
    chkp_idx = None
    if previous_training_path:
        # Find all snapshot in the chosen training folder
        chkp_path = os.path.join('results', previous_training_path, 'checkpoints')
        chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']
        # Find which snapshot to restore
        if chkp_idx is None:
            chosen_chkp = 'current_chkp.tar'
        else:
            chosen_chkp = np.sort(chkps)[chkp_idx]
        chosen_chkp = os.path.join('results', previous_training_path, 'checkpoints', chosen_chkp)
    else:
        chosen_chkp = None
    # ----------------------------------------------------------------- #

    # cfg.trainer.lr_decays = {i: 0.1 ** (1 / 200) for i in range(1, cfg.trainer.max_epoch)}
    print(OmegaConf.to_yaml(cfg))
    # prepare dataset and loaders
    DATASET_PATH = hydra.utils.to_absolute_path('data')
    train_set = SemanticKittiDataSet(path=DATASET_PATH, set='train')
    val_set = SemanticKittiDataSet(path=DATASET_PATH, set='val')
    train_loader = DataLoader(train_set, batch_size= 4, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size= 4, num_workers=1, pin_memory=True)
    
    # ----------------------------------------------------------------- #
    # remove later
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
    
    def inplace_relu(m):
        classname = m.__class__.__name__
        if classname.find('ReLU') != -1:
            m.inplace=True
    # ----------------------------------------------------------------- #
    
    
    
    # MODEL = importlib.import_module('pointnet2_sem_seg')
    # net = PointNet(train_set.label_values, train_set.ignored_labels).to(device)
    net = PointNet2(train_set.label_values, train_set.ignored_labels).to(device)
    # criterion = get_loss().to(device)
    net.apply(inplace_relu)
    # net = net.apply(weights_init) #TODO: check weight init error later !

    trainer = ModelTrainerPointNet(net, cfg.trainer, chkp_path=chosen_chkp)
    trainer.train(net, train_loader, val_loader, cfg.trainer)


if __name__ == '__main__':
    my_app()
    
