import importlib
import os, time

import numpy as np
import torch
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf

from models.vn_dgcnn_sem_seg import VNDGCNN
from utils.trainer_vn_dgcnn import ModelTrainerVNDGCNN
from datasets.semantic_kitti_dataset import SemanticKittiDataSet
from utils.debugging import d_print, write_pc, count_parameters, seed_torch

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')


@hydra.main(config_path="conf", config_name="config_vn_dgcnn")
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
    if cfg.saving:
        cfg.saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
        cfg.saving_path = hydra.utils.to_absolute_path(cfg.saving_path)
        d_print(os.path.abspath(cfg.saving_path))
    # cfg.trainer.lr_decays = {i: 0.1 ** (1 / 200) for i in range(1, cfg.trainer.max_epoch)}
    print(OmegaConf.to_yaml(cfg))
    # prepare dataset and loaders
    DATASET_PATH = hydra.utils.to_absolute_path('data')
    train_set = SemanticKittiDataSet(
        path=DATASET_PATH, 
        set='train', 
        num_samples=cfg.train_dataset.num_samples,
        augmentation=cfg.train_dataset.augmentation,
        requested_sequences=cfg.train_dataset.requested_sequences,
        balance_classes=cfg.train_dataset.balance_classes,
        )
    val_set = SemanticKittiDataSet(
        path=DATASET_PATH, 
        set='val', 
        num_samples=cfg.val_dataset.num_samples,
        augmentation=cfg.val_dataset.augmentation,
        requested_sequences=cfg.val_dataset.requested_sequences,
        balance_classes=cfg.val_dataset.balance_classes,
        saving_path=cfg.saving_path,
        in_R = cfg.val_dataset.in_R
        )
    train_loader = DataLoader(
        train_set, shuffle = cfg.train_loader.shuffle, 
        batch_size= cfg.train_loader.batch_size, 
        num_workers=4, 
        pin_memory=True)

    assert cfg.val_loader.batch_size == 1, "Val should be done using batch of size 1 only !"
    assert cfg.val_loader.shuffle == False, "Val data should be handled in sequential manner !"
    
    val_loader = DataLoader(
        val_set, shuffle = cfg.val_loader.shuffle, 
        batch_size= cfg.val_loader.batch_size, 
        num_workers=4, 
        pin_memory=True)

    # ----------------------------------------------------------------- #

    # net_class = importlib.import_module(".{}".format(cfg.trainer.net), 'models.vn_dgcnn_sem_seg')
    net = VNDGCNN(
        train_set.label_values,
        train_set.ignored_labels,
        pretreianed_weights=cfg.model.pretrained_weights,
        freeze_head_weights=cfg.model.freeze_head_weights,
        class_w=cfg.trainer.class_w
        ).to(device)
    # net = VNDGCNN_v1(train_set.label_values, train_set.ignored_labels, pretreianed_weights=False).to(device)
    # print detailed table for parameters
    count_parameters(net)
    

    d_print("launcing the training")
    trainer = ModelTrainerVNDGCNN(net, cfg.trainer, chkp_path=chosen_chkp)
    if cfg.trainer.style == 'train':
        trainer.train(net, train_loader, val_loader, cfg.trainer)
    elif cfg.trainer.style == 'overfit':
        trainer.overfit4D(net, train_loader, cfg.trainer)
    else:
        raise ValueError("unknow style")
    
    



if __name__ == '__main__':
    seed_torch()
    my_app()

"""
Steps
- modify the input features to be the xyz and reflection only, check first the inputs of kpconv
- define the backbone only and pass the data from dataloader to the backbone and get the correct bacbone outputs
- define the heads and assert check the correct outputs from the heads
- define the loss and make sure the backward path is up and running
- finalize and launch training
"""
