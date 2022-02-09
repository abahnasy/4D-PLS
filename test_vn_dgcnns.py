""" Initial script to test pre trained vn dgcnns models
the output results should be saved in ./test directory
"""
from multiprocessing.sharedctypes import Value
import os

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from datasets.semantic_kitti_dataset import SemanticKittiDataSet
from torch.utils.data import DataLoader
from models.vn_dgcnn_sem_seg import VNDGCNN
from utils.tester_vn_dgcnn import ModelTesterVNDGCNN

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')

@hydra.main(config_path="conf/test", config_name="config_vn_dgcnn")
def main(cfg : DictConfig) -> None:
    
    DATASET_PATH = hydra.utils.to_absolute_path('data')
    # get ckpt from folder
    best_ckpt_path = os.path.join(cfg.ckpt_path, 'checkpoints', 'best_chkp.tar')
    current_ckpt_path = os.path.join(cfg.ckpt_path, 'checkpoints', 'current_chkp.tar')
    if os.path.exists(hydra.utils.to_absolute_path(best_ckpt_path)):
        CKPT_PATH = best_ckpt_path
    elif os.path.exists(hydra.utils.to_absolute_path(best_ckpt_path)):
        CKPT_PATH = current_ckpt_path
    else:
        raise ValueError("cannot find ckpt weights")
    

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
    assert cfg.val_loader.batch_size == 1, "Val should be done using batch of size 1 only !"
    assert cfg.val_loader.shuffle == False, "Val data should be handled in sequential manner !"
    val_loader = DataLoader(
        val_set, 
        shuffle = cfg.val_loader.shuffle, # should be false
        batch_size= cfg.val_loader.batch_size, # should be only 1
        num_workers= cfg.val_loader.num_workers, # should be only 1
        pin_memory=True
    )

    # Model Definition
    net = VNDGCNN(
        val_set.label_values,
        val_set.ignored_labels,
        pretreianed_weights=cfg.model.pretrained_weights,
        freeze_head_weights=cfg.model.freeze_head_weights,
        ckpt_path = CKPT_PATH, # abs path will be resolved inside the model init function !
        class_w=cfg.trainer.class_w
    ).to(device)

    tester = ModelTesterVNDGCNN(net) # model weights should be loaded when the model is being initialized !
    tester.panoptic_4d_test(net, val_loader, cfg.tester)


if __name__ == '__main__':

    main()
    

    
    # create dataloader

    # create test module

