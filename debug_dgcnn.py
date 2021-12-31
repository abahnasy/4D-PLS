import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import JaccardIndex

import os
import numpy as np

from models.dgcnn_sem_seg import DGCNN_semseg
from datasets.semantic_kitti_dataset import SemanticKittiDataSet

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')


def cal_metric(pred, target):
    semseg_iou = JaccardIndex(num_classes=20, ignore_index=0)
    pred = pred.squeeze()
    pred = torch.transpose(pred, 0, 1)
    metric = semseg_iou(pred, target)
    return metric


def cal_loss(pred, target):
    target = target.contiguous().view(-1)
    loss = nn.CrossEntropyLoss(ignore_index=0)
    pred = pred.squeeze()
    pred = torch.transpose(pred, 0, 1)
    loss = loss(pred, target)
    return loss


def train_overfit(model, optimizer, samples):
    model.train()
    train_loss = 0.

    in_fts = torch.tensor(samples['in_fts']).unsqueeze(0)
    target = torch.tensor(samples['in_lbls'])
    for epoch in range(1000):
        pred = model(in_fts)
        train_loss = cal_loss(pred, target)
        semseg_iou = cal_metric(pred, target)
    
    return train_loss, semseg_iou


if __name__ == '__main__':

    DATASET_PATH = './data/SemanticKitti'
    train_set = SemanticKittiDataSet(path=DATASET_PATH, set='train')
    val_set = SemanticKittiDataSet(path=DATASET_PATH, set='val')
    train_loader = DataLoader(train_set, batch_size= 4, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size= 4, num_workers=1, pin_memory=True)
    
    lr=0.1

    model=DGCNN_semseg(feature_dims=9,out_dim=20)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    samples = train_set[0]
    # print(samples['in_pts'].shape)
    # print(samples['in_fts'].shape)
    # print(samples['in_lbls'].shape)
    # print(samples['in_slbls'].shape)
    # print(samples['in_pts'][0,:])
    # print(samples['in_fts'][0,:3])
    # print(samples['in_lbls'][:10])
    # print(samples['in_slbls'][:10])
    model_dir = './results/dgcnn_semseg'
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model_1.t7')))



