#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM

Modified by 
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@Time: 2020/3/9 9:32 PM
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from models.losses_pointnet import *



def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # idx: (batch_size, num_points, k)    indices of k nearest points in each point cloud
    return idx


def get_graph_feature(x, k=20, idx=None, dim3=False): # x: (batch_size, features, num_points)
    batch_size = x.size(0)
    num_points = x.size(2)
    # x = x.view(batch_size, -1, num_points)

    if idx is None:
        if dim3 == True:
            idx = knn(x, k=k)   # idx: (batch_size, num_points, k)
        else:
            idx = knn(x[:, :3, :], k=k) # Need to check the dimensions of data given by the dataloader
    if x.is_cuda: 
        device = torch.device('cuda')  
    else:
        device = torch.device('cpu')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base # number all points from 0 to batch_size*num_points

    idx = idx.view(-1)  # (batch_size * num_points * k)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature      # (batch_size, 2*num_dims, num_points, k)


class Transform_Net(nn.Module):
    def __init__(self):
        super(Transform_Net, self).__init__()
        # self.args = args
        # self.k = 3

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3*3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)                       # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)            # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x


class Transform_Net_s(nn.Module):
    def __init__(self):
        super(Transform_Net_s, self).__init__()

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(512)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 512, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(512, 256, bias=False)
        self.bn3 = nn.BatchNorm1d(256)
        self.linear2 = nn.Linear(256, 128, bias=False)
        self.bn4 = nn.BatchNorm1d(128)

        self.transform = nn.Linear(128, 3*3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)                       # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)            # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x




class DGCNN_semseg(nn.Module):
    def __init__(
        self, 
        lbl_values,
        ign_lbls,
        pretrained=False,
        input_feature_dims=4, # number of input feature deminsions
        first_features_dim=256, # output dimension
        free_dim=4,
        k=20, # number of KNN neighbors
        #dropout=0.,
        ):
        super(DGCNN_semseg, self).__init__()


        self.valid_labels = np.sort([c for c in lbl_values if c not in ign_lbls])

        self.C = len(lbl_values) - len(ign_lbls)
        self.pre_train = pretrained
        self.k = k
        emb_dims=1024 # dimension of embeddings after the EdgeConv layers in DGCNN
        self.first_features_dim = first_features_dim
        self.free_dim = free_dim
        out_dim = 256
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        
        self.transform_net = Transform_Net_s()#Transform_Net()

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(emb_dims)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(out_dim)

        self.conv1 = nn.Sequential(nn.Conv2d(input_feature_dims*2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(emb_dims+64*3, 512, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, out_dim, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        # self.conv9 = nn.Conv1d(256, self.C, kernel_size=1, bias=False)
        # self.dp1 = nn.Dropout(p=dropout)

        # add the prediction heads
        from models.blocks import UnaryBlock
        self.head_mlp = UnaryBlock(out_dim, self.first_features_dim, False, 0)
        self.head_var = UnaryBlock(self.first_features_dim, out_dim + self.free_dim, False, 0)
        self.head_softmax = UnaryBlock(self.first_features_dim, self.C, False, 0)
        self.head_center = UnaryBlock(self.first_features_dim, 1, False, 0, False)

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = torch.transpose(x, 1, 2) #(batch_size, num_points, feature_dims) -> (batch_size, feature_dims, num_points)
        batch_size = x.size(0)
        num_points = x.size(2)

        # STN layer
        x0 = get_graph_feature(x[:,:3,:], k=self.k)     # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        t = self.transform_net(x0)                      # (batch_size, 3, 3)
        t_4 = torch.eye(4,4).repeat(batch_size,1,1)
        t_4[:,:3,:3] = t
        x = x.transpose(2, 1)                           # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x = torch.bmm(x, t_4)                           # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x = x.transpose(2, 1)                           # (batch_size, num_points, 3) -> (batch_size, 3, num_points)

        x = get_graph_feature(x, k=self.k, dim3=False)  # (batch_size, feature_dims, num_points) -> (batch_size, feature_dims*2, num_points, k)
        x = self.conv1(x)                               # (batch_size, feature_dims*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                               # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]            # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)          # (batch_size, emb_dims, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, emb_dims+64*3, num_points)

        x = self.conv7(x)                       # (batch_size, emb_dims+64*3, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)                       # (batch_size, 512, num_points) -> (batch_size, first_features_dim, num_points)
        # x = self.dp1(x)
        # x = self.conv9(x)
        x = torch.transpose(x, 1, 2).contiguous()

        # Head of network
        f = self.head_mlp(x, None)
        c = self.head_center(f, None)
        c = self.sigmoid(c)
        v = self.head_var(f, None)
        v = F.relu(v)
        x = self.head_softmax(f, None)

        return x, c, v, f


    def cross_entropy_loss(self, outputs, labels):
        outputs = outputs.contiguous().view(-1,19)
        labels = labels.view(-1,)
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        return self.criterion(outputs, target)


    def semantic_seg_metric(self, outputs, labels):
        """ semantic segmentation metric calculation

        Args:
            outputs: sofrmax outputs from the segmentation head
            labels: ground truth semantic labels for each point
        Returns:
            iou_metrics: dictionary contains the average IoU and IoU for each class
        """
        
        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        outputs = outputs.contiguous().view(-1,19)
        labels = labels.view(-1,)
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i
        
        # Adjust the shape of the inputs
        
        # outputs = torch.squeeze(outputs, dim=0).cpu()
        preds = torch.argmax(outputs, dim=1)
        # target = target.unsqueeze(0).cpu()
        # target = target.transpose(0,1).squeeze() #AB: move from [1,N] -> [N,]
        
        # create a confusion matrix
        confusion_matrix = torch.zeros(self.C, self.C)
        # d_print("....")
        # d_print(preds.shape)
        # d_print(target.shape)
        for pred_class in range(0,self.C):
            for target_class in range(0, self.C):
                confusion_matrix[pred_class][target_class] = ((preds == pred_class) & (target == target_class)).sum().int()
        # d_print(confusion_matrix)
        intersection = torch.diag(confusion_matrix)
        union = confusion_matrix.sum(0) + confusion_matrix.sum(1) - intersection
        
        scores = intersection.float() / union.float()
        
        # return class IoUs
        return scores


    def loss(self, outputs, centers_p, variances, embeddings, 
                labels, ins_labels, centers_gt, points=None, times=None):
            """
            Args:
                outputs: [B,N,19]
                centers_p: [B,N,1]
                variances: [B,N,260]
                embeddings: [B,N,256]
                labels: [B,N,1]
                ins_labels: [B,N,1]
                centers_gt: [B,N,4]
                points: [B,N,3]
                times: [B,N]
            Returns:
                loss: total loss value
            """
           
            batch_size = outputs.shape[0]
            num_points = outputs.shape[1]
            

            # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
            target = - torch.ones_like(labels)
            for b in range(batch_size):
                for i, c in enumerate(self.valid_labels):
                    target[b][labels[b,:] == c] = i
            self.output_loss = self.criterion(outputs.view(-1,19), target.view(-1,).long())
            self.output_loss /= batch_size
            
            # reshape centers_gt
            centers_gt = centers_gt.view(num_points*batch_size, -1)
            weights = (centers_gt[:, 0] > 0) * 99 + (centers_gt[:, 0] >= 0) * 1
            # Reshape
            centers_p = centers_p.view(num_points*batch_size)
            self.center_loss = weighted_mse_loss(centers_p, centers_gt[:, 0], weights)


            if not self.pre_train:
                # d_print("shape of embeddings {}".format(embeddings.shape))
                #AB: remove batch dimensions from embeddings and varainces
                # embeddings = torch.squeeze(embeddings, dim=0)
                # variances = torch.squeeze(variances, dim=0)
                # Reshape
                embeddings = embeddings.view(batch_size*num_points, -1) # [B,N,256] -> [B*N,256]
                ins_labels = ins_labels.squeeze().view(batch_size*num_points) # [B,N,1] -> [B*N,]
                self.instance_half_loss = instance_half_loss(embeddings, ins_labels)
                self.instance_half_loss /= batch_size
                # Reshape
                variances = variances.view(batch_size*num_points, -1) # [B,N,260] -> [B*N,260]
                
                self.instance_loss = iou_instance_loss(
                    centers_p, embeddings, variances, ins_labels, 
                    points.view(batch_size*num_points, -1),
                    times.view(batch_size*num_points, 1)
                    )
                self.instance_loss /= batch_size
                self.variance_loss = variance_smoothness_loss(variances, ins_labels)
                self.variance_loss /= batch_size
                self.variance_l2 = variance_l2_loss(variances, ins_labels)
                self.variance_l2 /= batch_size
            
            #AB: comment out regularizer loss, related to KPConv only !
            self.reg_loss = torch.tensor(0.0).to(embeddings.device)
            # Regularization of deformable offsets
            # if self.deform_fitting_mode == 'point2point':
            #     self.reg_loss = p2p_fitting_regularizer(self)
            # elif self.deform_fitting_mode == 'point2plane':
            #     raise ValueError('point2plane fitting mode not implemented yet.')
            # else:
            #     raise ValueError('Unknown fitting mode: ' + self.deform_fitting_mode)

            # Combined loss
            #return self.instance_loss + self.variance_loss
            
            return self.output_loss + self.reg_loss + self.center_loss + self.instance_loss*0.1+ self.variance_loss*0.01
            
    
    def accuracy(self, outputs, labels):
        """
        outputs: [B,N,19]
        labels: [B,N,1]
        """
        #AB: remove batch dims
        outputs = outputs.contiguous().view(-1,19)
        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        labels = labels.view(-1,)
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        predicted = torch.argmax(outputs, dim=1)
        total = target.size(0)
        correct = (predicted == target).sum().item()

        return correct / total



