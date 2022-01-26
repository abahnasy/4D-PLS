#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Congyue Deng
@Contact: congyue@stanford.edu
@File: model_equi.py
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

EPS = 1e-6

def knn(x, k):                                      # x: (batch_size, 1, feature_dims, num_points)
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # idx: (batch_size, num_points, k)    indices of k nearest points in each point cloud
    return idx


def get_graph_feature(x, k=20, idx=None, dim3=True): # x: (batch_size, 1, feature_dims, num_points)
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
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
    num_dims = num_dims // 3


    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3) 
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 4, 1, 2).contiguous()
    return feature      # (batch_size, 2*num_dims, 3, num_points, k)


class VNLeakyReLU(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.2):
        super(VNLeakyReLU, self).__init__()
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        dotprod = (x*d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d*d).sum(2, keepdim=True)
        x_out = self.negative_slope * x + (1-self.negative_slope) * (mask*x + (1-mask)*(x-(dotprod/(d_norm_sq+EPS))*d))
        return x_out


class VNLinearLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, negative_slope=0.2):
        super(VNLinearLeakyReLU, self).__init__()
        self.dim = dim
        self.negative_slope = negative_slope
        
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
        self.batchnorm = VNBatchNorm(out_channels, dim=dim)
        
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, out_channels, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Linear
        p = self.map_to_feat(x.transpose(1,-1)).transpose(1,-1)
        # BatchNorm
        p = self.batchnorm(p)
        # LeakyReLU
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        dotprod = (p*d).sum(2, keepdims=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d*d).sum(2, keepdims=True)
        x_out = self.negative_slope * p + (1-self.negative_slope) * (mask*p + (1-mask)*(p-(dotprod/(d_norm_sq+EPS))*d))
        return x_out


class VNBatchNorm(nn.Module):
    def __init__(self, num_features, dim):
        super(VNBatchNorm, self).__init__()
        self.dim = dim
        if dim == 3 or dim == 4:
            self.bn = nn.BatchNorm1d(num_features)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # norm = torch.sqrt((x*x).sum(2))
        norm = torch.norm(x, dim=2) + EPS
        norm_bn = self.bn(norm)
        norm = norm.unsqueeze(2)
        norm_bn = norm_bn.unsqueeze(2)
        x = x / norm * norm_bn
        
        return x


class VNMaxPool(nn.Module):
    def __init__(self, in_channels):
        super(VNMaxPool, self).__init__()
        self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        dotprod = (x*d).sum(2, keepdims=True)
        idx = dotprod.max(dim=-1, keepdim=False)[1]
        index_tuple = torch.meshgrid([torch.arange(j) for j in x.size()[:-1]]) + (idx,)
        x_max = x[index_tuple]
        return x_max


def mean_pool(x, dim=-1, keepdim=False):
    return x.mean(dim=dim, keepdim=keepdim)


class VNStdFeature(nn.Module):
    def __init__(self, in_channels, dim=4, normalize_frame=False, share_nonlinearity=False, negative_slope=0.2):
        super(VNStdFeature, self).__init__()
        self.dim = dim
        self.normalize_frame = normalize_frame
        
        self.vn1 = VNLinearLeakyReLU(in_channels, in_channels//2, dim=dim, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)
        self.vn2 = VNLinearLeakyReLU(in_channels//2, in_channels//4, dim=dim, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)
        if normalize_frame:
            self.vn_lin = nn.Linear(in_channels//4, 2, bias=False)
        else:
            self.vn_lin = nn.Linear(in_channels//4, 3, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        z0 = x
        z0 = self.vn1(z0)
        z0 = self.vn2(z0)
        z0 = self.vn_lin(z0.transpose(1, -1)).transpose(1, -1)
        
        if self.normalize_frame:
            # make z0 orthogonal. u2 = v2 - proj_u1(v2)
            v1 = z0[:,0,:]
            #u1 = F.normalize(v1, dim=1)
            v1_norm = torch.sqrt((v1*v1).sum(1, keepdims=True))
            u1 = v1 / (v1_norm+EPS)
            v2 = z0[:,1,:]
            v2 = v2 - (v2*u1).sum(1, keepdims=True)*u1
            #u2 = F.normalize(u2, dim=1)
            v2_norm = torch.sqrt((v2*v2).sum(1, keepdims=True))
            u2 = v2 / (v2_norm+EPS)

            # compute the cross product of the two output vectors        
            u3 = torch.cross(u1, u2)
            z0 = torch.stack([u1, u2, u3], dim=1).transpose(1, 2)
        else:
            z0 = z0.transpose(1, 2)
        
        if self.dim == 4:
            x_std = torch.einsum('bijm,bjkm->bikm', x, z0)
        elif self.dim == 3:
            x_std = torch.einsum('bij,bjk->bik', x, z0)
        elif self.dim == 5:
            x_std = torch.einsum('bijmn,bjkmn->bikmn', x, z0)
        
        return x_std, z0


class vnDGCNN(nn.Module):
    def __init__(
        self, 
        lbl_values,
        ign_lbls,
        pretrained=False,
        first_features_dim=256, # output dimension
        free_dim=4,
        k=20, # number of KNN neighbors
        #dropout=0.,
        class_weights=None,
        ):
        super(vnDGCNN, self).__init__()


        self.valid_labels = np.sort([c for c in lbl_values if c not in ign_lbls])

        self.C = len(lbl_values) - len(ign_lbls)
        self.pre_train = pretrained
        self.k = k
        emb_dims=1024 # dimension of embeddings after the EdgeConv layers in DGCNN
        self.first_features_dim = first_features_dim
        self.free_dim = free_dim
        out_dim = 256
        pooling = 'max'
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
        

        # self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        # self.bn10 = nn.BatchNorm1d(128)
        
        self.conv1 = VNLinearLeakyReLU(2, 64//3)
        self.conv2 = VNLinearLeakyReLU(64//3, 64//3)
        self.conv3 = VNLinearLeakyReLU(64//3*2, 64//3)
        self.conv4 = VNLinearLeakyReLU(64//3, 64//3)
        self.conv5 = VNLinearLeakyReLU(64//3*2, 64//3)
        
        if pooling == 'max':
            self.pool1 = VNMaxPool(64//3)
            self.pool2 = VNMaxPool(64//3)
            self.pool3 = VNMaxPool(64//3)
        elif pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
        
        self.conv6 = VNLinearLeakyReLU(64//3*3, 1024//3, dim=4, share_nonlinearity=True)        # dim: dimension to do batch norm
        self.std_feature = VNStdFeature(1024//3*2, dim=4, normalize_frame=False)
        self.conv8 = nn.Sequential(nn.Conv1d(2235, 256, kernel_size=1, bias=False),
                               self.bn8,
                               nn.LeakyReLU(negative_slope=0.2))
        
        # self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
        #                            self.bn7,
        #                            nn.LeakyReLU(negative_slope=0.2))
        
        # self.dp1 = nn.Dropout(p=0.5)
        self.conv9 = nn.Sequential(nn.Conv1d(256, out_dim, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        # self.dp2 = nn.Dropout(p=0.5)
        # self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
        #                            self.bn10,
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.conv11 = nn.Conv1d(128, out_dim, kernel_size=1, bias=False)

        # add the prediction heads
        from models.blocks import UnaryBlock
        self.head_mlp = UnaryBlock(out_dim, self.first_features_dim, False, 0)
        self.head_var = UnaryBlock(self.first_features_dim, self.first_features_dim + self.free_dim, False, 0)
        self.head_softmax = UnaryBlock(self.first_features_dim, self.C, False, 0)
        self.head_center = UnaryBlock(self.first_features_dim, 1, False, 0, False)

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = torch.transpose(x, 1, 2) #(batch_size, num_points, feature_dims) -> (batch_size, feature_dims, num_points)
        batch_size = x.size(0)
        num_points = x.size(2)
        
        x = x.unsqueeze(1)                                  # (batch_size, feature_dims, num_points) -> (batch_size, 1, feature_dims, num_points)
        
        x = get_graph_feature(x, k=self.k)                  # (batch_size, 1, feature_dims, num_points) -> (batch_size, 2*num_dims, 3, num_points, k)
        x = self.conv1(x)                                   # (batch_size, 2*num_dims, 3, num_points, k) -> (batch_size, 64//3, 3, num_points, k)
        x = self.conv2(x)                                   # (batch_size, 64//3, 3, num_points, k) -> (batch_size, 64//3, 3, num_points, k)
        x1 = self.pool1(x)                                  # (batch_size, 64//3, 3, num_points, k) -> (batch_size, 64//3, 3, num_points)
        
        x = get_graph_feature(x1, k=self.k)                 # (batch_size, 64//3, 3, num_points, k) -> (batch_size, 64//3*2, 3, num_points, k)
        x = self.conv3(x)                                   # (batch_size, 64//3*2, 3, num_points, k) -> (batch_size, 64//3, 3, num_points, k)
        x = self.conv4(x)                                   # (batch_size, 64//3, 3, num_points, k) -> (batch_size, 64//3, 3, num_points, k)
        x2 = self.pool2(x)                                  # (batch_size, 64//3, 3, num_points, k) -> (batch_size, 64//3, 3, num_points)
        
        x = get_graph_feature(x2, k=self.k)                 # (batch_size, 64//3, 3, num_points, k) -> (batch_size, 64//3*2, 3, num_points, k)
        x = self.conv5(x)                                   # (batch_size, 64//3*2, 3, num_points, k) -> (batch_size, 64//3, 3, num_points, k)
        x3 = self.pool3(x)                                  # (batch_size, 64//3, 3, num_points, k) -> (batch_size, 64//3, 3, num_points)
        x123 = torch.cat((x1, x2, x3), dim=1)               # (batch_size, 64//3*3, 3, num_points)

        x = self.conv6(x123)                                # (batch_size, 64//3*3, 3, num_points) -> (batch_size, 1024//3, 3, num_points)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())  # (batch_size, 1024//3, 3, num_points)
        x = torch.cat((x, x_mean), 1)                       # (batch_size, 1024//3*2, 3, num_points)
        x, z0 = self.std_feature(x)                         # (batch_size, 1024//3*2, 3, num_points)
        x123 = torch.einsum('bijm,bjkm->bikm', x123, z0).view(batch_size, -1, num_points)   # (batch_size, 64//3*3*3, num_points)
        x = x.view(batch_size, -1, num_points)              # (batch_size, 1024//3*2*3, num_points)
        x = x.max(dim=-1, keepdim=True)[0]                  # (batch_size, 1024//3*2*3, 1)
        # l = l.view(batch_size, -1, 1)
        # l = self.conv7(l)

        # x = torch.cat((x, l), dim=1)
        x = x.repeat(1, 1, num_points)                      # (batch_size, 1024//3*2*3, num_points)

        x = torch.cat((x, x123), dim=1)                     # (batch_size, 1024//3*2*3 + 64//3*3*3, num_points)

        x = self.conv8(x)
        # x = self.dp1(x)
        x = self.conv9(x)
        # x = self.dp2(x)
        # x = self.conv10(x)
        # x = self.conv11(x)

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
        target = target.to(outputs.device)
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
            # print('weights size:',weights.size())
            # print('values in weights:', torch.unique(weights))
            
            # Reshape
            centers_p = centers_p.view(num_points*batch_size)
            # print('size of centers_p:',centers_p.size())
            # print('max of centers_p:', centers_p.max())
            # print('min of centers_p:', centers_p.min())

            # print('size of centers_gt:',centers_gt.size())
            # print('max of centers_gt:', centers_gt[:, 0].max())
            # print('min of centers_gt:', centers_gt[:, 0].min())
            self.center_loss = weighted_mse_loss(centers_p, centers_gt[:, 0], weights)
            # print('center loss:', self.center_loss.item())

            if not self.pre_train:
                # d_print("shape of embeddings {}".format(embeddings.shape))
                #AB: remove batch dimensions from embeddings and varainces
                # embeddings = torch.squeeze(embeddings, dim=0)
                # variances = torch.squeeze(variances, dim=0)
                # Reshape
                embeddings = embeddings.view(batch_size*num_points, -1) # [B,N,256] -> [B*N,256]
                ins_labels = ins_labels.squeeze().view(batch_size*num_points) # [B,N,1] -> [B*N,] ground truth
                # print('values in ins_labels:', torch.unique(ins_labels))
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



