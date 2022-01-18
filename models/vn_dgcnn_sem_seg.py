# TODO: define the model and the loss functions
"""
"""

import os
from typing import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from models.base_model import BaseModel

from models.losses import *
from models.vn_layers import *
from utils.config import bcolors
from utils.debugging import d_print

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, x_coord=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if x_coord is None: # dynamic knn graph
            idx = knn(x, k=k)
        else:          # fixed knn graph with input point coordinates
            idx = knn(x_coord, k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3) 
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 4, 1, 2).contiguous()
  
    return feature

class VNDGCNN(BaseModel):
    """
    
    """
    def __init__(
        self,
        lbl_values, 
        ign_lbls, 
        pre_train=False, 
        class_w = [],
        first_features_dim=256,
        free_dim = 4,
        pretreianed_weights = False, #AB: load pretrained weights for backbone and heads
        freeze_head_weights = False,
        normal_channel=False,
        ):
        super(VNDGCNN, self).__init__()
        self.pooling = 'mean' #TODO, move to configurations
        self.n_knn = 20 #TODO, move to configurations
        self.pre_train = pre_train
        self.class_w = class_w
        self.first_features_dim  = first_features_dim
        self.free_dim = free_dim
        self.pretreianed_weights = pretreianed_weights
        self.freeze_head_weights = freeze_head_weights
        #AB: variable is used in the loss function 
        self.valid_labels = np.sort([c for c in lbl_values if c not in ign_lbls])
        # Choose segmentation loss
        if len(self.class_w) > 0:
            self.class_w = torch.from_numpy(np.array(self.class_w, dtype=np.float32))
            self.criterion = torch.nn.CrossEntropyLoss(weight=self.class_w, ignore_index=-1)
            d_print("INFO: weighted cross entropy")
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        # self.k = num_class
        self.C = len(lbl_values) - len(ign_lbls)
        out_dim = 256 #AB: make the size of the output of PointNet match the output size of KPConv
        
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)
        
        self.conv1 = VNLinearLeakyReLU(2, 64//3)
        self.conv2 = VNLinearLeakyReLU(64//3, 64//3)
        self.conv3 = VNLinearLeakyReLU(64//3*2, 64//3)
        self.conv4 = VNLinearLeakyReLU(64//3, 64//3)
        self.conv5 = VNLinearLeakyReLU(64//3*2, 64//3)
        
        if self.pooling == 'max':
            self.pool1 = VNMaxPool(64//3)
            self.pool2 = VNMaxPool(64//3)
            self.pool3 = VNMaxPool(64//3)
        elif self.pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
        
        self.conv6 = VNLinearLeakyReLU(64//3*3, 1024//3, dim=4, share_nonlinearity=True)
        self.std_feature = VNStdFeature(1024//3*2, dim=4, normalize_frame=False)
        #AB: changed from 2299 to 2235
        self.conv8 = nn.Sequential(nn.Conv1d(2235, 256, kernel_size=1, bias=False),
                               self.bn8,
                               nn.LeakyReLU(negative_slope=0.2))

        #AB: specific to part segmentation        
        # self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
        #                            self.bn7,
        #                            nn.LeakyReLU(negative_slope=0.2))
        
        self.dp1 = nn.Dropout(p=0.5)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=0.5)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn10,
                                   nn.LeakyReLU(negative_slope=0.2))
        # AB: additional layer compared to the plane implementation, 
        # keep it since it outputs the same feature size needed for the heads !
        self.conv11 = nn.Conv1d(128, out_dim, kernel_size=1, bias=False)

        
        # AB: add the prediction heads
        from models.blocks import UnaryBlock
        self.head_mlp = UnaryBlock(out_dim, self.first_features_dim, False, 0)
        self.head_var = UnaryBlock(self.first_features_dim, out_dim + self.free_dim, False, 0)
        self.head_softmax = UnaryBlock(self.first_features_dim, self.C, False, 0)
        self.head_center = UnaryBlock(self.first_features_dim, 1, False, 0, False)
        self.sigmoid = nn.Sigmoid()

        if self.pretreianed_weights:
            self._load_pretrained_weights()

        if self.freeze_head_weights:
            #TODO:Freeze head weights
            heads = ['head_mlp', 'head_var', 'head_softmax', 'head_center']
            for head in heads:
                d_print("Freezing heads for {}".format(head))
                module = self.__getattr__(head)
                for p in  module.parameters():
                    if p.requires_grad:
                        p.requires_grad = False
                        d_print("freezed", bcolors.OKBLUE)


        def _load_pretrained_weights(self):
            """ Load pretrained weights for net, do it for the finetuning task
            backbone weights will be loaded from TODO
            heads weights will be loaded from 4D-Panoptic Segmentation check point
            """
            raise NotImplementedError
        

    def forward(self, x):
        """
        x: shape [B, feats, num_points]
        """
        assert x.shape[1] == 3, "error in feat dimensions"
        
        batch_size = x.size(0)
        num_points = x.size(2) # [B, feats, num_points] -> [B, 1, feats, num_points]
        
        x = x.unsqueeze(1)
        
        x = get_graph_feature(x, k=self.n_knn)
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = self.pool1(x)
        
        x = get_graph_feature(x1, k=self.n_knn)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = self.pool2(x)
        
        x = get_graph_feature(x2, k=self.n_knn)
        x = self.conv5(x)
        x3 = self.pool3(x)
        
        x123 = torch.cat((x1, x2, x3), dim=1) #AB: concatenate on feature dimension 
        # print("x123", x123.shape)
        
        x = self.conv6(x123)
        # print("after conv6", x.shape)
        #AB: calculation for the invariant layer
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        # print("x_mean after expansion", x_mean.shape)
        # print(x_mean[:,:,:,0])
        # assert torch.allclose(x_mean[:,:,:,0], x_mean[:,:,:,1]) == True
        x = torch.cat((x, x_mean), 1)
        # print("after first concat: ", x.shape)
        x, z0 = self.std_feature(x)
        # print("x shape after std_feat", x.shape)
        # print("z0 shape: ", z0.shape)
        x123 = torch.einsum('bijm,bjkm->bikm', x123, z0).view(batch_size, -1, num_points)
        # print("after multiplying x123 with z0: ", x123.shape)
        x = x.view(batch_size, -1, num_points)
        x = x.max(dim=-1, keepdim=True)[0]

        # l = l.view(batch_size, -1, 1)
        # l = self.conv7(l)

        # x = torch.cat((x, l), dim=1)
        x = x.repeat(1, 1, num_points)
        # print(">>>>>", x.shape)
        x = torch.cat((x, x123), dim=1)

        x = self.conv8(x)
        x = self.dp1(x)
        x = self.conv9(x)
        x = self.dp2(x)
        x = self.conv10(x)
        x = self.conv11(x)

        x = x.transpose(2,1).contiguous()

        f = self.head_mlp(x, None)
        c = self.head_center(f, None)
        c = self.sigmoid(c)
        v = self.head_var(f, None)
        v = F.relu(v)
        x = self.head_softmax(f, None)
        
        # trans_feat = None
        # return x.transpose(1, 2), trans_feat
        return x, c, v, f