import os
from typing import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra

from models.losses_pointnet import *
from models.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation
from utils.config import bcolors
from utils.debugging import d_print

def same_shape(shape1, shape2):
    if len(shape1) != len(shape2):
        return False
    for i in range(len(shape1)):
        if shape1[i] != shape2[i]:
            return False
    return True

class PointNet2(nn.Module):
    def __init__(
        self,
        lbl_values, 
        ign_lbls, 
        pre_train=False, 
        class_w = [],
        first_features_dim=256,
        free_dim = 4,
        pretreianed_weights = False, #AB: load pretrained weights for backbone and heads
        ):
        super(PointNet2, self).__init__()
        self.pre_train = pre_train
        self.class_w = class_w
        self.first_features_dim  = first_features_dim
        self.free_dim = free_dim
        self.pretreianed_weights = pretreianed_weights
        #AB: variable is used in the loss function 
        self.valid_labels = np.sort([c for c in lbl_values if c not in ign_lbls])
        # Choose segmentation loss
        if len(self.class_w) > 0:
            self.class_w = torch.from_numpy(np.array(self.class_w, dtype=np.float32))
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_w, ignore_index=-1)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        # self.k = num_class
        self.C = len(lbl_values) - len(ign_lbls)
        out_dim = 256 #AB: make the size of the output of PointNet match the output size of KPConv

        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 9 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, out_dim, 1) #AB: modify to get the same output for KPCONV to pass through heads

        # add the prediction heads
        from models.blocks import UnaryBlock
        self.head_mlp = UnaryBlock(out_dim, self.first_features_dim, False, 0)
        self.head_var = UnaryBlock(self.first_features_dim, out_dim + self.free_dim, False, 0)
        self.head_softmax = UnaryBlock(self.first_features_dim, self.C, False, 0)
        self.head_center = UnaryBlock(self.first_features_dim, 1, False, 0, False)
        self.sigmoid = nn.Sigmoid()

        if self.pretreianed_weights:
            self._load_pretrained_weights()

    def _load_pretrained_weights(self):
        """ Load pretrained weights for net, do it for the finetuning task
        backbone weights will be loaded from https://github.com/yanx27/Pointnet_Pointnet2_pytorch
        heads weights will be loaded from 4D-Panoptic Segmentation check point
        """
        BACKBONE_WEIGHTS_PATH = hydra.utils.to_absolute_path("./pretrained_weights/pointnet2/best_model.pth")
        HEADS_WEIGHTS_PATH = hydra.utils.to_absolute_path("./results/Log_2020-10-06_16-51-05/checkpoints/current_chkp.tar")
        # check valid path
        if not os.path.exists(BACKBONE_WEIGHTS_PATH):
            d_print(BACKBONE_WEIGHTS_PATH, bcolors.FAIL)
            raise ValueError(" above Path doesn't exist")
        if not os.path.exists(HEADS_WEIGHTS_PATH):
            d_print(HEADS_WEIGHTS_PATH, bcolors.FAIL)
            raise ValueError(" above Path doesn't exist")
        # load backbone weights
        # loaded_state_dict = torch.load(BACKBONE_WEIGHTS_PATH, map_location=torch.device('cpu'))['model_state_dict']
        loaded_state_dict = OrderedDict()
        # load KPConv weights which contains heads weights
        heads_loaded_sate_dict = torch.load(HEADS_WEIGHTS_PATH, map_location=torch.device('cpu'))['model_state_dict']
        # get heads weights
        heads_state_dict = OrderedDict()
        for k,v in heads_loaded_sate_dict.items():
            if "head" in k:
                heads_state_dict[k] = v
        # add heads weights to the backbone loaded weights
        loaded_state_dict.update(heads_state_dict)
        # get current architecture weights dictionary
        model_state_dict = self.state_dict()
        n, n_total = 0, len(model_state_dict.keys())
        updated_state_dict = OrderedDict()

        # get list of the unexpected keys
        unexpected_keys = []
        size_mismatch = []
        for k, v in loaded_state_dict.items():
            if k in model_state_dict.keys():
                if same_shape(v.shape, model_state_dict[k].shape):
                    updated_state_dict[k] = v
                    n += 1
                else:
                    size_mismatch.append(k)
            else:
                unexpected_keys.append(k)
        self.load_state_dict(updated_state_dict, strict = False)
        # get list of the missing keys
        missing_keys = [key for key in model_state_dict.keys() if key not in loaded_state_dict.keys()]

        # print the results of laoding
        if n == n_total: 
            color = bcolors.OKGREEN 
        else: 
            color = bcolors.WARNING
        d_print("loaded {}/{} from the backbone weights".format(n, n_total), color)
        d_print("Missing keys are : {}".format(missing_keys), bcolors.FAIL)
        d_print("Unexpected keys are: {}".format(unexpected_keys), bcolors.FAIL)
        d_print("size mismatch keys are: {}".format(size_mismatch), bcolors.FAIL)


    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)

        x = x.transpose(2,1).contiguous()
        # x = F.log_softmax(x, dim=1)
        # x = x.permute(0, 2, 1)
        # return x, l4_points
        f = self.head_mlp(x, None)
        c = self.head_center(f, None)
        c = self.sigmoid(c)
        v = self.head_var(f, None)
        v = F.relu(v)
        x = self.head_softmax(f, None)

        return x, c, v, f


# class get_loss(nn.Module):
#     def __init__(self):
#         super(get_loss, self).__init__()
#     def forward(self, pred, target, trans_feat, weight):
#         total_loss = F.nll_loss(pred, target, weight=weight)

#         return total_loss

    def semantic_seg_metric(self, outputs, labels):
        """ semantic segmentation metric calculation

        Args:
            outputs: sofrmax outputs from the segmentation head
            labels: ground truth semantic labels for each point
        Returns:
            iou_metrics: dictionary contains the average IoU and IoU for each class
        """
        
        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        outputs = outputs.view(-1,19)
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
        # d_print(outputs.shape)
        # d_print(centers_p.shape)
        # d_print(variances.shape)
        # d_print(embeddings.shape)
        # d_print(labels.shape)
        # d_print(ins_labels.shape)
        # d_print(centers_gt.shape)
        # d_print(points.shape)
        # d_print(times.shape)
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
        outputs = outputs.view(-1,19)
        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        labels = labels.view(-1,)
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        predicted = torch.argmax(outputs, dim=1)
        total = target.size(0)
        correct = (predicted == target).sum().item()

        return correct / total

if __name__ == '__main__':
    import  torch
    model = get_model(13)
    xyz = torch.rand(6, 9, 2048)
    (model(xyz))