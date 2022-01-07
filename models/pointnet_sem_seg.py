import torch
import torch.nn as nn
# import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from torchmetrics import IoU

import numpy as np

from models.pointnet_utils import PointNetEncoder, feature_transform_reguliarzer
from utils.debugging import d_print, write_pc
from utils.config import bcolors

from models.losses_pointnet import *

from models.architectures import p2p_fitting_regularizer

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')

class PointNet(nn.Module):
    def __init__(
        self, 
        lbl_values, 
        ign_lbls, 
        pre_train=False, 
        class_w = [],
        first_features_dim=256,
        free_dim = 4,
        ):
        super(PointNet, self).__init__()
        self.pre_train = pre_train
        self.class_w = class_w # not used since they were loading balanced point cloud
        self.first_features_dim  = first_features_dim
        self.free_dim = free_dim
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
        
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=9)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, out_dim, 1) #AB: 256 instead of the original 128
        # self.conv4 = torch.nn.Conv1d(128, self.C, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(out_dim)

        
        # add the prediction heads
        from models.blocks import UnaryBlock
        self.head_mlp = UnaryBlock(out_dim, self.first_features_dim, False, 0)
        self.head_var = UnaryBlock(self.first_features_dim, out_dim + self.free_dim, False, 0)
        self.head_softmax = UnaryBlock(self.first_features_dim, self.C, False, 0)
        self.head_center = UnaryBlock(self.first_features_dim, 1, False, 0, False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        # x = F.log_softmax(x.view(-1,self.C), dim=-1)
        # x = x.view(batchsize, n_pts, self.C)
        # return x, trans_feat
        # Head of network
        f = self.head_mlp(x, None)
        c = self.head_center(f, None)
        c = self.sigmoid(c)
        v = self.head_var(f, None)
        v = F.relu(v)
        x = self.head_softmax(f, None)

        return x, c, v, f

    # def forward(self, batch, config):
    #     """
    #     Args:
    #         batch: 
    #         config: TODO: remove from this and KPConv models
    #     Returns:
    #     """
    #     # d_print("inside forward function")
    #     #AB: prepare the correct feature inputs by concatenating the features again
    #     #AB: point feature is [Reflectance, c_x, c_y, c_z, t]
    #     #AB: Batch size is always one since we load only one 4D volume per dataloader
        
    #     # d_print("points shape {}".format(batch.points[0].shape))
    #     # for i in range(len(batch.points)):
    #         # d_print("shape of point list is {}".format(batch.points[i].shape), color=bcolors.OKBLUE)

        
    #     # d_print("centers shape {}".format(batch.centers.shape))
    #     # d_print("stacked features shape {}".format(batch.features.shape))
    #     # d_print("times shape: {}".format(batch.times.shape))
    #     # d_print("unique times {}".format(torch.unique(batch.times)))
    #     #AB: x [N,9]
    #     x = torch.cat([batch.features[:,1:], batch.centers, torch.unsqueeze(batch.times, dim=1)], dim=1)
    #     # d_print(x[:,:3])
    #     # write_pc(x[:,:3].detach().cpu().numpy(), 'grid_subsampling')
        
        
    #     #AB: prepare the input for PointNet
    #     #AB: add batch dim
    #     x = torch.unsqueeze(x, dim=0)
    #     #AB: PointNet takes shape [B,f,N]
    #     x = x.transpose(2,1) 

        

    #     # d_print(x.shape, color=bcolors.OKBLUE)

        
        
        
    #     # PointNet specific part

    #     # batchsize = x.size()[0]
    #     # n_pts = x.size()[2]
    #     x, trans, trans_feat = self.feat(x)
    #     # x = F.relu(self.bn1(self.conv1(x)))
    #     # x = F.relu(self.bn2(self.conv2(x)))
    #     # x = F.relu(self.bn3(self.conv3(x)))
    #     x = F.relu((self.conv1(x)))
    #     x = F.relu((self.conv2(x)))
    #     x = F.relu((self.conv3(x)))
    #     # x = self.conv4(x)
    #     x = x.transpose(2,1).contiguous()
    #     # x = F.log_softmax(x.view(-1,self.C), dim=-1)
    #     # x = x.view(batchsize, n_pts, self.C)
    #     # return x, trans_feat

    #     # d_print("output of point net is {}".format(x.shape))
        
    #     # Pipeline Specific part

    #     # Head of network
    #     f = self.head_mlp(x, batch)
    #     c = self.head_center(f, batch)
    #     c = self.sigmoid(c)
    #     v = self.head_var(f, batch)
    #     v = F.relu(v)
    #     x = self.head_softmax(f, batch)

    #     return x, c, v, f

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
            
# class get_loss(torch.nn.Module):
#     def __init__(self, mat_diff_loss_scale=0.001):
#         super(get_loss, self).__init__()
#         self.mat_diff_loss_scale = mat_diff_loss_scale

#     def forward(self, pred, target, trans_feat, weight):
#         loss = F.nll_loss(pred, target, weight = weight)
#         mat_diff_loss = feature_transform_reguliarzer(trans_feat)
#         total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
#         return total_loss
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

# if __name__ == '__main__':
    # model = get_model(13)
    # xyz = torch.rand(12, 3, 2048)
    # (model(xyz))