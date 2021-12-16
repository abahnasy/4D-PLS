import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from models.pointnet_utils import PointNetEncoder, feature_transform_reguliarzer
from utils.debugging import d_print, write_pc
from utils.config import bcolors

from models.losses import *

from models.architectures import p2p_fitting_regularizer

class get_model(nn.Module):
    def __init__(self, config, lbl_values, ign_lbls):
        super(get_model, self).__init__()
        
        self.pre_train = config.pre_train
        #AB: variable is used in the loss function 
        self.valid_labels = np.sort([c for c in lbl_values if c not in ign_lbls])
        # Choose segmentation loss
        if len(config.class_w) > 0:
            class_w = torch.from_numpy(np.array(config.class_w, dtype=np.float32))
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_w, ignore_index=-1)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        # self.k = num_class
        self.C = len(lbl_values) - len(ign_lbls)
        out_dim = 128
        
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=9)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        # self.conv4 = torch.nn.Conv1d(128, self.C, 1)
        # self.bn1 = nn.BatchNorm1d(512)
        # self.bn2 = nn.BatchNorm1d(256)
        # self.bn3 = nn.BatchNorm1d(128)

        
        # add the prediction heads
        from models.blocks import UnaryBlock
        self.head_mlp = UnaryBlock(out_dim, config.first_features_dim, False, 0)
        self.head_var = UnaryBlock(config.first_features_dim, out_dim + config.free_dim, False, 0)
        self.head_softmax = UnaryBlock(config.first_features_dim, self.C, False, 0)
        self.head_center = UnaryBlock(config.first_features_dim, 1, False, 0, False)

        self.sigmoid = nn.Sigmoid()

    # def forward(self, x):
    #     batchsize = x.size()[0]
    #     n_pts = x.size()[2]
    #     x, trans, trans_feat = self.feat(x)
    #     x = F.relu(self.bn1(self.conv1(x)))
    #     x = F.relu(self.bn2(self.conv2(x)))
    #     x = F.relu(self.bn3(self.conv3(x)))
    #     x = self.conv4(x)
    #     x = x.transpose(2,1).contiguous()
    #     x = F.log_softmax(x.view(-1,self.C), dim=-1)
    #     x = x.view(batchsize, n_pts, self.C)
    #     return x, trans_feat

    def forward(self, batch, config):
        """
        Args:
            batch: 
            config: TODO: remove from this and KPConv models
        Returns:
        """
        d_print("inside forward function")
        #AB: prepare the correct feature inputs by concatenating the features again
        #AB: point feature is [Reflectance, c_x, c_y, c_z, t]
        #AB: Batch size is always one since we load only one 4D volume per dataloader
        
        d_print("points shape {}".format(batch.points[0].shape))
        for i in range(len(batch.points)):
            d_print("shape of point list is {}".format(batch.points[i].shape), color=bcolors.OKBLUE)

        
        # d_print("centers shape {}".format(batch.centers.shape))
        # d_print("stacked features shape {}".format(batch.features.shape))
        # d_print("times shape: {}".format(batch.times.shape))
        # d_print("unique times {}".format(torch.unique(batch.times)))
        #AB: x [N,9]
        x = torch.hstack([batch.features[:,1:], batch.centers, torch.unsqueeze(batch.times, dim=1)])
        # d_print(x[:,:3])
        # write_pc(x[:,:3].detach().cpu().numpy(), 'grid_subsampling')
        
        
        #AB: prepare the input for PointNet
        #AB: add batch dim
        x = torch.unsqueeze(x, dim=0)
        #AB: PointNet takes shape [B,f,N]
        x = x.transpose(2,1) 

        

        # d_print(x.shape, color=bcolors.OKBLUE)

        
        
        
        # PointNet specific part

        # batchsize = x.size()[0]
        # n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu((self.conv1(x)))
        x = F.relu((self.conv2(x)))
        x = F.relu((self.conv3(x)))
        # x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        # x = F.log_softmax(x.view(-1,self.C), dim=-1)
        # x = x.view(batchsize, n_pts, self.C)
        # return x, trans_feat

        d_print("output of point net is {}".format(x.shape))
        
        # Pipeline Specific part

        # Head of network
        f = self.head_mlp(x, batch)
        c = self.head_center(f, batch)
        c = self.sigmoid(c)
        v = self.head_var(f, batch)
        v = F.relu(v)
        x = self.head_softmax(f, batch)

        return x, c, v, f


    def loss(self, outputs, centers_p, variances, embeddings, labels, ins_labels, centers_gt, points=None, times=None):
            """
            AB: copied as it is from architectures.py KPFCONV
            Runs the loss on outputs of the model
            :param outputs: logits
            :param labels: labels
            :return: loss
            """

            # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
            target = - torch.ones_like(labels)
            for i, c in enumerate(self.valid_labels):
                target[labels == c] = i

            # Reshape to have a minibatch size of 1
            # outputs = torch.transpose(outputs, 0, 1)
            # outputs = outputs.unsqueeze(0)
            outputs = torch.squeeze(outputs, dim=0)
            target = target.unsqueeze(0)
            target = target.transpose(0,1).squeeze() #AB: move from [1,N] -> [N,]
            centers_p = centers_p.squeeze().squeeze() #AB: make it of shape (N,)
            # Cross entropy loss
            d_print("before cross entropy loss")
            d_print("shape of outptus {}".format(outputs.shape))
            d_print("shape of the targets {}".format(target.shape))
            self.output_loss = self.criterion(outputs, target)
            weights = (centers_gt[:, 0] > 0) * 99 + (centers_gt[:, 0] >= 0) * 1
            self.center_loss = weighted_mse_loss(centers_p, centers_gt[:, 0], weights)

            if not self.pre_train:
                self.instance_half_loss = instance_half_loss(embeddings, ins_labels)
                self.instance_loss = iou_instance_loss(centers_p, embeddings, variances, ins_labels, points, times)
                self.variance_loss = variance_smoothness_loss(variances, ins_labels)
                self.variance_l2 = variance_l2_loss(variances, ins_labels)
            
            #AB: comment out regularizer loss, related to KPConv only !
            # Regularization of deformable offsets
            # if self.deform_fitting_mode == 'point2point':
            #     self.reg_loss = p2p_fitting_regularizer(self)
            # elif self.deform_fitting_mode == 'point2plane':
            #     raise ValueError('point2plane fitting mode not implemented yet.')
            # else:
            #     raise ValueError('Unknown fitting mode: ' + self.deform_fitting_mode)

            # Combined loss
            #return self.instance_loss + self.variance_loss
            # return self.output_loss + self.reg_loss + self.center_loss + self.instance_loss*0.1+ self.variance_loss*0.01
            return self.output_loss + self.center_loss + self.instance_loss*0.1+ self.variance_loss*0.01
# class get_loss(torch.nn.Module):
#     def __init__(self, mat_diff_loss_scale=0.001):
#         super(get_loss, self).__init__()
#         self.mat_diff_loss_scale = mat_diff_loss_scale

#     def forward(self, pred, target, trans_feat, weight):
#         loss = F.nll_loss(pred, target, weight = weight)
#         mat_diff_loss = feature_transform_reguliarzer(trans_feat)
#         total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
#         return total_loss


if __name__ == '__main__':
    model = get_model(13)
    xyz = torch.rand(12, 3, 2048)
    (model(xyz))