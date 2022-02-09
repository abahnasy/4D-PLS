import os
from typing import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra

from models.losses_pointnet import *
from models.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation
from utils.metrics import IoU_from_confusions, fast_confusion
from utils.config import bcolors
from utils.debugging import d_print
from utils.kalman_filter import KalmanBoxTracker

def same_shape(shape1, shape2):
    if len(shape1) != len(shape2):
        return False
    for i in range(len(shape1)):
        if shape1[i] != shape2[i]:
            return False
    return True

def kalman_box_to_eight_point(kalman_bbox):

    # x, y, z, theta, l, w, h to x1,x2,y1,y2,z1,z2
    x1 = kalman_bbox[0]-kalman_bbox[4]/2
    x2 = kalman_bbox[0]+kalman_bbox[4]/2
    y1 = kalman_bbox[1]-kalman_bbox[5]/2
    y2 = kalman_bbox[1]+kalman_bbox[5]/2
    z1 = kalman_bbox[2]-kalman_bbox[6]/2
    z2 = kalman_bbox[2]+kalman_bbox[6]/2

    return [x1,y1,z1,x2,y2,z2]

def get_bbox_from_points(points):
    """
    Runs the loss on outputs of the model
    :param points: instance points Nx3
    :return: 3D bbox [x1,y1,z1,x2,y2,z2]
    """
    points = points.cpu() #AB: fix 
    x1 = torch.min(points[:, 0])
    x2 = torch.max(points[:, 0])
    y1 = torch.min(points[:, 1])
    y2 = torch.max(points[:, 1])
    z1 = torch.min(points[:, 2])
    z2 = torch.max(points[:, 2])

    return [x1,y1,z1,x2,y2,z2], np.array([x1 + (x2-x1)/2, y1+ (y2-y1)/2,z1+ (z2-z1)/2, 0, x2-x1,y2-y1,z2-z1]) # x, y, z, theta, l, w, h

class BaseModel(nn.Module):
    """ super class for all models
    common functions: semantic_seg_metric, accuracy, loss
    model specific functions are: init, forward, pretrained weights
    """
    def __init__(self):
        super(BaseModel, self).__init__()

    def _load_pretrained_weights(self, ckpt_path):
        """ heads weights will be loaded from 4D-Panoptic Segmentation check point
        """
        # HEADS_WEIGHTS_PATH = hydra.utils.to_absolute_path("./results/Log_2020-10-06_16-51-05/checkpoints/current_chkp.tar")
        # if not os.path.exists(HEADS_WEIGHTS_PATH):
        #     d_print(HEADS_WEIGHTS_PATH, bcolors.FAIL)
        #     raise ValueError(" above Path doesn't exist")
        # loaded_state_dict = OrderedDict()
        # heads_loaded_sate_dict = torch.load(HEADS_WEIGHTS_PATH, map_location=torch.device('cpu'))['model_state_dict']
        # heads_state_dict = OrderedDict()
        # for k,v in heads_loaded_sate_dict.items():
        #     if "head" in k:
        #         heads_state_dict[k] = v
        # # add heads weights to the backbone loaded weights
        # loaded_state_dict.update(heads_state_dict)

        CKPT_PATH = hydra.utils.to_absolute_path(ckpt_path)
        if not os.path.exists(CKPT_PATH):
            d_print(CKPT_PATH, bcolors.FAIL)
            raise ValueError(" above Path doesn't exist")
        loaded_state_dict = torch.load(CKPT_PATH, map_location=torch.device('cpu'))['model_state_dict']
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
        raise NotImplementedError


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
        confusion_matrix = torch.zeros(self.C, self.C, dtype=torch.int32)
        # d_print("....")
        # d_print(preds.shape)
        # d_print(target.shape)
        for pred_class in range(0,self.C):
            for target_class in range(0, self.C):
                confusion_matrix[pred_class][target_class] = ((preds == pred_class) & (target == target_class)).sum().int()
        # d_print(confusion_matrix)
        intersection = torch.diag(confusion_matrix)
        union = confusion_matrix.sum(0) + confusion_matrix.sum(1) - intersection
        # d_print(confusion_matrix)
        # d_print(confusion_matrix.sum(0))
        # d_print(confusion_matrix.sum(1))
        scores = intersection.float() / (union.float() + 1e-10)
        
        mask = union.numpy() < 1e-3
        counts = np.sum(1 - mask)
        
        meanIoU = scores.sum() / counts
        
        
        
        
        # return class IoUs
        return scores, meanIoU

    
    def fast_semantic_seg_metric(self, outputs, labels, label_values, ignored_labels):
        """ semantic segmentation metric calculation

        Args:
            outputs: sofrmax outputs from the segmentation head
            labels: ground truth semantic labels for each point
        Returns:
            iou_metrics: dictionary contains the average IoU and IoU for each class
        """
        
        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        probs = outputs.view(-1,19)
        labels = labels.view(-1, 1).squeeze(dim=-1).numpy()
        
        # add zeros to the missing labels from the model
        for l_ind, label_value in enumerate(label_values):
            if label_value in ignored_labels:
                probs = np.insert(probs, l_ind, 0, axis=1)
        # Predicted labels
        preds = label_values[np.argmax(probs, axis=1)]

        frame_C = fast_confusion(
            labels,
            preds.astype(np.int32),
            label_values
        )
        
        # Remove ignored labels from confusions
        for l_ind, label_value in reversed(list(enumerate(label_values))):
            if label_value in ignored_labels:
                frame_C = np.delete(frame_C, l_ind, axis=0)
                frame_C = np.delete(frame_C, l_ind, axis=1)

        # d_print(frame_C)
        # d_print(frame_C.sum(0))
        # d_print(frame_C.sum(1))
        
        IoUs = IoU_from_confusions(frame_C)
        # d_print(IoUs)

        return IoUs
        





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
                labels, ins_labels, centers_gt, points=None, times=None, verbose = False):
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
            
    def plot_class_statistics(self, outputs, labels):
        """
        outputs: [B,N,19]
        labels: [B,N,1]
        TODO: do it later using confusion matrix !
        """
        from prettytable import PrettyTable
        x = PrettyTable()
        # x.field_names = ["class", "true", "preds"]
        outputs = outputs.view(-1,19)
        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        labels = labels.view(-1,)
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i
        predicted = torch.argmax(outputs, dim=1)
        predicted = predicted.cpu().numpy()
        target = target.numpy()
        
        true_list = []
        pred_list = []
        for c in range(19):
            true_list.append((target == c).sum())
            pred_list.append((predicted == c).sum())
        x.add_column("class", [i for i in range(19)])
        x.add_column("true", true_list)
        x.add_column("pred", pred_list)
        print(x)
    
    def accuracy(self, outputs, labels, verbose= False):
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

    def cross_entropy_loss(self, outputs, labels):
        batch_size = outputs.shape[0]
        num_points = outputs.shape[1]
        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for b in range(batch_size):
            for i, c in enumerate(self.valid_labels):
                target[b][labels[b,:] == c] = i
        self.output_loss = self.criterion(outputs.view(-1,19), target.view(-1,).long())
        self.output_loss /= batch_size
        return self.output_loss

    def ins_pred(self, predicted, centers_output, var_output, embedding, points=None, times=None):
        """
        Calculate instance probabilities for each point on current frame
        :param predicted: class labels for each point
        :param centers_output: center predictions
        :param var_output : variance predictions
        :param embedding : embeddings for all points
        :param points: xyz location of points
        :return: instance ids for all points
        """
        #predicted = torch.argmax(outputs.data, dim=1)

        if var_output.shape[1] - embedding.shape[1] > 4:
            global_emb, _ = torch.max(embedding, 0, keepdim=True)
            embedding = torch.cat((embedding, global_emb.repeat(embedding.shape[0], 1)), 1)

        if  var_output.shape[1] - embedding.shape[1] == 3:
            embedding = torch.cat((embedding, points[0]), 1)
        if  var_output.shape[1] - embedding.shape[1] == 4:
            # embedding = torch.cat((embedding, points[0], times), 1)
            #AB: removed the indexing of points, done before calling the function 
            embedding = torch.cat((embedding, points, times), 1)

        if var_output.shape[1] == 3:
            embedding = points[0]
        if var_output.shape[1] == 4:
            embedding = torch.cat((points[0], times), 1)

        ins_prediction = torch.zeros_like(predicted)
        d_print(ins_prediction.shape)

        counter = 0 # AB: used to search for the suitable instance center within certain group of points
        ins_id = 1 # label number to be assigned
        while True:
            # AB: collect points for things class and still not assigned an instance label
            ins_idxs = torch.where((predicted < 9) & (predicted != 0) & (ins_prediction == 0))
            #AB: if there is no remaining unlabeled points, break
            if len(ins_idxs[0]) == 0:
                break
            #AB: for chosen points, get the center scroes, embeddings and variances
            ins_centers = centers_output[ins_idxs]
            ins_embeddings = embedding[ins_idxs]
            ins_variances = var_output[ins_idxs]
            if counter == 0:
                # AB: sort points according to their center prob, highest first
                sorted, indices = torch.sort(ins_centers, 0, descending=True)  # center score of instance classes
            if sorted[0+counter] < 0.1 or (ins_id ==1 and sorted[0] < 0.7):
                break
            # AB: get the point with the higest score and consider it the center of the object
            idx = indices[0+counter]
            mean = ins_embeddings[idx]
            var = ins_variances[idx]
            #probs = pdf_normal(ins_embeddings, mean, var)
            # AB: measure the likelihood of other points belong to this instance
            probs = new_pdf_normal(ins_embeddings, mean, var)
            # AB: choose points with probs > 0.5 as part of this instance
            ins_points = torch.where(probs >= 0.5)
            # AB: if you didn't find points, choose another point as center and redo calculations
            if ins_points[0].size()[0] < 2:
                counter +=1
                if counter == sorted.shape[0]:
                    break
                continue
            # AB: assign the instance id to the selected points
            ids = ins_idxs[0][ins_points[0]]
            ins_prediction[ids] = ins_id
            counter = 0
            ins_id += 1
        return ins_prediction

    def ins_pred_in_time(self, predicted, centers_output, var_output, embedding, prev_instances, next_ins_id, points=None, times=None, pose=None):
        """
        Calculate instance probabilities for each point with considering old predictions also
        :param predicted: class labels for each point
        :param centers_output: center predictions
        :param var_output : variance predictions
        :param embedding : embeddings for all points
        :param prev_instances : instances which detected in previous frames
        :param next_ins_id : next avaliable ins id
        :param points: xyz location of points
        :return: instance ids for all points, and new instances and next available ins_id
        """
        new_instances = {}
        ins_prediction = torch.zeros_like(predicted)

        if var_output.shape[1] - embedding.shape[1] > 4:
            global_emb, _ = torch.max(embedding, 0, keepdim=True)
            embedding = torch.cat((embedding, global_emb.repeat(embedding.shape[0], 1)), 1)

        if var_output.shape[1] - embedding.shape[1] == 3:
            embedding = torch.cat((embedding, points[0]), 1)
        if var_output.shape[1] - embedding.shape[1] == 4:
            embedding = torch.cat((embedding, points[0], times), 1)

        pose = torch.from_numpy(pose)
        pose = pose.to(embedding.device)

        counter = 0
        ins_id = next_ins_id

        while True:
            ins_idxs = torch.where((predicted < 9) & (predicted != 0) & (ins_prediction == 0))
            if len(ins_idxs[0]) == 0:
                break
            ins_centers = centers_output[ins_idxs]
            ins_embeddings = embedding[ins_idxs]
            ins_variances = var_output[ins_idxs]
            ins_points = points[0][ins_idxs]
            if counter == 0:
                sorted, indices = torch.sort(ins_centers, 0, descending=True)  # center score of instance classes
            if sorted[0 + counter] < 0.1 or (sorted[0] < 0.7):
                break
            idx = indices[0 + counter]
            mean = ins_embeddings[idx]
            var = ins_variances[idx]

            center = points[0][ins_idxs][idx]
            distances = torch.sum((ins_points - center)**2,1)
            # if torch.cuda.device_count() > 1:
            #     new_device = torch.device("cuda:1")
            #     probs = new_pdf_normal(ins_embeddings.to(new_device), mean.to(new_device), var.to(new_device))
            # else:
            probs = new_pdf_normal(ins_embeddings, mean, var)

            probs[distances>20] = 0
            ins_points = torch.where(probs >= 0.5)
            if ins_points[0].size()[0] < 2:
                counter += 1
                if counter == sorted.shape[0]:
                    break
                continue

            ids = ins_idxs[0][ins_points[0]]
            ins_prediction[ids] = ins_id
            if ins_points[0].size()[0] > 25: #add to instance history
                d_print("I found an instance with 25")
                ins_prediction[ids] = ins_id
                mean = torch.mean(embedding[ids], 0, True)
                bbox, kalman_bbox = get_bbox_from_points(points[0][ids])
                tracker = KalmanBoxTracker(kalman_bbox ,ins_id)
                bbox_proj = None
                #var = torch.mean(var_output[ids], 0, True)
                new_instances[ins_id] = {'mean': mean, 'var': var, 'life' : 5, 'bbox': bbox, 'bbox_proj':bbox_proj, 'tracker': tracker, 'kalman_bbox' : kalman_bbox}

            counter = 0
            ins_id += 1
        
        # associate instances by hungarian alg. & bbox prediction via kalman filter
        if len(prev_instances.keys()) > 0 :

            #association_costs, associations = self.associate_instances(config, prev_instances, new_instances, pose)
            associations = []
            for prev_id, new_id in associations:
                raise ValueError("am i never visited?")
                ins_points = torch.where((ins_prediction == new_id))
                ins_prediction[ins_points[0]] = prev_id
                prev_instances[prev_id]['mean'] = new_instances[new_id]['mean']
                prev_instances[prev_id]['bbox_proj'] = new_instances[new_id]['bbox_proj']

                prev_instances[prev_id]['life'] += 1
                prev_instances[prev_id]['tracker'].update(new_instances[new_id]['kalman_bbox'], prev_id)
                prev_instances[prev_id]['kalman_bbox'] = prev_instances[prev_id]['tracker'].get_state()
                prev_instances[prev_id]['bbox'] = kalman_box_to_eight_point(prev_instances[prev_id]['kalman_bbox'])

                del new_instances[new_id]

        return ins_prediction, new_instances, ins_id


                
        



# if __name__ == '__main__':
#     import  torch
#     model = get_model(13)
#     xyz = torch.rand(6, 9, 2048)
#     (model(xyz))