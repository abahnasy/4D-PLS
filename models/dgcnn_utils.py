#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: util
@Time: 4/5/19 3:47 PM
"""


import numpy as np
import torch
import torch.nn.functional as F
from kernels.kernel_points import create_3D_rotations
from sklearn.utils.class_weight import compute_class_weight


def get_class_weights(lbl_values, ign_lbls, labels):
    
    valid_labels = np.sort([c for c in lbl_values if c not in ign_lbls])
    labels = labels.view(-1,)
    target = - torch.ones_like(labels)
    for i, c in enumerate(valid_labels):
        target[labels == c] = i
    
    y = target[target!=-1].numpy()
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    all_class_weights = torch.zeros(19)
    for i in range(len(class_weights)):
        all_class_weights[np.unique(y)[i]] = class_weights[i]
    return all_class_weights


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


def get_model_parameters(model):
    total_parameters = 0
    for layer in list(model.parameters()):
        layer_parameter = 1
        for l in list(layer.size()):
            layer_parameter *= l
        total_parameters += layer_parameter
    return total_parameters


def rotate_pointcloud(config, points, angle_range_z=60):
    ##########
    # Rotation
    ##########

    # Initialize rotation matrix
    R = np.eye(points.shape[1])

    if points.shape[-1] == 3:
        if config.eval_rotation == 'vertical':

            # Create random rotations
            theta = angle_range_z/360 * 2 * np.pi #* (2*np.random.rand()-1)   # a random angle within +/- angle_range_z
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

        elif config.eval_rotation == 'all':
            raise NotImplementedError

            # Choose two random angles for the first vector in polar coordinates
            theta = np.random.rand() * 2 * np.pi
            phi = (np.random.rand() - 0.5) * np.pi

            # Create the first vector in carthesian coordinates
            u = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

            # Choose a random rotation angle
            alpha = np.random.rand() * 2 * np.pi

            # Create the rotation matrix with this vector and angle
            R = create_3D_rotations(np.reshape(u, (1, -1)), np.reshape(alpha, (1, -1)))[0]

    R = R.astype(np.float32)
    rotated_points = np.sum(np.expand_dims(points, points.ndim) * R, axis=-1)
    return rotated_points