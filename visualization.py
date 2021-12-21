import signal
import os
import numpy as np
import sys
import torch
import open3d as o3d
import numpy as np

from utils.config import Config
from datasets.SemanticKitti import *
from torch.utils.data import DataLoader
from kernels.kernel_points import create_3D_rotations

def augmentation_transform(self, config, points, normals=None, verbose=False):
        """Implementation of an augmentation transform for point clouds."""

        ##########
        # Rotation
        ##########

        # Initialize rotation matrix
        R = np.eye(points.shape[1])

        if points.shape[1] == 3:
            if self.config.augment_rotation == 'vertical':

                # Create random rotations
                theta =  config.augment_rotation_angle/360 * 2 * np.pi
                c, s = np.cos(theta), np.sin(theta)
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

            elif self.config.augment_rotation == 'all':

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

        augmented_points = np.sum(np.expand_dims(points, 2) * R, axis=1) #* scale + noise


        if normals is None:
            return augmented_points, scale, R
        else:
            # Anisotropic scale of the normals thanks to cross product formula
            normal_scale = scale[[1, 2, 0]] * scale[[2, 0, 1]]
            augmented_normals = np.dot(normals, R) * normal_scale
            # Renormalise
            augmented_normals *= 1 / (np.linalg.norm(augmented_normals, axis=1, keepdims=True) + 1e-6)

            if verbose:
                test_p = [np.vstack([points, augmented_points])]
                test_n = [np.vstack([normals, augmented_normals])]
                test_l = [np.hstack([points[:, 2]*0, augmented_points[:, 2]*0+1])]
                show_ModelNet_examples(test_p, test_n, test_l)

            return augmented_points, augmented_normals, scale, R


if __name__ == '__main__':

    config = Config()
    config.on_gpu=True

    config.global_fet = False
    config.validation_size = 270
    config.input_threads = 16
    config.n_frames = 4
    config.n_test_frames = 4 #it should be smaller than config.n_frames
    if config.n_frames < config.n_test_frames:
        config.n_frames = config.n_test_frames
    config.big_gpu = False
    config.dataset_task = '4d_panoptic'
    #config.sampling = 'density'
    config.sampling = 'importance'
    config.decay_sampling = 'None'
    config.stride = 1
    config.first_subsampling_dl = 0.061

     # Augmentation parameters
    config.augment_scale_anisotropic = True
    config.augment_scale_min = 1
    config.augment_scale_max = 1
    config.augment_symmetries = [False, False, False]
    config.augment_rotation = 'vertical'
    config.augment_rotation_angle = 15
    config.augment_noise = 0
    config.augment_color = 1 # no color augmentation

    set = 'validation'
    test_dataset = SemanticKittiDataset(config, set=set, balance_classes=False, seqential_batch=True)
    test_sampler = SemanticKittiSampler(test_dataset)
    collate_fn = SemanticKittiCollate

    # Data loader
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=collate_fn,
                             num_workers=0,#config.input_threads,
                             pin_memory=True)

    # Calibrate samplers
    #test_sampler.calibration(test_loader, verbose=True)
    for batch in enumerate(test_loader):
      print(batch.size())

      break

    #pts, scale, R = augmentation_transform(config, pts)
