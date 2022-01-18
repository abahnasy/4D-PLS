import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pickle
import os
from os import makedirs, remove
from os.path import exists, join
import time
import sys
from models.losses import isnan
from models.dgcnn_utils import rotate_pointcloud
from utils.debugging import d_print

from models.dgcnn_sem_seg import DGCNN_semseg
from models.dgcnn_utils import get_class_weights
from datasets.semantic_kitti_dataset import SemanticKittiDataSet

# PLY reader
from utils.ply import read_ply, write_ply

# Metrics
from utils.metrics import IoU_from_confusions, fast_confusion
from sklearn.neighbors import KDTree


class ModelTrainerDGCNN:

    # Initialization methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, net, config, chkp_path=None, finetune=False, on_gpu=True):
        """
        Initialize training parameters and reload previous model for restore/finetune
        :param net: network object
        :param config: configuration object
        :param chkp_path: path to the checkpoint that needs to be loaded (None for new training)
        :param finetune: finetune from checkpoint (True) or restore training from checkpoint (False)
        :param on_gpu: Train on GPU or CPU
        """

        # Writer will output to ./runs/ directory by default
        self.date_time = time.strftime("%Y%m%d-%H%M")
        self.epoch = 0
        self.global_step = 0
        self.step = 0

        self.optimizer = torch.optim.SGD(net.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=1e-4)
        if config.lr_scheduler == True:
            milestones = [200, 400, 600]
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.45, verbose=False)

        # Choose to train on CPU or GPU
        if on_gpu and torch.cuda.is_available():
            print('On GPU')
            self.device = torch.device("cuda:0")
        else:
            print('On CPU')
            self.device = torch.device("cpu")
        

        ##########################
        # Load previous checkpoint
        ##########################
        if finetune==True:
            load_dgcnn_weights = True
            load_heads = True

        if (chkp_path is not None):
            if load_dgcnn_weights: 
                pretrained_dgcnn = torch.load(chkp_path, map_location=self.device)
                # Rename the pretrained model for loading
                renamed_parameters = {}
                for key, value in pretrained_dgcnn.items():
                    not_loading = ['conv1', 'conv8', 'bn1', 'bn8', 'conv9']
                    if not any(layer in key for layer in not_loading):
                        renamed_parameters[key[7:]] = value
                net.load_state_dict(renamed_parameters, strict=False)
                print('dgcnn pretrained weights loaded.')

            if load_heads:
                chkp_path_kpconv = './results/Log_2020-10-06_16-51-05/checkpoints/current_chkp.tar'
                checkpoint_heads = torch.load(chkp_path_kpconv, map_location=self.device)
                net.load_state_dict(checkpoint_heads['model_state_dict'], strict=False)
                print('kpconv decoder heads pretrained weights loaded.')
                freezed_layers = ['head_mlp.mlp.weight', 
                                'head_mlp.batch_norm.bias',
                                'head_var.mlp.weight',
                                'head_var.batch_norm.bias',
                                'head_softmax.mlp.weight', 
                                'head_softmax.batch_norm.bias',
                                'head_center.mlp.weight',
                                'head_center.batch_norm.bias']
                for name, value in net.named_parameters():
                    if name in freezed_layers:
                        value.requires_grad = False

            # print(pretrained_dgcnn['module.conv5.0.weight'][:5,0,0])
            # print(net.conv5[0].weight[:5,0,0])         
            # print(pretrained_dgcnn['module.conv1.0.weight'][:5,0,0])
            # print(net.conv1[0].weight[:5,0,0])         
            # print(checkpoint_heads['model_state_dict']['head_var.mlp.weight'][:5,0])
            # print(net.head_var.mlp.weight[:5,0])
        
        net.to(self.device)   

         # Path of the result folder
        if config.saving:
            if config.saving_path is None:
                config.saving_path = time.strftime('results/dgcnn/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
            if not exists(config.saving_path):
                makedirs(config.saving_path)
            config.save()


        self.logger = SummaryWriter(log_dir=config.saving_path+'/runs')

        return

    def train_overfit_oneframe(self, net, train_loader, epochs=1000):
        net.train()

        # Overfit one frame: 
        # samples = train_set[0]
        # in_fts = torch.tensor(samples['in_fts']).unsqueeze(0)
        # labels = torch.tensor(samples['in_lbls']).type(torch.LongTensor)
        
        # Overfit one batch:
        for batch in train_loader:

            # move to device (GPU)
            sample_gpu ={}
            if 'cuda' in self.device.type:
                for k, v in batch.items():
                    sample_gpu[k] = v.to(self.device)
            else:
                sample_gpu = batch

            in_fts = sample_gpu['in_fts'][:,:,:4]
            labels = sample_gpu['in_lbls'].type(torch.LongTensor)

            for epoch in range(epochs):
                
                self.optimizer.zero_grad()
                outputs = net(in_fts)

                loss = net.cross_entropy_loss(outputs, labels)
                ious = net.semantic_seg_metric(outputs, labels)
                nan_idx = torch.isnan(ious)
                ious[nan_idx] = 0.
                acc = net.accuracy(outputs, labels)
                loss.backward()
                self.optimizer.step()

                for i, iou in enumerate(ious):
                    if isnan(iou):
                        iou = 0.
                    self.logger.add_scalar('ious/{}'.format(i), iou, self.global_step)    
                # log mean IoU
                self.logger.add_scalar('ious/meanIoU', ious.mean(), self.global_step)    
                #AB: log into tensorbaord
                self.logger.add_scalar('Loss/CrossEntropy', loss.item(), self.global_step)
                self.logger.add_scalar('acc/train', acc*100, self.global_step)
                print('Epoch:{0:4d}, loss:{1:2.3f}, iou_mean:{2:2.3f}, accuracy:{3:.3f}'.format(self.global_step, loss.item(), ious.mean(), acc*100))
            
                self.global_step += 1
            
            break
    

    def train_overfit_4D(self, config, net=None, train_loader=None):
        ################
        # Initialization
        ################

        if config.saving:
            # Training log file
            with open(join(config.saving_path, 'training.txt'), "w") as file:
                file.write('epochs steps out_loss offset_loss train_accuracy time\n')

            # Checkpoints directory
            checkpoint_directory = join(config.saving_path, 'checkpoints')
            if not exists(checkpoint_directory):
                makedirs(checkpoint_directory)
        else:
            checkpoint_directory = None
        
        if train_loader is None:
            DATASET_PATH = './data'
            train_set = SemanticKittiDataSet(path=DATASET_PATH, set='train')
            train_loader = DataLoader(train_set, batch_size= 4, num_workers=4, pin_memory=True)
        
        # Overfit one batch:
        for batch in train_loader:
            
            if net is None:
                labels = batch['in_lbls']
                class_weights = get_class_weights(train_set.label_values, train_set.ignored_labels, labels)
                net=DGCNN_semseg(train_set.label_values, train_set.ignored_labels, input_feature_dims=4, class_weights=class_weights)
                self.optimizer = torch.optim.SGD(net.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=1e-4)
                # load_dgcnn_weights: 
                chkp_path = './results/dgcnn_semseg_pretrained/model_1.t7'
                pretrained_dgcnn = torch.load(chkp_path, map_location=self.device)
                # Rename the pretrained model for loading
                renamed_parameters = {}
                for key, value in pretrained_dgcnn.items():
                    not_loading = ['conv1', 'conv8', 'bn1', 'bn8', 'conv9']
                    if not any(layer in key for layer in not_loading):
                        renamed_parameters[key[7:]] = value
                net.load_state_dict(renamed_parameters, strict=False)
                print('dgcnn pretrained weights loaded.')

                 # load_heads:
                chkp_path_kpconv = './results/Log_2020-10-06_16-51-05/checkpoints/current_chkp.tar'
                checkpoint_heads = torch.load(chkp_path_kpconv, map_location=self.device)
                net.load_state_dict(checkpoint_heads['model_state_dict'], strict=False)
                print('kpconv decoder heads pretrained weights loaded.')
                freezed_layers = ['head_mlp.mlp.weight', 
                                'head_mlp.batch_norm.bias',
                                'head_var.mlp.weight',
                                'head_var.batch_norm.bias',
                                'head_softmax.mlp.weight', 
                                'head_softmax.batch_norm.bias',
                                'head_center.mlp.weight',
                                'head_center.batch_norm.bias']
                for name, value in net.named_parameters():
                    if name in freezed_layers:
                        value.requires_grad = False

                
            net.to(self.device)
            net.train() 
            # move to device (GPU)
            sample_gpu ={}
            if 'cuda' in self.device.type:
                for k, v in batch.items():
                    sample_gpu[k] = v.to(self.device)
            else:
                sample_gpu = batch

            # Save the batch used in this overfitting experiment
            batch_path = join(checkpoint_directory, 'batch.tar')
            torch.save(sample_gpu, batch_path)           

            centers = sample_gpu['in_fts'][:,:,4:8]
            times = sample_gpu['in_fts'][:,:,8]

            for epoch in range(config.max_epoch):

                self.optimizer.zero_grad()
                outputs, centers_output, var_output, embedding = net(sample_gpu['in_fts'][:,:,:4])

                loss = net.loss(
                    outputs, centers_output, var_output, embedding, 
                    sample_gpu['in_lbls'], sample_gpu['in_slbls'], centers, sample_gpu['in_pts'], times)                
                acc = net.accuracy(outputs.cpu(), sample_gpu['in_lbls'].cpu())

                ious = net.semantic_seg_metric(outputs.cpu(), sample_gpu['in_lbls'].cpu())               
                nan_idx = torch.isnan(ious)
                ious[nan_idx] = 0.

                loss.backward()
                if config.grad_clip_norm > 0:
                    # torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip_norm)
                    torch.nn.utils.clip_grad_value_(net.parameters(), config.grad_clip_norm)
                self.optimizer.step()
                if config.lr_scheduler == True:        
                    self.lr_scheduler.step()
                if 'cuda' in self.device.type:
                    torch.cuda.synchronize(self.device)

                for i, iou in enumerate(ious):
                    if isnan(iou):
                        iou = 0.
                    self.logger.add_scalar('ious/{}'.format(i), iou, self.global_step)    
                # log mean IoU
                self.logger.add_scalar('ious/meanIoU', ious.mean(), self.global_step)    
                #AB: log into tensorbaord
                self.logger.add_scalar('Loss/total', loss.item(), self.global_step)
                self.logger.add_scalar('Loss/cross_entropy', net.output_loss.item(), self.global_step)
                self.logger.add_scalar('Loss/center_loss', net.center_loss.item(), self.global_step)
                self.logger.add_scalar('Loss/instance_half_loss', net.instance_half_loss.item(), self.global_step)
                self.logger.add_scalar('Loss/instance_loss', net.instance_loss.item(), self.global_step)
                self.logger.add_scalar('Loss/center_loss', net.variance_loss.item(), self.global_step)
                self.logger.add_scalar('Loss/variance_loss', net.variance_l2.item(), self.global_step)               
                self.logger.add_scalar('acc/train', acc*100, self.global_step)
                
                print('Epoch:{0:4d}, loss:{1:2.3f}, iou_mean:{2:2.3f}, accuracy:{3:.3f}'.format(self.global_step, loss.item(), ious.mean(), acc*100))

                # Log file
                if config.saving:
                    with open(join(config.saving_path, 'training.txt'), "a") as file:
                        file.write('Epoch:{0:4d}, loss:{1:2.3f}, iou_mean:{2:2.3f}, accuracy:{3:.3f}\n'.format(self.global_step, loss.item(), ious.mean(), acc*100))

                self.global_step += 1

                # Saving
                if config.saving:
                    # Get current state dict
                    save_dict = {'epoch': self.epoch,
                                'model_state_dict': net.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'saving_path': config.saving_path}

                    # Save current state of the network (for restoring purposes)
                    checkpoint_path = join(checkpoint_directory, 'current_chkp.tar')
                    torch.save(save_dict, checkpoint_path)
                    
                    # Save checkpoints occasionally
                    if (self.epoch + 1) % config.checkpoint_gap == 0:
                        checkpoint_path = join(checkpoint_directory, 'chkp_{:04d}.tar'.format(self.epoch + 1))
                        torch.save(save_dict, checkpoint_path)
            
            break


    def train(self, net, training_loader, val_loader, config):
        """
        Train the model on a particular dataset.
        """
        ################
        # Initialization
        ################

        if config.saving:
            # Training log file
            with open(join(config.saving_path, 'training.txt'), "w") as file:
                file.write('epochs steps out_loss offset_loss train_accuracy time\n')

            # Checkpoints directory
            checkpoint_directory = join(config.saving_path, 'checkpoints')
            if not exists(checkpoint_directory):
                makedirs(checkpoint_directory)
        else:
            checkpoint_directory = None
            # PID_file = None

        # Loop variables
        t0 = time.time()
        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        net.train()
        # Start training loop
        for epoch in range(config.max_epoch):

            self.step = 0
            for batch in training_loader:          
                # New time
                t = t[-1:]
                t += [time.time()]

                # move to device (GPU)
                sample_gpu ={}
                if 'cuda' in self.device.type:
                    for k, v in batch.items():
                        sample_gpu[k] = v.to(self.device)
                else:
                    sample_gpu = batch
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # extract data used in loss functions 
                # split centers and times -> original feats composition [x,y,z,r,c1,c2,c3,objectness,time]
                centers = sample_gpu['in_fts'][:,:,4:8]
                times = sample_gpu['in_fts'][:,:,8]
                # Forward pass
                outputs, centers_output, var_output, embedding = net(sample_gpu['in_fts'][:,:,:4])
                # getting loss 
                loss = net.loss(
                    outputs, centers_output, var_output, embedding, 
                    sample_gpu['in_lbls'], sample_gpu['in_slbls'], centers, sample_gpu['in_pts'], times)
                # getting accuracy
                acc = net.accuracy(outputs.cpu(), sample_gpu['in_lbls'].cpu())
                # getting mean iou and ious for each class
                ious = net.semantic_seg_metric(outputs.cpu(), sample_gpu['in_lbls'].cpu())
                nan_idx = torch.isnan(ious)
                ious[nan_idx] = 0.

                for i, iou in enumerate(ious):
                    if isnan(iou):
                        iou = 0.
                    self.logger.add_scalar('ious/{}'.format(i), iou, self.global_step)    
                # log mean IoU
                self.logger.add_scalar('ious/meanIoU', ious.mean(), self.global_step)    
                #AB: log into tensorbaord
                self.logger.add_scalar('Loss/total', loss.item(), self.global_step)
                self.logger.add_scalar('Loss/cross_entropy', net.output_loss.item(), self.global_step)
                self.logger.add_scalar('Loss/center_loss', net.center_loss.item(), self.global_step)
                self.logger.add_scalar('Loss/instance_half_loss', net.instance_half_loss.item(), self.global_step)
                self.logger.add_scalar('Loss/instance_loss', net.instance_loss.item(), self.global_step)
                self.logger.add_scalar('Loss/center_loss', net.variance_loss.item(), self.global_step)
                self.logger.add_scalar('Loss/variance_loss', net.variance_l2.item(), self.global_step)
                
                self.logger.add_scalar('acc/train', acc*100, self.global_step)
                self.global_step += 1

                #print('Step:{0:4d}, total_loss:{1:2.3f}, iou_mean:{2:2.3f}, accuracy:{3:.3f}'.format(self.global_step, loss.item(), ious.mean(), acc*100))
                
                t += [time.time()]

                # Backward + optimize
                loss.backward()

                if config.grad_clip_norm > 0:
                    # torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip_norm)
                    torch.nn.utils.clip_grad_value_(net.parameters(), config.grad_clip_norm)
                self.optimizer.step()
                if config.lr_scheduler == True:        
                    self.lr_scheduler.step()

                if 'cuda' in self.device.type:
                    torch.cuda.synchronize(self.device)

                t += [time.time()]

                # Average timing
                if self.step < 2:
                    mean_dt = np.array(t[1:]) - np.array(t[:-1])
                else:
                    mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

                # Console display (only one per second)
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'e{:03d}-i{:04d} => L={:.3f} L_C={:.3f} L_I={:.3f} L_V={:.3f} L_VL2={:.3f} acc={:3.0f}% / t(ms): {:5.1f} {:5.1f} {:5.1f})'
                    print(message.format(self.epoch, self.step,
                                         loss.item(),
                                         net.center_loss.item(),
                                         net.instance_loss.item(),
                                         net.variance_loss.item(),
                                         net.variance_l2.item(),
                                         100 * acc,
                                         1000 * mean_dt[0],
                                         1000 * mean_dt[1],
                                         1000 * mean_dt[2]))

                # Log file
                if config.saving:
                    with open(join(config.saving_path, 'training.txt'), "a") as file:
                        message = '{:d} {:d} {:.3f} {:.3f} {:.3f} {:.3f}\n'
                        file.write(message.format(self.epoch,
                                                  self.step,
                                                  net.output_loss,
                                                  net.reg_loss,
                                                  acc,
                                                  t[-1] - t0))

                self.step += 1
                
            # Update epoch
            self.epoch += 1

            # Saving
            if config.saving:
                # Get current state dict
                save_dict = {'epoch': self.epoch,
                             'model_state_dict': net.state_dict(),
                             'optimizer_state_dict': self.optimizer.state_dict(),
                             'saving_path': config.saving_path}

                # Save current state of the network (for restoring purposes)
                checkpoint_path = join(checkpoint_directory, 'current_chkp.tar')
                torch.save(save_dict, checkpoint_path)

                # Save checkpoints occasionally
                if (self.epoch + 1) % config.checkpoint_gap == 0:
                    checkpoint_path = join(checkpoint_directory, 'chkp_{:04d}.tar'.format(self.epoch + 1))
                    torch.save(save_dict, checkpoint_path)


def evaluate_rotated(net, chkp_dir, config):
    net.train()

    chkp_path = join(chkp_dir, 'current_chkp.tar')
    pretrained_model = torch.load(chkp_path, map_location=torch.device("cpu"))
    net.load_state_dict(pretrained_model['model_state_dict'], strict=False)

    batch_path = join(chkp_dir, 'batch.tar')
    batch = torch.load(batch_path, map_location=torch.device("cpu"))

    sample_gpu = batch
    centers = sample_gpu['in_fts'][:,:,4:8]
    times = sample_gpu['in_fts'][:,:,8]

    if config.angle_z==0:
        # Evaluate on original pointcloud
        outputs, centers_output, var_output, embedding = net(sample_gpu['in_fts'][:,:,:4])
        loss = net.loss(
                outputs, centers_output, var_output, embedding, 
                sample_gpu['in_lbls'], sample_gpu['in_slbls'], centers, sample_gpu['in_pts'], times)                
        acc = net.accuracy(outputs.cpu(), sample_gpu['in_lbls'].cpu())
        ious = net.semantic_seg_metric(outputs.cpu(), sample_gpu['in_lbls'].cpu())
        nan_idx = torch.isnan(ious)
        ious[nan_idx] = 0.

        if config.saving:
            # Evaluation log file
            with open(join(chkp_dir, 'evaluation.txt'), "a") as file:
                file.write('\n')
                file.write('Original pointclouds: loss:{0:2.3f}, iou_mean:{1:2.3f}, accuracy:{2:.3f}%\n'.format(loss.item(), ious.mean(), 100*acc))
                # for i, iou in enumerate(ious):
                #     if isnan(iou):
                #         iou = 0.
                #     file.write('ious/{0}: {1:2.3f}\n'.format(i, iou))
                file.write('\n')

    else:
        # Evaluate on rotated pointcloud
        rotated_pc = rotate_pointcloud(config, sample_gpu['in_pts'], angle_range_z=config.angle_z)
        rotated_sample = sample_gpu['in_fts'][:,:,:4].contiguous()
        rotated_sample[:,:,:3] = torch.tensor(rotated_pc)
        # print(sample_gpu['in_fts'][0,0,:4])
        # print(rotated_sample[0,0,:4])
        # print(sample_gpu['in_pts'][0,0,:])
        # print(rotated_pc[0,0,:])

        outputs, centers_output, var_output, embedding = net(rotated_sample)
        loss = net.loss(
                outputs, centers_output, var_output, embedding, 
                sample_gpu['in_lbls'], sample_gpu['in_slbls'], centers, sample_gpu['in_pts'], times)                
        acc = net.accuracy(outputs.cpu(), sample_gpu['in_lbls'].cpu())
        ious = net.semantic_seg_metric(outputs.cpu(), sample_gpu['in_lbls'].cpu())               
        nan_idx = torch.isnan(ious)
        ious[nan_idx] = 0.

        if config.saving:
            # Evaluation log file
            with open(join(chkp_dir, 'evaluation.txt'), 'a') as file:
                file.write('Rotate: {0}, Angle: {1}\n'.format(config.eval_rotation, config.angle_z))
                file.write('Rotated pointclouds: loss:{0:2.3f}, iou_mean:{1:2.3f}, accuracy:{2:.3f}%\n'.format(loss.item(), ious.mean(), 100*acc))
                # for i, iou in enumerate(ious):
                #     if isnan(iou):
                #         iou = 0.
                #     file.write('ious/{0}: {1:2.3f}\n'.format(i, iou))
                file.write('\n')
    print('Rotate: {0}, Angle: {1}'.format(config.eval_rotation, config.angle_z))
    print('loss:{0:2.3f}, iou_mean:{1:2.3f}, accuracy:{2:.3f}%'.format(loss.item(), ious.mean(), 100*acc))



   