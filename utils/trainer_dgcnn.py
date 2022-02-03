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
from models.losses import instance_half_loss, isnan
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

    def __init__(self, net, config, chkp_path=None, resume_training=False, finetune=False, on_gpu=True):
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

        # self.optimizer = torch.optim.SGD(net.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=1e-4)
        self.optimizer = torch.optim.Adam(net.parameters(), lr=0.1, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

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
        
        if resume_training==True:
            pretrained_model = torch.load(chkp_path, map_location=self.device)
            self.epoch = pretrained_model['epoch']
            print('Resume training from epoch:', self.epoch)
            self.optimizer.load_state_dict(pretrained_model['optimizer_state_dict']) 
            if 'cuda' in self.device.type:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()
            net.load_state_dict(pretrained_model['model_state_dict'], strict=True)
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

        
        ##########################
        # Load pretrained model
        ##########################
        else:
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
            
            else:
                print('Not loading pretrained model, training from scratch')
        
        if config.lr_scheduler == True:
            # milestones = [200, 400, 600]
            # self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.45, verbose=False)
            # self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=2e-7, end_factor=1, total_iters=config.max_epoch, last_epoch=- 1, verbose=True)
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.max_epoch, eta_min=0, last_epoch=-1, verbose=True)
            # if resume_training == True:
            #     self.lr_scheduler.load_state_dict(pretrained_model['scheduler'])
        
        net.to(self.device)   

         # Path of the result folder
        if config.saving:
            if config.saving_path is None:
                config.saving_path = time.strftime('results/dgcnn/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
            if not exists(config.saving_path):
                makedirs(config.saving_path)
            config.save()


        self.train_logger = SummaryWriter(log_dir=config.saving_path+'/runs/train')
        if config.val_sem == True:
            self.val_logger = SummaryWriter(log_dir=config.saving_path+'/runs/validation')
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
                    self.train_logger.add_scalar('ious/{}'.format(i), iou, self.global_step)    
                # log mean IoU
                self.train_logger.add_scalar('ious/meanIoU', ious.mean(), self.global_step)    
                #AB: log into tensorbaord
                self.train_logger.add_scalar('Loss/CrossEntropy', loss.item(), self.global_step)
                self.train_logger.add_scalar('acc/train', acc*100, self.global_step)
                print('Epoch:{0:4d}, loss:{1:2.3f}, iou_mean:{2:2.3f}, accuracy:{3:.3f}'.format(self.global_step, loss.item(), ious.mean(), acc*100))
            
                self.global_step += 1
            
            break
    

    def train_overfit_4D(self, config, net=None, train_loader=None, val_loader=None, loss_type='4DPLSloss'):
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
            
        net.to(self.device)
        
        for epoch in range(config.max_epoch):
            net.train()
            # Training loop
            self.step = 0
            total_loss_mean =0.
            train_acc_mean = 0.
            train_iou_mean = 0.
            # ious = torch.zeros(19)
            # total_loss = 0. 
            # CE_loss = 0.
            # center_loss = 0.
            # instance_half_l = 0.
            # instance_loss = 0.
            # variance_loss = 0.

            for batch in train_loader:
                # move to device (GPU)
                sample_gpu ={}
                if 'cuda' in self.device.type:
                    for k, v in batch.items():
                        sample_gpu[k] = v.to(self.device)
                else:
                    sample_gpu = batch

                centers = sample_gpu['in_fts'][:,:,4:8]
                times = sample_gpu['in_fts'][:,:,8]

                self.optimizer.zero_grad()
                outputs, centers_output, var_output, embedding = net(sample_gpu['in_fts'][:,:,:4])
                
                if loss_type == 'CEloss':
                    labels = sample_gpu['in_lbls'].type(torch.LongTensor)
                    loss = net.cross_entropy_loss(outputs, labels)
                if loss_type == '4DPLSloss':
                    loss = net.loss(
                        outputs, centers_output, var_output, embedding, 
                        sample_gpu['in_lbls'], sample_gpu['in_slbls'], centers, sample_gpu['in_pts'], times)                
                
                acc = net.accuracy(outputs.cpu(), sample_gpu['in_lbls'].cpu())
                ious = net.semantic_seg_metric(outputs.cpu(), sample_gpu['in_lbls'].cpu())               
                nan_idx = torch.isnan(ious)
                ious[nan_idx] = 0.
                train_acc_mean += acc         
                meanIOU = ious.sum()/torch.count_nonzero(ious)
                train_iou_mean += meanIOU
                total_loss_mean += loss.item()
                loss.backward()
                # if config.grad_clip_norm > 0:
                #     # torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip_norm)
                #     torch.nn.utils.clip_grad_value_(net.parameters(), config.grad_clip_norm)
                self.optimizer.step()

                if 'cuda' in self.device.type:
                    torch.cuda.synchronize(self.device)

                for i, iou in enumerate(ious):
                    if isnan(iou):
                        iou = 0.
                    self.train_logger.add_scalar('ious/{}'.format(i), iou, self.global_step)    
                # log mean IoU
                self.train_logger.add_scalar('ious/meanIoU', meanIOU, self.global_step)    
                #AB: log into tensorbaord
                if loss_type == 'CEloss': # TODO: debug
                    self.train_logger.add_scalar('Loss/cross_entropy_loss', loss.item(), self.global_step)
                if loss_type == '4DPLSloss':
                    self.train_logger.add_scalar('Loss/total', loss.item(), self.global_step)
                    self.train_logger.add_scalar('Loss/cross_entropy', net.output_loss.item(), self.global_step)
                    self.train_logger.add_scalar('Loss/center_loss', net.center_loss.item(), self.global_step)
                    # self.train_logger.add_scalar('Loss/instance_half_loss', net.instance_half_loss.item(), self.global_step)
                    self.train_logger.add_scalar('Loss/instance_loss', net.instance_loss.item(), self.global_step)
                    self.train_logger.add_scalar('Loss/variance_loss', net.variance_loss.item(), self.global_step)
                    # self.train_logger.add_scalar('Loss/variance_loss', net.variance_l2.item(), self.global_step)               
                self.train_logger.add_scalar('acc/train', acc*100, self.global_step)
                
                # print('Epoch:{0:4d}, loss:{1:2.3f}, iou_mean:{2:2.3f}, accuracy:{3:.3f}'.format(epoch, loss.item(), meanIOU, acc*100))
                message = 'e{:03d}-i{:04d} => L={:.3f} L_C={:.3f} L_I={:.3f} L_V={:.3f} L_VL2={:.3f} meanIOU={:3.0f}% acc={:3.0f}%\n'
                print(message.format(epoch, self.step,
                                        loss.item(),
                                        net.center_loss.item(),
                                        net.instance_loss.item(),
                                        net.variance_loss.item(),
                                        net.variance_l2.item(),
                                        100 * meanIOU, 
                                        100 * acc,))

                # Log file
                if config.saving:
                    with open(join(config.saving_path, 'training.txt'), "a") as file:
                        file.write(message.format(epoch, self.step,
                                        loss.item(),
                                        net.center_loss.item(),
                                        net.instance_loss.item(),
                                        net.variance_loss.item(),
                                        net.variance_l2.item(),
                                        100 * meanIOU,
                                        100 * acc,))
                
                self.step +=1
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

            # log mean acc and mean iou for an epoch
            total_loss_mean = total_loss_mean/len(train_loader)
            train_acc_mean = train_acc_mean/len(train_loader)
            train_iou_mean = train_iou_mean/len(train_loader)
            self.train_logger.add_scalar('report/acc', train_acc_mean*100, epoch)
            self.train_logger.add_scalar('report/mean_ious', train_iou_mean*100, epoch)
            self.train_logger.add_scalar('report/mean_total_loss', total_loss_mean, epoch)
            
            # update lr_scheduler every epoch
            if config.lr_scheduler == True:        
                self.lr_scheduler.step()
            
            # save the best model
            if train_acc_mean > best_train_acc:
                best_train_acc = train_acc_mean
                checkpoint_path = join(checkpoint_directory, 'best_chkp.tar')
                torch.save(save_dict, checkpoint_path)



            if config.val_sem == True and epoch%5 == 0:
                print('Validation process...')
                with torch.no_grad():
                    val_acc_mean = 0.
                    val_iou_mean = 0.
                    val_step = self.global_step
                    for batch in val_loader:
                        # move to device (GPU)
                        sample_gpu ={}
                        if 'cuda' in self.device.type:
                            for k, v in batch.items():
                                sample_gpu[k] = v.to(self.device)
                        else:
                            sample_gpu = batch
            

                        centers = sample_gpu['in_fts'][:,:,4:8]
                        times = sample_gpu['in_fts'][:,:,8]

                        outputs, centers_output, var_output, embedding = net(sample_gpu['in_fts'][:,:,:3])
                        if loss_type == 'CEloss': 
                            labels = sample_gpu['in_lbls'].type(torch.LongTensor)
                            loss = net.cross_entropy_loss(outputs, labels)
                        if loss_type == '4DPLSloss':
                            loss = net.loss(
                                outputs, centers_output, var_output, embedding, 
                                sample_gpu['in_lbls'], sample_gpu['in_slbls'], centers, sample_gpu['in_pts'], times)                

                        acc = net.accuracy(outputs.cpu(), sample_gpu['in_lbls'].cpu())
                        ious = net.semantic_seg_metric(outputs.cpu(), sample_gpu['in_lbls'].cpu())               
                        nan_idx = torch.isnan(ious)
                        ious[nan_idx] = 0.
                        val_acc_mean+=acc
                        meanIOU = ious.sum()/torch.count_nonzero(ious)
                        val_iou_mean += meanIOU
                        if 'cuda' in self.device.type:
                            torch.cuda.synchronize(self.device)

                        for i, iou in enumerate(ious):
                            if isnan(iou):
                                iou = 0.
                            self.val_logger.add_scalar('ious/{}'.format(i), iou, val_step)    
                        # log mean IoU
                        self.val_logger.add_scalar('ious/meanIoU', meanIOU, val_step)    
                        #AB: log into tensorbaord
                        if loss_type == 'CEloss': # TODO: debug
                            self.val_logger.add_scalar('Loss/cross_entropy_loss', loss.item(), val_step)
                        if loss_type == '4DPLSloss':
                            self.val_logger.add_scalar('Loss/total', loss.item(), val_step)
                            self.val_logger.add_scalar('Loss/cross_entropy', net.output_loss.item(), val_step)
                            self.val_logger.add_scalar('Loss/center_loss', net.center_loss.item(), val_step)
                            # self.val_logger.add_scalar('Loss/instance_half_loss', net.instance_half_loss.item(), val_step)
                            self.val_logger.add_scalar('Loss/instance_loss', net.instance_loss.item(), val_step)
                            self.val_logger.add_scalar('Loss/variance_loss', net.variance_loss.item(), val_step)
                            # self.val_logger.add_scalar('Loss/variance_loss', net.variance_l2.item(), val_step)               
                        self.val_logger.add_scalar('acc/train', acc*100, val_step)
                        
                        print('Validation: Epoch:{0:4d}, loss:{1:2.3f}, iou_mean:{2:2.3f}, accuracy:{3:.3f}'.format(epoch, loss.item(), meanIOU, acc*100))

                        # Log file
                        if config.saving:
                            with open(join(config.saving_path, 'training.txt'), "a") as file:
                                file.write('Validation: Epoch:{0:4d}, loss:{1:2.3f}, iou_mean:{2:2.3f}, accuracy:{3:.3f}\n'.format(epoch, loss.item(), meanIOU, acc*100))

                        val_step += 1
                    
                val_acc_mean = val_acc_mean/len(val_loader)
                val_iou_mean = val_iou_mean/len(val_loader)
                self.val_logger.add_scalar('report/acc', val_acc_mean*100, epoch)
                self.val_logger.add_scalar('report/mean_ious', val_iou_mean*100, epoch)
                if config.saving:
                    with open(join(config.saving_path, 'training.txt'), "a") as file:
                        file.write('Training_acc:{0:.3f}, Validation_acc:{1:.3f}\n'.format(train_acc_mean*100,val_acc_mean*100))



    def train(self, config, net=None, train_loader=None, val_loader=None, loss_type='4DPLSloss'):
        """
        Train the model on a particular dataset.
        """
        ################
        # Initialization
        ################

        if config.saving:
            # Training log file
            with open(join(config.saving_path, 'training.txt'), "w") as file:
                if config.resume_training:
                    file.write('Resume training from epoch:{0:4d}\n'.format(self.epoch))
                else:
                    file.write('Start training... \n')
            # Checkpoints directory
            checkpoint_directory = join(config.saving_path, 'checkpoints')
            if not exists(checkpoint_directory):
                makedirs(checkpoint_directory)
        else:
            checkpoint_directory = None

        net.to(self.device)
        best_train_acc = 0.
        if config.resume_training:
            self.global_step = (self.epoch+1)*len(train_loader)

        while self.epoch < config.max_epoch: 
            net.train()
            # Training loop
            self.step = 0
            total_loss_mean =0.
            train_acc_mean = 0.
            train_iou_mean = 0.

            for batch in train_loader:
                # move to device (GPU)
                sample_gpu ={}
                if 'cuda' in self.device.type:
                    for k, v in batch.items():
                        sample_gpu[k] = v.to(self.device)
                else:
                    sample_gpu = batch

                centers = sample_gpu['in_fts'][:,:,4:8]
                times = sample_gpu['in_fts'][:,:,8]

                self.optimizer.zero_grad()

                outputs, centers_output, var_output, embedding = net(sample_gpu['in_fts'][:,:,:4])
                
                if loss_type == 'CEloss':
                    labels = sample_gpu['in_lbls'].type(torch.LongTensor)
                    loss = net.cross_entropy_loss(outputs, labels)
                if loss_type == '4DPLSloss':
                    loss = net.loss(
                        outputs, centers_output, var_output, embedding, 
                        sample_gpu['in_lbls'], sample_gpu['in_slbls'], centers, sample_gpu['in_pts'], times)                

                acc = net.accuracy(outputs.cpu(), sample_gpu['in_lbls'].cpu())
                ious = net.semantic_seg_metric(outputs.cpu(), sample_gpu['in_lbls'].cpu())               
                nan_idx = torch.isnan(ious)
                ious[nan_idx] = 0.
                train_acc_mean += acc         
                meanIOU = ious.sum()/torch.count_nonzero(ious)
                train_iou_mean += meanIOU
                total_loss_mean += loss.item()
                loss.backward()
                if config.grad_clip_norm > 0:
                    # torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip_norm)
                    torch.nn.utils.clip_grad_value_(net.parameters(), config.grad_clip_norm)
                self.optimizer.step()
                
                if 'cuda' in self.device.type:
                    torch.cuda.synchronize(self.device)

                for i, iou in enumerate(ious):
                    if isnan(iou):
                        iou = 0.
                    self.train_logger.add_scalar('ious/{}'.format(i), iou, self.global_step)    
                # log mean IoU
                self.train_logger.add_scalar('ious/meanIoU', meanIOU, self.global_step)    
                #AB: log into tensorbaord
                if loss_type == 'CEloss': # TODO: debug
                    self.train_logger.add_scalar('Loss/cross_entropy_loss', loss.item(), self.global_step)
                if loss_type == '4DPLSloss':
                    self.train_logger.add_scalar('Loss/total', loss.item(), self.global_step)
                    self.train_logger.add_scalar('Loss/cross_entropy', net.output_loss.item(), self.global_step)
                    self.train_logger.add_scalar('Loss/center_loss', net.center_loss.item(), self.global_step)
                    # self.train_logger.add_scalar('Loss/instance_half_loss', net.instance_half_loss.item(), self.global_step)
                    self.train_logger.add_scalar('Loss/instance_loss', net.instance_loss.item(), self.global_step)
                    self.train_logger.add_scalar('Loss/variance_loss', net.variance_loss.item(), self.global_step)
                    # self.train_logger.add_scalar('Loss/variance_l2_loss', net.variance_l2.item(), self.global_step)               
                self.train_logger.add_scalar('acc/train', acc*100, self.global_step)
                
                # print('Epoch:{0:4d}, loss:{1:2.3f}, iou_mean:{2:2.3f}, accuracy:{3:.3f}'.format(self.epoch, loss.item(), meanIOU, acc*100))
                message = 'e{:04d}-i{:04d} => L={:.3f} L_CE={:.3f} L_C={:.3f} L_I={:.3f} L_V={:.3f} meanIOU={:3.0f}% acc={:3.0f}%\n'
                print(message.format(self.epoch, self.step,
                                        loss.item(),
                                        net.output_loss.item(),
                                        net.center_loss.item(),
                                        net.instance_loss.item(),
                                        net.variance_loss.item(),
                                        100 * meanIOU, 
                                        100 * acc,))

                # Log file
                if config.saving:
                    with open(join(config.saving_path, 'training.txt'), "a") as file:
                        file.write(message.format(self.epoch, self.step,
                                        loss.item(),
                                        net.output_loss.item(),
                                        net.center_loss.item(),
                                        net.instance_loss.item(),
                                        net.variance_loss.item(),
                                        100 * meanIOU,
                                        100 * acc,))

                self.step +=1
                self.global_step += 1

                # Saving
                if config.saving:
                    # Get current state dict
                    save_dict = {'epoch': self.epoch,
                                'model_state_dict': net.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'saving_path': config.saving_path}
                    if config.lr_scheduler == True:
                        save_dict.update({'lr_scheduler': self.lr_scheduler.state_dict()})

                    # Save current state of the network (for restoring purposes)
                    checkpoint_path = join(checkpoint_directory, 'current_chkp.tar')
                    torch.save(save_dict, checkpoint_path)
                    
                    # Save checkpoints occasionally
                    if (self.epoch + 1) % config.checkpoint_gap == 0:
                        checkpoint_path = join(checkpoint_directory, 'chkp_{:04d}.tar'.format(self.epoch + 1))
                        torch.save(save_dict, checkpoint_path)
            
            # log mean acc and mean iou for an epoch
            total_loss_mean = total_loss_mean/len(train_loader)
            train_acc_mean = train_acc_mean/len(train_loader)
            train_iou_mean = train_iou_mean/len(train_loader)
            self.train_logger.add_scalar('report/acc', train_acc_mean*100, self.epoch)
            self.train_logger.add_scalar('report/mean_ious', train_iou_mean*100, self.epoch)
            self.train_logger.add_scalar('report/mean_total_loss', total_loss_mean, self.epoch)
            if config.saving:
                with open(join(config.saving_path, 'training.txt'), "a") as file:
                    file.write('Epoch:{0:4d}, Training_acc:{1:.3f}, Training_meanIOU:{2:.3f}\n'.format(self.epoch, train_acc_mean*100, train_iou_mean*100))

            # update lr_scheduler every epoch
            if config.lr_scheduler == True:        
                self.lr_scheduler.step()
            
            # save the best model
            if train_acc_mean > best_train_acc:
                best_train_acc = train_acc_mean
                checkpoint_path = join(checkpoint_directory, 'best_chkp.tar')
                torch.save(save_dict, checkpoint_path)

            # validation process
            if config.val_sem == True and self.epoch%5 == 0:
                self.val_sem_seg(config, net, val_loader, self.epoch, loss_type)
            
            self.epoch += 1


    def val_sem_seg(self, config, net, val_loader, epoch, loss_type):
        print('Semantic segmentation validation process... ')
        with torch.no_grad():
            val_acc_mean = 0.
            val_iou_mean = 0.
            val_step = self.global_step
            for batch in val_loader:
                # move to device (GPU)
                sample_gpu ={}
                if 'cuda' in self.device.type:
                    for k, v in batch.items():
                        sample_gpu[k] = v.to(self.device)
                else:
                    sample_gpu = batch

                centers = sample_gpu['in_fts'][:,:,4:8]
                times = sample_gpu['in_fts'][:,:,8]

                outputs, centers_output, var_output, embedding = net(sample_gpu['in_fts'][:,:,:3])
                if loss_type == 'CEloss': 
                    labels = sample_gpu['in_lbls'].type(torch.LongTensor)
                    loss = net.cross_entropy_loss(outputs, labels)
                if loss_type == '4DPLSloss':
                    loss = net.loss(
                        outputs, centers_output, var_output, embedding, 
                        sample_gpu['in_lbls'], sample_gpu['in_slbls'], centers, sample_gpu['in_pts'], times)                

                acc = net.accuracy(outputs.cpu(), sample_gpu['in_lbls'].cpu())
                ious = net.semantic_seg_metric(outputs.cpu(), sample_gpu['in_lbls'].cpu())               
                nan_idx = torch.isnan(ious)
                ious[nan_idx] = 0.
                val_acc_mean+=acc
                meanIOU = ious.sum()/torch.count_nonzero(ious)
                val_iou_mean += meanIOU
                
                if 'cuda' in self.device.type:
                    torch.cuda.synchronize(self.device)

                for i, iou in enumerate(ious):
                    if isnan(iou):
                        iou = 0.
                    self.val_logger.add_scalar('ious/{}'.format(i), iou, val_step)    
                # log mean IoU
                self.val_logger.add_scalar('ious/meanIoU', meanIOU, val_step)    
                #AB: log into tensorbaord
                if loss_type == 'CEloss': 
                    self.val_logger.add_scalar('Loss/cross_entropy_loss', loss.item(), val_step)
                if loss_type == '4DPLSloss':
                    self.val_logger.add_scalar('Loss/total', loss.item(), val_step)
                    self.val_logger.add_scalar('Loss/cross_entropy', net.output_loss.item(), val_step)
                    self.val_logger.add_scalar('Loss/center_loss', net.center_loss.item(), val_step)
                    # self.val_logger.add_scalar('Loss/instance_half_loss', net.instance_half_loss.item(), val_step)
                    self.val_logger.add_scalar('Loss/instance_loss', net.instance_loss.item(), val_step)
                    self.val_logger.add_scalar('Loss/variance_loss', net.variance_loss.item(), val_step)
                    # self.val_logger.add_scalar('Loss/variance_loss', net.variance_l2.item(), val_step)               
                self.val_logger.add_scalar('acc/train', acc*100, val_step)
                
                message = 'Validation: e{:04d}-i{:04d} => L={:.3f} L_CE={:.3f} L_C={:.3f} L_I={:.3f} L_V={:.3f} meanIOU={:3.0f}% acc={:3.0f}%\n'
                print(message.format(epoch, val_step-self.global_step,
                                        loss.item(),
                                        net.output_loss.item(),
                                        net.center_loss.item(),
                                        net.instance_loss.item(),
                                        net.variance_loss.item(),
                                        100 * meanIOU, 
                                        100 * acc,))

                # Log file
                if config.saving:
                    with open(join(config.saving_path, 'training.txt'), "a") as file:
                        file.write(message.format(epoch, val_step-self.global_step,
                                        loss.item(),
                                        net.output_loss.item(),
                                        net.center_loss.item(),
                                        net.instance_loss.item(),
                                        net.variance_loss.item(),
                                        100 * meanIOU,
                                        100 * acc,))

                val_step += 1
        
        val_acc_mean = val_acc_mean/len(val_loader)
        val_iou_mean = val_iou_mean/len(val_loader)
        self.val_logger.add_scalar('report/acc', val_acc_mean*100, epoch)
        self.val_logger.add_scalar('report/mean_ious', val_iou_mean*100, epoch)
        if config.saving:
            with open(join(config.saving_path, 'training.txt'), "a") as file:
                file.write('Epoch:{0:4d}, Validation_acc:{1:.3f}, Validation_meanIOU:{2:.3f}\n'.format(epoch, val_acc_mean*100, val_iou_mean*100))



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



   