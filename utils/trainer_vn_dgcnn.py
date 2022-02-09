import logging
from multiprocessing.sharedctypes import Value

import hydra
import open3d as o3d
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pickle
import os
from os import makedirs, remove
from os.path import exists, join
import time
import sys
from models.losses import isnan
from utils.debugging import d_print

# PLY reader
# from utils.ply import read_ply, write_ply

# Metrics
from utils.metrics import IoU_from_confusions, fast_confusion
from utils.config import Config, bcolors
# from sklearn.neighbors import KDTree

# from models.blocks import KPConv

logger = logging.getLogger(__name__)

def get_lr(optimizer):
    ret = []
    for param_group in optimizer.param_groups:
        ret.append(str(param_group['lr']))
    return ", ".join(ret)


def visualize_semantic_acc(pc, probs, labels, label_values, ignored_labels, abs_path):
    """ Write colored point cloud into val_preds folder
    Args:
        outputs: predicted semantic labels
        labels: gt data
    Return:
        None
    """
    assert pc.shape[1] == 3
    d_print(pc.shape)
    # get correctly classified labels
    # plot them in green
    # plot wrongly classified in red
    probs = probs.view(-1,19)
    # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
    for l_ind, label_value in enumerate(label_values):
        if label_value in ignored_labels:
            probs = np.insert(probs, l_ind, 0, axis=1)
    preds = label_values[np.argmax(probs, axis=1)]
    preds = torch.from_numpy(preds)
    # preds.to(outputs.device)



    # predicted = torch.argmax(outputs, dim=1)
    total = preds.size(0)
    colors = np.zeros((total, 3))
    colors[:,0] = 1 # mark all as incorrect
    correct = np.where(preds == labels.squeeze(0))
    # d_print(correct.shape)
    colors[correct] = np.array([0,1,0]) # mark as green

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud("{}.ply".format(abs_path), pcd)
    if os.path.exists("{}.ply".format(abs_path)):
        d_print("successful write on desk", bcolors.OKGREEN)
    else:
        d_print("Invalid Point CLoud write", bcolors.FAIL)

    # write another ply file for semantics of the things class
    things_idx = np.where(np.logical_and(preds > 0,preds < 9))
    pc = pc[things_idx]
    d_print("things shape: {}".format(pc.shape))
    colors = np.zeros((pc.shape[0], 3))
    colors[:] = np.array([0,1,1])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.colors = o3d.utility.Vector3dVector(colors)


    o3d.io.write_point_cloud("{}_{}.ply".format(abs_path, "things"), pcd)
    if os.path.exists("{}_{}.ply".format(abs_path, "things")):
        d_print("successful write on desk", bcolors.OKGREEN)
    else:
        d_print("Invalid Point CLoud write", bcolors.FAIL)




class ModelTrainerVNDGCNN:

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

        # Path of the result folder
        if config.saving:
            if config.saving_path is None:
                raise ValueError("shouldn't be None !")
                config.saving_path = config.saving_path
                config.saving_path = hydra.utils.to_absoulte_path(config.saving_path)
                d_print("Saving path is {}".format(config.saving_path))
            if not exists(config.saving_path):
                makedirs(config.saving_path)
            # config.save()

        # Writer will output to ./runs/ directory by default
        self.train_logger = SummaryWriter(log_dir="vn_dgcnn_train")
        self.val_logger = SummaryWriter(log_dir="vn_dgcnn_val")
        # Epoch index
        self.epoch = 0
        self.step = 0
        self.global_step = 0

        var_params = [v for k, v in net.named_parameters() if 'head_var' in k]
        # Optimizer with specific learning rate for deformable KPConv
        # deform_params = [v for k, v in net.named_parameters() if 'offset' in k and not 'head_var' in k]
        other_params = [v for k, v in net.named_parameters() if 'offset' not in k and not 'head_var' in k]
        # deform_lr = config.learning_rate * config.deform_lr_factor
        var_lr =  1e-3
        if config.optimizer.name == 'sgd':
            logger.info("SGD Optimizer")
            self.optimizer = torch.optim.SGD(
                [
                    {'params': other_params},
                    {'params': var_params, 'lr': var_lr},
                #   {'params': deform_params, 'lr': deform_lr},
                ],
                lr=config.optimizer.learning_rate,
                momentum=config.momentum,
                weight_decay=config.optimizer.weight_decay
            )
        elif config.optimizer.name == 'adam':
            logger.info("Adam Optimizer")
            self.optimizer = torch.optim.Adam(
                [
                    {'params': other_params},
                    {'params': var_params, 'lr': var_lr},
                ],
                lr=config.optimizer.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=config.optimizer.weight_decay
            )
        elif config.optimizer.name == 'adamw':
            logger.info("AdamW Optimizer")
            self.optimizer= torch.optim.AdamW(
                [
                    {'params': other_params},
                    {'params': var_params, 'lr': var_lr},
                ],
                lr=config.optimizer.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=config.optimizer.weight_decay
            )
        else:
            raise NotImplementedError

        # lr_scheduler = config.getattr("lr_scheduler", None)
        # if lr_scheduler:
        #     self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #         self.optimizer, 
        #         milestones=config.lr_scheduler.milestones, 
        #         gamma=config.lr_scheduler.gamma
        #         )
        #MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)

        # Choose to train on CPU or GPU
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        net.to(self.device)

        ##########################
        # Load previous checkpoint
        ##########################

        # if (chkp_path is not None):
        #     if finetune:
        #         checkpoint = torch.load(chkp_path)
        #         if checkpoint['model_state_dict']['head_var.mlp.weight'].shape[0] == \
        #                 checkpoint['model_state_dict']['head_var.mlp.weight'].shape[1] and config.free_dim != 0:
        #             checkpoint['model_state_dict']['head_var.mlp.weight'] = net.head_var.mlp.weight
        #             checkpoint['model_state_dict']['head_var.batch_norm.bias'] = net.head_var.batch_norm.bias

        #         if checkpoint['model_state_dict']['head_var.mlp.weight'].shape[0] -  \
        #             checkpoint['model_state_dict']['head_var.mlp.weight'].shape[1] != config.free_dim:
        #             checkpoint['model_state_dict']['head_var.mlp.weight'] = net.head_var.mlp.weight
        #             checkpoint['model_state_dict']['head_var.batch_norm.bias'] = net.head_var.batch_norm.bias

        #         if checkpoint['model_state_dict']['head_var.mlp.weight'].shape[0] != net.head_var.mlp.weight.shape[0] \
        #                 or checkpoint['model_state_dict']['head_var.mlp.weight'].shape[1] !=net.head_var.mlp.weight.shape[1]:
        #             checkpoint['model_state_dict']['head_var.mlp.weight'] = net.head_var.mlp.weight
        #             checkpoint['model_state_dict']['head_var.batch_norm.bias'] = net.head_var.batch_norm.bias

        #         if config.reinit_var:
        #             checkpoint['model_state_dict']['head_var.mlp.weight'] = net.head_var.mlp.weight
        #             checkpoint['model_state_dict']['head_var.batch_norm.bias'] = net.head_var.batch_norm.bias

        #         net.load_state_dict(checkpoint['model_state_dict'])
        #         net.train()
        #         print("Model restored and ready for finetuning.")
        #     else:
        #         checkpoint = torch.load(chkp_path)
        #         if config.reinit_var:
        #             checkpoint['model_state_dict']['head_var.mlp.weight'] = net.head_var.mlp.weight
        #             checkpoint['model_state_dict']['head_var.batch_norm.bias'] = net.head_var.batch_norm.bias
        #         if not config.reinit_var:
        #             self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #         net.load_state_dict(checkpoint['model_state_dict'])
        #         self.epoch = checkpoint['epoch']
        #         net.train()
        #         print("Model and training state restored.")


        # Path of the result folder
        # if config.saving:
        #     if config.saving_path is None:
        #         config.saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
        #     if not exists(config.saving_path):
        #         makedirs(config.saving_path)
        #     config.save()

        # return

    # Training main method
    # ------------------------------------------------------------------------------------------------------------------

    def train(self, net, train_loader, val_loader, config):
        """
        Train the model on a particular dataset.
        """
        ################
        # Initialization
        ################

        # if config.saving:
        #     # Training log file
        #     with open(join(config.saving_path, 'training.txt'), "w") as file:
        #         file.write('epochs steps out_loss offset_loss train_accuracy time\n')

        #     # Killing file (simply delete this file when you want to stop the training)
        #     PID_file = join(config.saving_path, 'running_PID.txt')
        #     if not exists(PID_file):
        #         with open(PID_file, "w") as file:
        #             file.write('Launched with PyCharm')

        #     # Checkpoints directory
        #     checkpoint_directory = join(config.saving_path, 'checkpoints')
        #     if not exists(checkpoint_directory):
        #         makedirs(checkpoint_directory)
        # else:
        #     checkpoint_directory = None
        #     PID_file = None

        # Start training loop
        for epoch in range(config.max_epoch):
            # d_print("starting epoch {}, lr is {}".format(epoch, get_lr(self.optimizer)))
            net.train()
            self.step = 0
            for batch in train_loader:
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # move to device (GPU)
                sample_gpu = {}
                if 'cuda' in self.device.type:
                    for k, v in batch.items():
                        sample_gpu[k] = v.to(self.device)
                else:
                    sample_gpu = batch
                
                # extract data used in loss functions 
                # split centers and times -> original feats composition [x,y,z,r,c1,c2,c3,objectness,time]
                centers = sample_gpu['in_fts'][:,:,4:8]
                times = sample_gpu['in_fts'][:,:,8]
                # prepare inputs for PointNet Architecture
                sample_gpu['in_fts'] = sample_gpu['in_fts'][:,:,:3].transpose(2,1) #TODO: fix the feature handling with the dataloader !
                # Forward pass
                outputs, centers_output, var_output, embedding = net(sample_gpu['in_fts'])
                # getting loss 
                if config.task == 'pls':
                    loss = net.loss(
                        outputs, centers_output, var_output, embedding, 
                        sample_gpu['in_lbls'], sample_gpu['in_slbls'], centers, sample_gpu['in_pts'], times)
                    # log specific losses values
                    self.train_logger.add_scalar('Loss/total', loss.item(), self.global_step)
                    self.train_logger.add_scalar('Loss/cross_entropy', net.output_loss.item(), self.global_step)
                    self.train_logger.add_scalar('Loss/center_loss', net.center_loss.item(), self.global_step)
                    self.train_logger.add_scalar('Loss/instance_half_loss', net.instance_half_loss.item(), self.global_step)
                    self.train_logger.add_scalar('Loss/instance_loss', net.instance_loss.item(), self.global_step)
                    self.train_logger.add_scalar('Loss/variance_loss', net.variance_loss.item(), self.global_step)
                    self.train_logger.add_scalar('Loss/variance_l2', net.variance_l2.item(), self.global_step)
                elif config.task == 'sem_seg':
                    loss = net.cross_entropy_loss(outputs, sample_gpu['in_lbls'])
                    self.train_logger.add_scalar('Loss/cross_entropy', net.output_loss.item(), self.global_step)
                else:
                    raise ValueError("unknow requested task")
                acc = net.accuracy(outputs.cpu(), sample_gpu['in_lbls'].cpu())
                
                
                
                ious, meanIoU = net.semantic_seg_metric(outputs.cpu(), sample_gpu['in_lbls'].cpu())
                for i, iou in enumerate(ious):
                    self.train_logger.add_scalar('iou/{}'.format(i), iou, self.global_step)    
                # log mean IoU
                self.train_logger.add_scalar('iou/mIoU', meanIoU, self.global_step)    
                
                
                # ious = net.fast_semantic_seg_metric(
                #     outputs.detach().cpu(), 
                #     sample_gpu['in_lbls'].cpu(), 
                #     train_loader.dataset.label_values, 
                #     train_loader.dataset.ignored_labels
                # )
                # # d_print(ious)
                # # nan_idx = torch.isnan(ious)
                # if np.any(np.isnan(ious)): raise ValueError("should not be nan !!")
                # # ious[nan_idx] = 0.
                # for i, iou in enumerate(ious):
                #     # if isnan(iou):
                #         # iou = 0.
                #     self.train_logger.add_scalar('iou/{}'.format(i), iou, self.global_step)    
                # # log mean IoU
                # self.train_logger.add_scalar('iou/mIoU', ious.mean(), self.global_step)    
                #AB: log into tensorbaord
                self.train_logger.add_scalar('acc', acc*100, self.global_step)
                

                # t += [time.time()]

                # Backward + optimize
                loss.backward()

                if config.grad_clip_norm > 0:
                    # torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip_norm)
                    torch.nn.utils.clip_grad_value_(net.parameters(), config.grad_clip_norm)
                self.optimizer.step()
               
                if config.task == 'pls':
                    message = 'e{:03d}-i{:04d} => L={:.3f} L_C={:.3f} L_I={:.3f} L_V={:.3f} L_VL2={:.3f} acc={:3.0f}'
                    logger.info(message.format(self.epoch, self.step,
                                            loss.item(),
                                            net.center_loss.item(),
                                            net.instance_loss.item(),
                                            net.variance_loss.item(),
                                            net.variance_l2.item(),
                                            100 * acc,
                                            #  1000 * mean_dt[0],
                                            #  1000 * mean_dt[1],
                                            #  1000 * mean_dt[2]
                                            ))
                elif config.task == 'sem_seg':
                    message = 'e{:03d}-i{:04d} => L={:.3f} acc={:3.0f}'
                    logger.info(message.format(self.epoch, self.step,
                                            loss.item(),
                                            # net.center_loss.item(),
                                            # net.instance_loss.item(),
                                            # net.variance_loss.item(),
                                            # net.variance_l2.item(),
                                            100 * acc,
                                            #  1000 * mean_dt[0],
                                            #  1000 * mean_dt[1],
                                            #  1000 * mean_dt[2]
                                            ))
                else:
                    raise ValueError("unknown task")

                # mini validation check
                if self.global_step % 20 == 0:
                    if config.task == 'sem_seg':
                        self.validation_sem_seg(net, val_loader, config)
                    elif config.task == 'pls':
                        pass
                        # self.validation_pls(net, val_loader, config)
                    else:
                        raise NotImplementedError
                    #TODO: implement the mini validation function 

                # net.train()
                self.step += 1
                self.global_step += 1

            # End of epoch

            # Update learning rate
            # if self.epoch in config.lr_decays:
            #     for param_group in self.optimizer.param_groups:
            #         param_group['lr'] *= config.lr_decays[self.epoch]

            # Update epoch
            self.epoch += 1
            # self.lr_scheduler.step()

            # Saving
            # if config.saving:
            #     # Get current state dict
            #     save_dict = {'epoch': self.epoch,
            #                  'model_state_dict': net.state_dict(),
            #                  'optimizer_state_dict': self.optimizer.state_dict(),
            #                  'saving_path': config.saving_path}

            #     # Save current state of the network (for restoring purposes)
            #     checkpoint_path = join(checkpoint_directory, 'current_chkp.tar')
            #     torch.save(save_dict, checkpoint_path)

            #     # Save checkpoints occasionally
            #     if (self.epoch + 1) % config.checkpoint_gap == 0:
            #         checkpoint_path = join(checkpoint_directory, 'chkp_{:04d}.tar'.format(self.epoch + 1))
            #         torch.save(save_dict, checkpoint_path)

            # TODO: main validation after epoch
            # if epoch % 10 == 0:
            #     # Validation
            #     net.eval()
            #     self.optimizer.zero_grad()
            #     self.validation(net, val_loader, config)
                

        print('Finished Training')
        return

    # Validation methods
    # ------------------------------------------------------------------------------------------------------------------

    def validation_pls(self, net, val_loader, config):
        """ Validation function, gives ious and accuracy
        """
        logger.info("Launching validation function")
        softmax = torch.nn.Softmax(1)
        # Create folder for validation predictions, Visualizations files
        if not exists(join(config.saving_path, 'val_preds')):
            makedirs(join(config.saving_path, 'val_preds'))
        # Number of classes including ignored labels
        nc_tot = val_loader.dataset.num_classes
        predictions = []
        targets = []
        # inds = []
        val_i = 0
        c_ious = []
        s_ious = []
        t_acc = []
        # loss buffers
        t_total_loss = []
        t_cross_entropy_loss = []
        t_center_loss = []
        t_instance_half_loss = []
        t_instance_loss = []
        t_variance_loss = []
        t_variance_l2_loss = []
        # o_scores = []
        # Start validation loop
        for i, batch in enumerate(val_loader):
            logger.info("Evaluating batch: {} / {}".format(i, len(val_loader)))
            batch_size = batch['in_fts'].shape[0]
            num_points = batch['in_fts'].shape[1]
            s_ind = batch['s_ind'][0] # under assumption only one sample per batch
            f_ind = batch['f_ind'][0]

            # Forward pass
            with torch.no_grad():
                sample_gpu = {}
                if 'cuda' in self.device.type:
                    for k, v in batch.items():
                        if k in ['in_pts', 'in_fts', 'in_lbls', 'in_slbls']:
                            sample_gpu[k] = v.to(self.device)
                else:
                    sample_gpu = batch
                centers = sample_gpu['in_fts'][:,:,4:8]
                times = sample_gpu['in_fts'][:,:,8]
                # save for visualization
                viz_pc = sample_gpu['in_fts'][:,:,:3].squeeze(0).cpu().numpy()
                # prepare inputs for PointNet Architecture
                sample_gpu['in_fts'] = sample_gpu['in_fts'][:,:,:3].transpose(2,1) #TODO: fix the feature handling with the dataloader !
                # Forward pass
                outputs, centers_output, var_output, embedding = net(sample_gpu['in_fts'])
                # get the loss and log into tensorboard
                loss = net.loss(
                    outputs, centers_output, var_output, embedding, 
                    sample_gpu['in_lbls'], sample_gpu['in_slbls'], centers, sample_gpu['in_pts'], times)
                t_total_loss.append(loss.item())
                t_cross_entropy_loss.append(net.output_loss.item())
                t_center_loss.append(net.center_loss.item())
                t_instance_half_loss.append(net.instance_half_loss.item())
                t_instance_loss.append(net.instance_loss.item())
                t_variance_loss.append(net.variance_loss.item())
                t_variance_l2_loss.append(net.variance_l2.item())
                # get the performance over the subsampled point cloud (the one used )
                
                # get the performance over the subsampled point cloud (the one used in the backbone !)
                acc = net.accuracy(outputs.cpu(), sample_gpu['in_lbls'].cpu())
                t_acc.append(acc)
                # create name and path for the accuracy point cloud
                filename = '{:s}_{:07d}'.format(val_loader.dataset.sequences[s_ind], f_ind)
                visualize_semantic_acc(
                    viz_pc,
                    outputs.cpu(), 
                    sample_gpu['in_lbls'].cpu(),
                    val_loader.dataset.label_values,
                    val_loader.dataset.ignored_labels,
                    os.path.join(config.saving_path, 'val_preds', filename)
                )
                # TODO: get mIOU metrics for subsampled point cloud

                probs = softmax(outputs.view(-1, 19)).cpu().detach().numpy()
                #AB: why ignoring the instance labels before epoch 50 ?!
                if self.epoch > -1: # TODO: change later to 50
                    # Insert false columns for ignored labels
                    for l_ind, label_value in enumerate(val_loader.dataset.label_values):
                        if label_value in val_loader.dataset.ignored_labels:
                            probs = np.insert(probs, l_ind, 0, axis=1)
                    preds = val_loader.dataset.label_values[np.argmax(probs, axis=1)]
                    preds = torch.from_numpy(preds)
                    preds.to(outputs.device)
                    # d_print(embedding.squeeze(0).shape)
                    # d_print(batch['in_pts'][0].to(embedding.device).shape)
                    # d_print(times.squeeze(0).unsqueeze(1).to(embedding.device).shape)
                    ins_preds = net.ins_pred(
                        preds, 
                        centers_output.squeeze(0), 
                        var_output.squeeze(0), 
                        embedding.squeeze(0), 
                        batch['in_pts'][0].to(embedding.device), 
                        times.squeeze(0).unsqueeze(1).to(embedding.device)
                    )
                else:
                    ins_preds = torch.zeros(outputs.shape[1]) # outputs shape: (B, num_points, num_classes)
            # Get probs and labels
            stk_probs = softmax(outputs.view(-1,19)).cpu().detach().numpy()
            centers_output = centers_output.view(batch_size*num_points, -1)
            centers_output = centers_output.cpu().detach().numpy()
            ins_preds = ins_preds.cpu().detach().numpy()
            # s_inds_list = batch['s_ind'] # list of sequences in the batch
            # f_inds_list = batch['f_ind'] # list of the frames in the batch
            # proj_inds_list = batch['proj_inds']
            # loop for every frame in the batch
            i0 = 0
            # Get predictions and labels per volume in batch
            for volume_idx in range(batch_size): # useless loop since we use only 1 sample / batch
                probs = stk_probs[i0:i0 + num_points]
                center_probs  = centers_output[i0:i0 + num_points]
                # ins_probs = ins_preds[i0:i0 + num_points]
                # s_ind = s_inds_list[volume_idx]
                # f_ind = f_inds_list[volume_idx]
                # proj_inds = proj_inds_list[volume_idx]
                # proj_mask = batch['reproj_mask'][volume_idx].numpy()
                # frame_labels = batch['val_labels_list'][volume_idx].numpy() # original unsampled point cloud labels
                frame_labels = sample_gpu['in_lbls'].cpu().numpy()
                centers_gt = centers.cpu().numpy()
                # project predictions on the original frame points
                # proj_probs = probs[proj_inds]
                # proj_center_probs = center_probs[proj_inds]
                # proj_ins_probs = ins_probs[proj_inds]
                # Insert false columns for ignored labels
                for l_ind, label_value in enumerate(val_loader.dataset.label_values):
                    if label_value in val_loader.dataset.ignored_labels:
                        probs = np.insert(probs, l_ind, 0, axis=1)
                # Predicted labels
                preds = val_loader.dataset.label_values[np.argmax(probs, axis=1)]

                # Save predictions in a binary file
                # filename = '{:s}_{:07d}.npy'.format(val_loader.dataset.sequences[s_ind], f_ind)
                # filename_c = '{:s}_{:07d}_c.npy'.format(val_loader.dataset.sequences[s_ind], f_ind)
                # filename_i = '{:s}_{:07d}_i.npy'.format(val_loader.dataset.sequences[s_ind], f_ind)
                # filepath = join(config.saving_path, 'val_preds', filename)
                # filepath_c = join(config.saving_path, 'val_preds', filename_c)
                # filepath_i = join(config.saving_path, 'val_preds', filename_i)

                # if exists(filepath):
                #     frame_preds = np.load(filepath)
                #     center_preds = np.load(filepath_c)
                #     ins_preds = np.load(filepath_i)

                # else:
                #     frame_preds = np.zeros(frame_labels.shape, dtype=np.uint8)
                #     center_preds = np.zeros(frame_labels.shape, dtype=np.float32)
                #     ins_preds = np.zeros(frame_labels.shape, dtype=np.uint8)

                # center_preds[proj_mask] = proj_center_probs[:, 0]
                # frame_preds[proj_mask] = preds.astype(np.uint8)
                # ins_preds[proj_mask] = proj_ins_probs
                # np.save(filepath, frame_preds)
                # np.save(filepath_c, center_preds)
                # np.save(filepath_i, ins_preds)
                
                
                center_gt = centers_gt[:, 0]
                c_iou = (np.sum(np.logical_and(center_probs > 0.5, center_gt > 0.5))) / \
                        (np.sum(center_probs > 0.5) + np.sum(center_gt > 0.5) + 1e-10)
                logger.info("batch_ciou {}".format(c_iou))
                c_ious.append(c_iou)
                s_iou = np.sum(center_probs > 0.5)
                logger.info("batch_siou {}".format(s_iou))
                s_ious.append(s_iou)

                # Update validation confusions
                frame_C = fast_confusion(frame_labels,
                                         preds.astype(np.int32),
                                         val_loader.dataset.label_values)
                val_loader.dataset.val_confs[s_ind][f_ind, :, :] = frame_C
                
                # Stack all prediction for this epoch
                predictions += [preds]
                targets += [frame_labels] #AB: why retrieving from the original point cloud, you can just use the accompained labels for the batch?!
                # inds += [f_inds[b_i, :]]
                val_i += 1
                i0 += num_points

        # log specific losses values
        self.val_logger.add_scalar('Loss/total', t_total_loss.mean(), self.global_step)
        self.val_logger.add_scalar('Loss/cross_entropy', t_cross_entropy_loss.mean(), self.global_step)
        self.val_logger.add_scalar('Loss/center_loss', t_center_loss.mean(), self.global_step)
        self.val_logger.add_scalar('Loss/instance_half_loss', t_instance_half_loss.mean(), self.global_step)
        self.val_logger.add_scalar('Loss/instance_loss', t_instance_loss.mean(), self.global_step)
        self.val_logger.add_scalar('Loss/variance_loss', t_variance_loss.mean(), self.global_step)
        self.val_logger.add_scalar('Loss/variance_l2', t_variance_l2_loss.mean(), self.global_step)
        logger.info("WARNING: Don't directly compare the results of val loss and train loss curves as the batches are not equal !!")
        
        # Plot the metrics over the subsampled point cloud
        mean_acc = np.array(t_acc).mean()
        self.val_logger.add_scalar('acc', mean_acc*100, self.global_step)

        #TODO: plot mIOU 
        
        # Confusions for our subparts of validation set
        Confs = np.zeros((len(predictions), nc_tot, nc_tot), dtype=np.int32)
        for i, (preds, truth) in enumerate(zip(predictions, targets)):
            # Confusions
            Confs[i, :, :] = fast_confusion(truth, preds, val_loader.dataset.label_values).astype(np.int32)

        # Sum all confusions
        C = np.sum(Confs, axis=0).astype(np.float32)

        # Balance with real validation proportions
        # C *= np.expand_dims(val_loader.dataset.class_proportions / (np.sum(C, axis=1) + 1e-6), 1)
        #TODO: check its meaning later !

        # Remove ignored labels from confusions
        for l_ind, label_value in reversed(list(enumerate(val_loader.dataset.label_values))):
            if label_value in val_loader.dataset.ignored_labels:
                C = np.delete(C, l_ind, axis=0)
                C = np.delete(C, l_ind, axis=1)

        # Objects IoU
        IoUs = IoU_from_confusions(C)

        # Sum all validation confusions
        C_tot = [np.sum(seq_C, axis=0) for seq_C in val_loader.dataset.val_confs if len(seq_C) > 0]
        C_tot = np.sum(np.stack(C_tot, axis=0), axis=0)
        # Remove ignored labels from confusions
        for l_ind, label_value in reversed(list(enumerate(val_loader.dataset.label_values))):
            if label_value in val_loader.dataset.ignored_labels:
                C_tot = np.delete(C_tot, l_ind, axis=0)
                C_tot = np.delete(C_tot, l_ind, axis=1)

        # Objects IoU
        val_IoUs = IoU_from_confusions(C_tot)

        # Print instance mean
        mIoU = 100 * np.mean(IoUs)
        logger.info('subpart mIoU = {:.1f} %'.format(mIoU))
        mIoU = 100 * np.mean(val_IoUs)
        logger.info('val mIoU = {:.1f} %'.format(mIoU))
        cIoU = 200 * np.mean(c_ious)
        logger.info('val center mIoU = {:.1f} %'.format(cIoU))
        sIoU = np.mean(s_ious)
        logger.info('val centers sum  = {:.1f} %'.format(sIoU))

            


            


            


    def validation_sem_seg(self, net, val_loader, config):
        d_print("validation call")
        # net.eval()
        with torch.no_grad():
            avg_acc = []
            avg_iou_s = {i: [] for i in range(19)}
            avg_loss = []
            avg_miou = []
            for batch in val_loader:
                # get batch
                sample_gpu = {}
                if 'cuda' in self.device.type:
                    for k, v in batch.items():
                        sample_gpu[k] = v.to(self.device)
                else:
                    sample_gpu = batch
                #TODO: fix the feature handling with the dataloader !
                sample_gpu['in_fts'] = sample_gpu['in_fts'][:,:,:3].transpose(2,1)
                # Forward pass
                outputs, centers_output, var_output, embedding = net(sample_gpu['in_fts'])
                # calculate loss
                loss = net.cross_entropy_loss(outputs, sample_gpu['in_lbls'])
                avg_loss.append(loss.item())
                # self.val_logger.add_scalar('Loss/cross_entropy', net.output_loss.item(), self.global_step)
                acc = net.accuracy(outputs.cpu(), sample_gpu['in_lbls'].cpu())
                avg_acc.append(acc*100)
                ious = net.semantic_seg_metric(outputs.cpu(), sample_gpu['in_lbls'].cpu())
                nan_idx = torch.isnan(ious)
                ious[nan_idx] = 0.
                for i, iou in enumerate(ious):
                    if isnan(iou):
                        iou = 0.
                    # self.train_logger.add_scalar('iou/{}'.format(i), iou, self.global_step)    
                    avg_iou_s[i].append(iou)
                # log mean IoU
                # self.train_logger.add_scalar('iou/mIoU', ious.mean(), self.global_step)    
                avg_miou.append(ious.mean())

            #AB: log into tensorbaord
            avg_loss = np.array(avg_loss, dtype=np.float32)
            d_print("validation loss {}".format(avg_loss.mean()))
            avg_acc = np.array(avg_acc, dtype=np.float32)
            avg_miou = np.array(avg_miou, dtype=np.float32)
            for k,v in avg_iou_s.items():
                avg_iou_s[k] = np.array(v, dtype=np.float32)
            
            self.val_logger.add_scalar("Loss/cross_entropy", avg_loss.mean(), self.global_step)
            self.val_logger.add_scalar('acc', avg_acc.mean(), self.global_step)
            self.val_logger.add_scalar('iou/mIoU', avg_miou.mean(), self.global_step)
            for i, k in avg_iou_s.items():
                self.val_logger.add_scalar("iou/{}".format(i), k.mean(), self.global_step)


        # if config.dataset_task == 'classification':
        #     raise NotImplementedError
        #     self.object_classification_validation(net, val_loader, config)
        # elif config.dataset_task == 'segmentation':
        #     raise NotImplementedError
        #     self.object_segmentation_validation(net, val_loader, config)
        # elif config.dataset_task == 'cloud_segmentation':
        #     raise NotImplementedError
        #     self.cloud_segmentation_validation(net, val_loader, config)
        # elif config.dataset_task == 'slam_segmentation':
        #     self.slam_segmentation_validation(net, val_loader, config)
        # else:
        #     raise ValueError('No validation method implemented for this network type')