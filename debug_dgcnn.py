import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np

from datasets.semantic_kitti_dataset import SemanticKittiDataSet
from models.dgcnn_sem_seg import DGCNN_semseg
from utils.trainer_dgcnn import ModelTrainerDGCNN
from utils.config import Config




def seed_torch(seed=0):
    #random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    seed_torch(seed=0)

    DATASET_PATH = './data'
    train_set = SemanticKittiDataSet(path=DATASET_PATH, set='train')
    val_set = SemanticKittiDataSet(path=DATASET_PATH, set='val')
    train_loader = DataLoader(train_set, batch_size= 4, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size= 4, num_workers=1, pin_memory=True)
    
    #lr=0.1

    net=DGCNN_semseg(train_set.label_values, train_set.ignored_labels, input_feature_dims=4)
    #optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    
    # samples = train_set[0]
    # print(samples['in_pts'].shape)
    # print(samples['in_fts'].shape)
    # print(samples['in_lbls'].shape)
    # print(samples['in_slbls'].shape)
    # print(samples['in_pts'][0,:])
    # print(samples['in_fts'][0,:3])
    # print(samples['in_lbls'][:10])
    # print(samples['in_slbls'][:10])
    
    # model_dir = './results/dgcnn_semseg'
    # model.load_state_dict(torch.load(os.path.join(model_dir, 'model_1.t7')))


    # trainer = ModelTrainerDGCNN(net)
    # trainer.train_overfit(net, optimizer, train_loader, epochs=1000)

    # in_fts = torch.tensor(samples['in_fts']).unsqueeze(0)
    # labels = torch.tensor(samples['in_lbls']).unsqueeze(0)
    # centers = in_fts[:,:,4:8]
    # times = in_fts[:,:,8]

    # outputs, centers_output, var_output, embedding = net(in_fts)
    # loss = net.loss(
    #                 outputs, centers_output, var_output, embedding, 
    #                 torch.tensor(samples['in_lbls']).unsqueeze(0), torch.tensor(samples['in_slbls']).unsqueeze(0), centers, torch.tensor(samples['in_pts']).unsqueeze(0), times)
    # print(loss)

    config = Config()
    config.learning_rate = 0.1
    config.max_epoch = 1000
    #config.saving_path = './results/dgcnn'
    config.checkpoint_gap = 50

    # Training from scratch
    # trainer = ModelTrainerDGCNN(net, config, on_gpu=True)
    # trainer.train_overfit_4D(net, train_loader, config)
    
    # trainer.train(net, train_loader, val_loader, config)
    

    # Pretrained weights of both dgcnn and loss heads
    # chkp_path = './results/dgcnn_semseg_pretrained/model_1.t7'
    # trainer = ModelTrainerDGCNN(net, config, chkp_path=chkp_path, finetune=True, on_gpu=True)
    # trainer.train_overfit_4D(net, train_loader, config)

    # trainer.train(net, train_loader, val_loader, config)


    # from models.dgcnn_utils import rotate_pointcloud
    # for batch in train_loader:
    #     print(batch['in_pts'][0,0,:])
    #     rotated_pc = rotate_pointcloud(config, points=batch['in_pts'], angle_range_z=60)
    #     print(rotated_pc[0,0,:])
        
    #     break
       
    # Training from scratch
    # trainer = ModelTrainerDGCNN(net, config, on_gpu=True)
    # trainer.train_overfit_4D(net, train_loader, config)
    
    # Evaluation
    # from utils.trainer_dgcnn import evaluate_rotated
    # config.eval_rotation = 'vertical'
    # # chkp_dir = './results/dgcnn/Log_2022-01-12_09-56-47/checkpoints' 
    # chkp_dir = './results/dgcnn/Log_2022-01-12_10-43-29/checkpoints' # pretrained weights
    # for angle in [0,60,30,15,5,-5,-15,-30,-60]:
    #     config.angle_z = angle
    #     evaluate_rotated(net, chkp_dir=chkp_dir, config=config)