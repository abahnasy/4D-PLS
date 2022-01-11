"""
semantic segmentation dataloader for semantic kitti dataset
"""
import os
import yaml
import hydra
import numpy as np
from torch.utils.data import Dataset

from utils.debugging import d_print, bcolors

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc




class SemanticKittiSegDataset(Dataset):
    def __init__(self, data_dir='./data', mode ='train', num_points=4096, uniform=True) -> None:
        super().__init__()
        self.path = hydra.utils.to_absolute_path(data_dir)
        self.mode = mode
        self.num_points = num_points
        self.uniform = uniform
        # select sequences
        if self.mode == 'train':
            # self.sequences = ["{:02d}".format(i) for i in range(11) if i != 8]
            self.sequences = ["{:02d}".format(i) for i in range(11) if i == 4]
        elif self.mode == 'val':
            # self.sequences = ["{:02d}".format(i) for i in range(11) if i == 8]
            self.sequences = ["{:02d}".format(i) for i in range(11) if i == 4]
        else:
            # test mode 
            raise NotImplementedError
        
        # List all files in each sequence
        self.frames = []
        for seq in self.sequences:
            velo_path = os.path.join(self.path, 'sequences', seq, 'velodyne')
            frames = np.sort([vf[:-4] for vf in os.listdir(velo_path) if vf.endswith('.bin')])
            self.frames.append(frames)
        seq_inds = np.hstack([np.ones(len(_), dtype=np.int32) * i for i, _ in enumerate(self.frames)])
        frame_inds = np.hstack([np.arange(len(_), dtype=np.int32) for _ in self.frames])
        self.all_inds = np.vstack((seq_inds, frame_inds)).T

        # prepare labels
        config_file = os.path.join(self.path, 'semantic-kitti.yaml')
        with open(config_file, 'r') as stream:
            doc = yaml.safe_load(stream)
            all_labels = doc['labels']
            learning_map_inv = doc['learning_map_inv']
            learning_map = doc['learning_map']
            self.learning_map = np.zeros((np.max([k for k in learning_map.keys()]) + 1), dtype=np.int32)
            for k, v in learning_map.items():
                self.learning_map[k] = v
            self.learning_map_inv = np.zeros((np.max([k for k in learning_map_inv.keys()]) + 1), dtype=np.int32)
            for k, v in learning_map_inv.items():
                self.learning_map_inv[k] = v
            self.content = doc['content']
        # Dict from labels to names
        self.label_to_names = {k: all_labels[v] for k, v in learning_map_inv.items()}
        # Initiate a bunch of variables concerning class labels
        self._init_labels()

        # print weight matrix
        self.class_weights = np.zeros((self.num_classes,), dtype=np.float32)
        for k, v in self.content.items():
            used_key = self.learning_map[k]
            self.class_weights[used_key] += v

        # for i, v in enumerate(self.class_weights):
        #     d_print("{} -> {}".format(i, v), bcolors.FAIL)
        # d_print(">>>>>")
        # d_print(self.class_weights)
        # d_print("<<<<<")
        """
        [3.1501833e-02 4.2607829e-02 1.6609539e-04 3.9838615e-04 2.1649399e-03
        1.8070553e-03 3.3758327e-04 1.2711105e-04 3.7461064e-05 1.9879648e-01
        1.4717169e-02 1.4392298e-01 3.9048553e-03 1.3268620e-01 7.2359227e-02
        2.6681501e-01 6.0350122e-03 7.8142218e-02 2.8554981e-03 6.1559578e-04]
        """


    def _init_labels(self):
        # Initialize all label parameters given the label_to_names dict
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_names = [self.label_to_names[k] for k in self.label_values]
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.name_to_label = {v: k for k, v in self.label_to_names.items()}
        


    def __getitem__(self, index):
        seq_idx, frame_idx = self.all_inds[index]
        seq_path = os.path.join(self.path, 'sequences', self.sequences[seq_idx])
        velo_file = os.path.join(seq_path, 'velodyne', self.frames[seq_idx][frame_idx] + '.bin')
        label_file = os.path.join(seq_path, 'labels', self.frames[seq_idx][frame_idx] + '.label')
        # Read points
        frame_points = np.fromfile(velo_file, dtype=np.float32)
        points = frame_points.reshape((-1, 4))
        points = points[:,:3] #TODO: take xyz only
        # Read labels
        frame_labels = np.fromfile(label_file, dtype=np.int32)
        sem_labels = frame_labels & 0xFFFF  # semantic label in lower half
        sem_labels = self.learning_map[sem_labels]

        # do subsampling to adjust the size of the point clouds size
        curr_num_points = points.shape[0]
        if curr_num_points >= self.num_points:
            selected_point_idxs = np.random.choice(curr_num_points, self.num_points, replace=False)
        else:
            selected_point_idxs = np.random.choice(curr_num_points, self.num_points, replace=True)

        selected_points = points[selected_point_idxs, :]
        selected_points = pc_normalize(selected_points)
        selected_labels = sem_labels[selected_point_idxs]
        return (selected_points, selected_labels)

    def __len__(self):
        return self.all_inds.shape[0]
