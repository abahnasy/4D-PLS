from multiprocessing.sharedctypes import Value
import os
import time
import yaml
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
import hydra
from sklearn.utils.class_weight import compute_class_weight
from sklearn.neighbors import KDTree

from utils.debugging import d_print, write_pc
from datasets.common import grid_subsampling

class SemanticKittiDataSet(Dataset):
    """ Dataset used for PointNet and DGCNN experiments
    
    Note: During Training, the volumes can be constructed parallelly using the gt data
    During Validation, the volume is build from the prediction of the previous frames, 
    so importance sampling should rely on the prediction files located in `val_probs` 
    to sample for the current volume
    """
    def __init__(
        self, 
        path='./data',
        set='train',
        n_frames= 4,
        n_test_frames = 4, 
        sequential_batch = False, 
        balance_classes= False,
        sampling = 'importance', # sampling for points in t-i frames
        decay_sampling = None, 
        grid_subsampling = False,
        num_samples= 99999,
        in_R = None,
        augmentation = 'aligned', # opt:[aligned, z,so3]
        requested_sequences = None,
        saving_path = None,
        verbose = True,
        ) -> None:
        """
        Args:
            config: configuration struct
            set: train, val or test split
            sequential_batch: TODO
            balance_classes:
        Returns: None
        """
        # Assertion section
        # assert balance_classes == False, 'invalid subsampling strategy'
        # assert ((set == 'val') and (saving_path != None)) == True, 'provide the path of the prev predictions for validation'
        if verbose: d_print("Initializing {} dataset".format(set))
        super().__init__()
        # num of training samples
        self.num_samples = num_samples
        # rotation augmentation
        self.augmentation = augmentation
        # Dataset folder
        self.path = hydra.utils.to_absolute_path(path)
        # train or test set
        self.set = set
        # number of concatenated frames
        self.grid_subsampling = grid_subsampling
        self.pointnet_size = 1024*4 #TODO: move to config file
        self.n_frames = n_frames
        self.sampling = sampling
        self.n_test_frames = n_test_frames
        # self.saving_path = ""
        self.decay_sampling = decay_sampling
        self.requested_sequences = requested_sequences
        self.balance_classes = balance_classes
        self.saving_path = saving_path # location where the prev frame predictions are saved
        self.in_R = in_R
        self.verbose = verbose
        # Get a list of sequences
        if self.set == 'train':
            # self.sequences = ['{:02d}'.format(i) for i in range(11) if i != 8]
            self.sequences = ['{:02d}'.format(i) for i in range(11) if i == 4]
        elif self.set == 'val':
            # self.sequences = ['{:02d}'.format(i) for i in range(11) if i == 8]
            self.sequences = ['{:02d}'.format(i) for i in range(11) if i == 4]
        elif self.set == 'test':
            self.sequences = ['{:02d}'.format(i) for i in range(11, 22)]
        else:
            raise ValueError('Unknown set for SemanticKitti data: ', self.set)
        # overwrite the previous default definitions if specific sequence is requested
        if self.requested_sequences:            
            self.sequences = ['{:02d}'.format(i) for i in self.requested_sequences]
        if self.verbose: d_print("INFO: sequences are: {}".format(self.sequences))
        
        # List all files in each sequence
        self.frames = []
        for seq in self.sequences:
            velo_path = os.path.join(self.path, 'sequences', seq, 'velodyne')
            frames = np.sort([vf[:-4] for vf in os.listdir(velo_path) if vf.endswith('.bin')])
            self.frames.append(frames)

        seq_inds = np.hstack([np.ones(len(_), dtype=np.int32) * i for i, _ in enumerate(self.frames)])
        frame_inds = np.hstack([np.arange(len(_), dtype=np.int32) for _ in self.frames])
        self.all_inds = np.vstack((seq_inds, frame_inds)).T
        # choose the training samples
        self.num_samples = min(self.num_samples, self.all_inds.shape[0])
        self.all_inds = self.all_inds[:self.num_samples]
        d_print("Num of training samples is {}".format(self.all_inds.shape[0]))

        self.sequential_batch = sequential_batch
        
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
        # Dict from labels to names
        self.label_to_names = {k: all_labels[v] for k, v in learning_map_inv.items()}
        # Initiate a bunch of variables concerning class labels
        self._init_labels()
        # List of classes ignored during training (can be empty)
        self.ignored_labels = np.sort([0])


        # Load Calibrations
        self._load_calib_poses()
        # Calclate Seq statistics
        self._collect_seq_stats()
        if self.set == 'val':
            # self.val_points = []
            # self.val_labels = []

            # set confusion matrix container for validation use
            self.val_confs = []

            for s_ind, seq_frames in enumerate(self.frames):
                self.val_confs.append(np.zeros((len(seq_frames), self.num_classes, self.num_classes)))


    def _load_calib_poses(self):
        """
        load calib poses and times.
        """
        self.calibrations = []
        self.times = []
        self.poses = []
        for seq in self.sequences:
            seq_folder = os.path.join(self.path, 'sequences', seq)
            # Read Calib
            self.calibrations.append(self._parse_calibration(os.path.join(seq_folder, "calib.txt")))
            # Read times
            self.times.append(np.loadtxt(os.path.join(seq_folder, 'times.txt'), dtype=np.float32))
            # Read poses
            poses_f64 = self._parse_poses(os.path.join(seq_folder, 'poses.txt'), self.calibrations[-1])
            self.poses.append([pose.astype(np.float32) for pose in poses_f64])

    def _collect_seq_stats(self):
        """
        """
        class_frames_bool = np.zeros((0, self.num_classes), dtype=np.bool)
        self.class_proportions = np.zeros((self.num_classes,), dtype=np.int32)
        for s_ind, (seq, seq_frames) in enumerate(zip(self.sequences, self.frames)):
            frame_mode = 'single'
            if self.n_frames > 1:
                frame_mode = 'multi'
            seq_stat_file = os.path.join(self.path, 'sequences', seq, 'stats_{:s}_{}_{}.pkl'.format(frame_mode, self.n_frames, self.num_classes))
            # Check if inputs have already been computed
            if os.path.exists(seq_stat_file):
                # Read pkl
                with open(seq_stat_file, 'rb') as f:
                    seq_class_frames, seq_proportions = pickle.load(f)
            else:
                # Initiate dict
                print('Preparing seq {:s} class frames. (Long but one time only)'.format(seq))
                # Class frames as a boolean mask
                seq_class_frames = np.zeros((len(seq_frames), self.num_classes), dtype=np.bool)
                # Proportion of each class
                seq_proportions = np.zeros((self.num_classes,), dtype=np.int32)
                # Sequence path
                seq_path = os.path.join(self.path, 'sequences', seq)
                # Read all frames
                for f_ind, frame_name in enumerate(seq_frames):
                    # Path of points and labels
                    label_file = os.path.join(seq_path, 'labels', frame_name + '.label')
                    # Read labels
                    frame_labels = np.fromfile(label_file, dtype=np.int32)
                    sem_labels = frame_labels & 0xFFFF  # semantic label in lower half
                    sem_labels = self.learning_map[sem_labels]
                    # Get present labels and there frequency
                    unique, counts = np.unique(sem_labels, return_counts=True)
                    # Add this frame to the frame lists of all class present
                    frame_labels = np.array([self.label_to_idx[l] for l in unique], dtype=np.int32)
                    seq_class_frames[f_ind, frame_labels] = True
                    # Add proportions
                    seq_proportions[frame_labels] += counts
                # Save pickle
                with open(seq_stat_file, 'wb') as f:
                    pickle.dump([seq_class_frames, seq_proportions], f)
            class_frames_bool = np.vstack((class_frames_bool, seq_class_frames))
            self.class_proportions += seq_proportions
        # Transform boolean indexing to int indices.
        self.class_frames = []
        for i, c in enumerate(self.label_values):
            if c in self.ignored_labels:
                self.class_frames.append(torch.zeros((0,), dtype=torch.int64))
            else:
                integer_inds = np.where(class_frames_bool[:, i])[0]
                self.class_frames.append(torch.from_numpy(integer_inds.astype(np.int64)))

    def _parse_calibration(self, filename):
        """
        Returns:
            calibrations: 4x4 array
        """
        calib = {}
        calib_file = open(filename)
        for line in calib_file:
            key, content = line.strip().split(":")
            values = [float(v) for v in content.strip().split()]
            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0
            calib[key] = pose
        calib_file.close()
        return calib

    def _parse_poses(self, filename, calibration):
        """
        Returns:
            poses: 4x4 array
        """
        file = open(filename)
        poses = []
        Tr = calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)
        for line in file:
            values = [float(v) for v in line.strip().split()]
            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0
            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))
        return poses

    def _init_labels(self):
        # Initialize all label parameters given the label_to_names dict
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_names = [self.label_to_names[k] for k in self.label_values]
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.name_to_label = {v: k for k, v in self.label_to_names.items()}

    def __len__(self) -> int:
        return len(self.all_inds)

    def _sample_data(self, data, num_sample, weights = None):
        """ data is in N x ...
            we want to keep num_samplexC of them.
            if N > num_sample, we will randomly keep num_sample of them.
            if N < num_sample, we will randomly duplicate samples.
        """
        N = data.shape[0]
        if (N == num_sample):
            return data, range(N)
        elif (N > num_sample):
            sample = np.random.choice(N, num_sample, replace=False, p = weights) # don't select the sampe point multiple times
            assert np.unique(sample).shape[0] == sample.shape[0]
            return data[sample, ...], sample
        else:
            sample = np.random.choice(N, num_sample-N, p=weights)
            dup_data = data[sample, ...]
            return np.concatenate([data, dup_data], 0), list(range(N))+list(sample)

    def __getitem__(self, index: int):
        """ when getting an index, make it the main frame, concatenate the prev n frames if found
        Returns:
            in_pts:(N,4)
            in_fts: (N,9)
            in_lbls: (N,)
            in_slbls: (N,)
        """
        s_ind, f_ind = self.all_inds[index]
        # Get center of the first frame in world coordinates
        p_origin = np.zeros((1, 4))
        p_origin[0, 3] = 1
        pose0 = self.poses[s_ind][f_ind]
        p0 = p_origin.dot(pose0.T)[:, :3]
        p0 = np.squeeze(p0)
        # save the original 4D volume data, as loaded from the desk, unchanged
        o_pts = None # point locations in the global coordinates, after applying SLAM Pose transformations
        o_labels = None
        o_ins_labels= None
        o_center_labels = None



        num_merged = 0
        f_inc = 0
        f_inc_points = []  # used to get the projection indices of the t-1, t-2, etc... frames. don't add to return dict.

        # Initiate merged points
        merged_points = np.zeros((0, 3), dtype=np.float32)
        merged_labels = np.zeros((0,), dtype=np.int32)
        merged_ins_labels = np.zeros((0,), dtype=np.int32)
        merged_coords = np.zeros((0, 9), dtype=np.float32)

        # volume_prev_frames = [] #AB: used for visualization and debugging only
        while num_merged < self.n_frames and f_ind - f_inc >= 0:
            # Current frame pose
            pose = self.poses[s_ind][f_ind - f_inc]
            # TODO: take frame only if it has changed a certain distance
            # Path of points and labels
            seq_path = os.path.join(self.path, 'sequences', self.sequences[s_ind])
            velo_file = os.path.join(seq_path, 'velodyne', self.frames[s_ind][f_ind - f_inc] + '.bin')
            if self.set == 'test':
                label_file = None
            else:
                label_file = os.path.join(seq_path, 'labels', self.frames[s_ind][f_ind - f_inc] + '.label')
                center_file = os.path.join(seq_path, 'labels', self.frames[s_ind][f_ind - f_inc] + '.center.npy')
            # Read points
            frame_points = np.fromfile(velo_file, dtype=np.float32)
            points = frame_points.reshape((-1, 4))
            # Read labels
            frame_labels = np.fromfile(label_file, dtype=np.int32)
            sem_labels = frame_labels & 0xFFFF  # semantic label in lower half
            ins_labels = frame_labels >> 16
            ins_labels = ins_labels.astype(np.int32)
            sem_labels = self.learning_map[sem_labels]
            center_labels = np.load(center_file)
            if np.isnan(center_labels).any():
                center_labels = np.zeros_like(center_labels)
            #center_labels = (center_labels > 0.3) * 1
            center_labels = center_labels.astype(np.float32)
            # Move the loaded point cloud into global position
            # Apply pose (without np.dot to avoid multi-threading)
            hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
            new_points = np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1)
            
            # ===================================================================#
            # save the original data for use later in validation
            # ===================================================================#
            
            # In case of validation, keep the original points in memory
            if self.set in ['val', 'test'] and f_inc == 0:
                o_pts = new_points[:, :3].astype(np.float32)
                o_labels = sem_labels.astype(np.int32)
                o_center_labels = center_labels
                o_ins_labels = ins_labels.astype(np.int32)

            if self.set in ['val', 'test'] and self.n_test_frames > 1 and f_inc > 0:
                f_inc_points.append(new_points[:, :3].astype(np.float32))
            
            # ===================================================================#
            # Importance subsampling #
            # ===================================================================#

            # Do subsampling if the point cloud is not the main one
            mask = np.ones_like(new_points[:,0]) # (N,1)
            # this case is only during training or when evaluating on a voume of size = 1
            if self.set in ['train', 'val'] and f_inc > 0 and self.n_test_frames == 1:
                if self.sampling == 'objectness':
                    mask = ((sem_labels > 0) & (sem_labels < 9))
                elif self.sampling == 'importance':
                    n_points_to_sample = np.sum((sem_labels > 0) & (sem_labels < 9))
                    probs = (center_labels[:,0] + 0.1)
                    idxs = np.random.choice(np.arange(center_labels.shape[0]), n_points_to_sample, p=probs/np.sum(probs))
                    mask = np.zeros_like(new_points[:,0]) # (N,1)
                    mask[idxs] = 1
                else:
                    pass
            # this case during validation, we need to sample the things classes from 
            # the previous point clouds based on the predictions of the previous cycles, 
            # not the ground truth data
            if self.set in ['val'] and f_inc > 0 and self.n_test_frames > 1:
                # folder path, where the predictions of the prev frames are saved
                test_path = os.path.join('test', self.saving_path.split('/')[-1] + str(self.n_test_frames))
                test_path = hydra.utils.to_absolute_path(test_path)
                if self.set == 'val':
                    test_path = os.path.join(test_path, 'val_probs')
                else:
                    raise ValueError("unknown set, should be Val !")

                if self.sampling == 'objectness':
                    filename = '{:s}_{:07d}.npy'.format(self.sequences[s_ind], f_ind-f_inc)
                    file_path = os.path.join(test_path, filename)
                    label_pred = None
                    counter = 0
                    while label_pred is None:
                        try:
                            label_pred = np.load(file_path)
                        except:
                            print ('label cannot be read {}'.format(file_path))
                            counter +=1
                            if counter > 5:
                                break
                            continue
                    #eliminate points which are not belong to any instance class for future frame
                    if label_pred is not None:
                        mask = (( label_pred > 0) & (label_pred < 9) & mask)
                elif self.sampling == 'importance':
                    filename = '{:s}_{:07d}_c.npy'.format(self.sequences[s_ind], f_ind - f_inc)
                    file_path = os.path.join(test_path, filename)
                    center_pred = None
                    counter = 0
                    while center_pred is None:
                        try:
                            center_pred = np.load(file_path)
                        except:
                            time.sleep(2)
                            counter +=1
                            if counter > 5:
                                break
                            continue
                    if center_pred is not None:
                        n_points_to_sample = int(np.sum(mask)/10)
                        decay_ratios = np.array([np.exp(i/self.n_test_frames) for i in range(1,self.n_test_frames)])
                        decay_ratios = decay_ratios *  ((self.n_test_frames-1)/np.sum(decay_ratios))#normalize sums
                        if self.decay_sampling == 'forward':
                            n_points_to_sample = int(n_points_to_sample*decay_ratios[f_inc-1])
                        if self.decay_sampling == 'backward':
                            n_points_to_sample = int(n_points_to_sample * decay_ratios[-f_inc])
                        probs = (center_pred[:, 0] + 0.1)
                        idxs = np.random.choice(np.arange(center_pred.shape[0]), n_points_to_sample,
                                                p= (probs / np.sum(probs)))
                        new_mask = np.zeros_like(mask)
                        new_mask[idxs] = 1
                        mask = (new_mask & mask)
                else:
                    pass
                # volume_prev_frames.append(str(f_ind - f_inc))
            
            # ===================================================================#
            
            assert mask.shape[0] == center_labels.shape[0]
            assert mask.shape[0] == new_points.shape[0]
            mask_inds = np.where(mask)[0].astype(np.int32)

            # Shuffle points
            rand_order = np.random.permutation(mask_inds)
            new_points = new_points[rand_order, :3]
            sem_labels = sem_labels[rand_order]
            ins_labels = ins_labels[rand_order]
            center_labels = center_labels[rand_order]
            # Place points in original frame reference to get coordinates
            if f_inc == 0:
                new_coords = points[rand_order, :]
                assert new_coords.shape[1] == 4, "debugging check" #AB
            else:
                # We have to project in the first frame coordinates
                new_coords = new_points - pose0[:3, 3]
                # new_coords = new_coords.dot(pose0[:3, :3])
                new_coords = np.sum(np.expand_dims(new_coords, 2) * pose0[:3, :3], axis=1)
                new_coords = np.hstack((new_coords, points[rand_order, 3:]))
                assert new_coords.shape[1] == 4 ," debugging check" #AB

            

            #center_labels = np.reshape(center_labels,(-1,1))
            d_coords = new_coords.shape[1]
            d_centers = center_labels.shape[1]
            times = np.ones((center_labels.shape[0],1)) * f_inc
            times = times.astype(np.float32)
            new_coords = np.hstack((new_coords, center_labels))
            new_coords = np.hstack((new_coords, times))
            #labels = np.hstack((sem_labels, ins_labels))
            # Increment merge count

            # append to the current list of points
            merged_points = np.vstack((merged_points, new_points))
            merged_labels = np.hstack((merged_labels, sem_labels))
            merged_ins_labels = np.hstack((merged_ins_labels, ins_labels))
            merged_coords = np.vstack((merged_coords, new_coords))

            num_merged += 1
            f_inc += 1

        # grid subsampling for the  merged frames
        # refer to : KPCONV author phd thesis
        first_subsampling_dl = 0.06 * 2        
        if self.grid_subsampling:
            in_pts, in_fts, in_lbls, in_slbls = grid_subsampling(
                merged_points,
                features=merged_coords,
                labels=merged_labels,
                ins_labels=merged_ins_labels,
                sampleDl=first_subsampling_dl
                )
        else:
            in_pts = merged_points
            in_fts = merged_coords
            in_lbls = merged_labels
            in_slbls = merged_ins_labels

        # Number collected
        n = in_pts.shape[0]
        # Safe check
        if n < 2:
            raise ValueError("not enough points after subsampling !!")

    
        
        
        #AB: print class weights
        # print(np.unique(in_lbls))
        # print(np.bincount(in_lbls))
        # # n_samples / (n_classes * np.bincount(y))
        # n_samples = in_lbls.shape[0]
        # n_classes = 20
        # weights = (n_samples / (n_classes * np.bincount(in_lbls) + 1e-6))
        # weights = [w if w < 1/1e-6 else 1 for w in weights]
        # print(weights)
        
        # from prettytable import PrettyTable
        # x = PrettyTable()
        # x.add_column("classes", [i for i in range(20)])
        # x.add_column("weights", weights)
        # print(x)
        # exit()

        # from prettytable import PrettyTable
        # table = PrettyTable()
        # table.add_column("cls", [i for i in range(20)])
        # (unique, counts) = np.unique(in_lbls, return_counts=True)
        # table_counts = [counts[np.where(unique == i)][0] if i in unique else 0 for i in range(20)]
        # weights = [in_lbls.shape[0] / count if count != 0 else 0 for count in table_counts]
        # print(unique)
        # print(counts)
        # print(table_counts)
        # table.add_column("lbls", table_counts)
        # table.add_column("weights", weights)
        # print(table)
        

        if self.balance_classes:
            (unique, counts) = np.unique(in_lbls, return_counts=True)
            _counts = [counts[np.where(unique == i)][0] if i in unique else 0 for i in range(20)]
            weights = [in_lbls.shape[0] / count if count != 0 else 0 for count in _counts]
            p_weights = np.zeros(in_lbls.shape[0])
            for i in range(20):
                idx = np.where(in_lbls == i)
                p_weights[idx] = weights[i]
            p_weights = p_weights / p_weights.sum()
        else:
            p_weights = None

        # sampling to maintain fixed batch size
        in_pts , idxs = self._sample_data(in_pts, self.pointnet_size, weights = p_weights)
        # in_pts_ = in_pts[idxs]
        in_fts = in_fts[idxs]
        in_lbls = in_lbls[idxs]
        in_slbls = in_slbls[idxs]


        # ===================================================================#
        # For validation: get the reprojection indices
        # AB: according to my understanding, these are the indices of the 
        # points in the original point cloud that are equivalent to the final subsampled one, 
        # so the predictions done on the subsampled one could be reflected on the original point cloud
        # ===================================================================#
        # Before augmenting, compute reprojection inds (only for validation and test)
        if self.set in ['val', 'test']:
            # get val_points that are in range
            radiuses = np.sum(np.square(o_pts - p0), axis=1)
            reproj_mask = radiuses < (0.99 * self.in_R) ** 2

            # Project predictions on the frame points
            search_tree = KDTree(in_pts, leaf_size=50)
            proj_inds = search_tree.query(o_pts[reproj_mask, :], return_distance=False)
            # proj_inds = search_tree.query(o_pts, return_distance=False)
            proj_inds = np.squeeze(proj_inds).astype(np.int32)

        else:
            proj_inds = np.zeros((0,))
            reproj_mask = np.zeros((0,))

        if self.set in ['val', 'test'] and self.n_test_frames > 1:
            f_inc_proj_inds = []
            f_inc_reproj_mask = []
            for i in range(len(f_inc_points)):
                # get val_points that are in range
                radiuses = np.sum(np.square(f_inc_points[i] - p0), axis=1)
                f_inc_reproj_mask.append(radiuses < (0.99 * self.in_R) ** 2)

                # Project predictions on the frame points
                search_tree = KDTree(in_pts, leaf_size=100)
                f_inc_proj_inds.append(search_tree.query(f_inc_points[i][f_inc_reproj_mask[-1], :], return_distance=False))
                # f_inc_proj_inds.append(search_tree.query(f_inc_points[i], return_distance=False))
                f_inc_proj_inds[-1] = np.squeeze(f_inc_proj_inds[-1]).astype(np.int32)

        # ===================================================================#
        # Rotation augmentations
        # ===================================================================#
        if self.augmentation == 'z':
            rotation_options = np.array([i for i in range(0,60,5)], dtype=np.float32)
            z_angle = np.random.choice(rotation_options)
            in_fts[:,:3] = rotate(in_fts[:,:3], rot_z = z_angle)
            d_print(z_angle)
            # write_pc(in_fts[:,:3], 'after_aug_{}'.format(z_angle))
        elif self.augmentation == 'so3':
            rotation_options = np.array([i for i in range(0,30,5)], dtype=np.float32)
            x_angle = np.random.choice(rotation_options)
            y_angle = np.random.choice(rotation_options)
            z_angle = np.random.choice(rotation_options)
            in_fts[:,:3] = rotate(in_fts[:,:3], rot_x = x_angle, rot_y = y_angle, rot_z = z_angle)
        else:
            pass # no rotations

        
    
        sample = {
            # temp variable for visualizations and debugging
            # 'volume_name': "seq_{:02d}_frames_{:06d}_{}".format((s_ind), (f_ind), "_".join(volume_prev_frames)),
            'in_pts': in_pts, # points in the global position according to SLAM transformations
            'in_fts': in_fts, # points in frame t frame of reference
            'in_lbls': in_lbls, # semantic labels
            'in_slbls': in_slbls, # instance labels
        }
        # Validation related variables
        if self.set == 'val':
            sample.update ({
                's_ind': s_ind,
                'f_ind': f_ind,
                'val_labels_list': o_labels ,
                'val_ins_labels_list': o_ins_labels ,
                'val_center_label_list': o_center_labels ,
                'proj_inds': proj_inds,
                'reproj_mask': reproj_mask,
                'f_inc_proj_inds': f_inc_proj_inds ,
                'f_inc_reproj_mask': f_inc_reproj_mask,
            })
            
        
        return sample

def rotate(in_pts, rot_x=0, rot_y=0, rot_z=0):
    """angles: array of angles for x,y,z axes
    """
    c = np.cos(rot_x *np.pi / 180.)
    s = np.sin(rot_x*np.pi / 180.)
    x_rot_mat = np.array([
        [1, 0,  0],
        [0, c, -s],
        [0, s,  c]
        ], dtype=np.float32)
    # d_print(x_rot_mat)
    c = np.cos(rot_y*np.pi / 180.)
    s = np.sin(rot_y*np.pi / 180.)
    y_rot_mat = np.array(
        [[c,  0,  s],
        [0,  1,  0],
        [-s, 0,  c]
        ], dtype=np.float32)
    # d_print(y_rot_mat)
    c = np.cos(rot_z*np.pi / 180.)
    s = np.sin(rot_z*np.pi / 180.)
    z_rot_mat = np.array([
        [c, -s,  0],
        [s,  c,  0],
        [0,  0,  1]
        ], dtype=np.float32)
    # d_print(z_rot_mat)
    rot_mat = np.dot(z_rot_mat, y_rot_mat)
    rot_mat = np.dot(rot_mat, x_rot_mat)

    ctr = in_pts.mean(axis=0) 
    in_pts = np.dot(in_pts-ctr, rot_mat) + ctr
    return in_pts
    # ctr = in_fts[:,:3].mean(axis = 0)
    # in_fts[:,:3] = np.dot(in_fts[:,:3] - ctr, rot_mat) + ctr

