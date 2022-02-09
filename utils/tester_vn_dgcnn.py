import os
import hydra

import torch
import numpy as np

from utils.metrics import IoU_from_confusions, fast_confusion

class ModelTesterVNDGCNN ():
    """
    """
    def __init__(self) -> None:
        """
        """
        self.instances = {}
        self.next_ins_id = 1

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        

    def panoptic_4d_test(self, net, val_loader, cfg):
        """
        Args:
            cfg: configuration parameters from hydra config file
        """
        test_smooth = 0.5
        softmax = torch.nn.Softmax(1)
        # Number of classes including ignored labels
        nc_tot = val_loader.dataset.num_classes
        nc_model = net.C
        # Test saving path
        test_path = None
        report_path = None

        assoc_saving = cfg.sampling
        cfg.assoc_saving = cfg.sampling+'_'+ cfg.decay_sampling
        if hasattr(cfg, 'stride'):
            cfg.assoc_saving = cfg.sampling + '_' + cfg.decay_sampling+ '_str' + str(cfg.stride) +'_'
        
        if cfg.saving:
            test_path = os.path.join('test', cfg.saving_path.split('/')[-1]+ '_'+cfg.assoc_saving+str(cfg.n_test_frames))
            test_path = hydra.utils.to_absolute_path(test_path)
            if not os.path.exists(test_path):
                os.makedirs(test_path)
            report_path = os.path.join(test_path, 'reports')
            if not os.path.exists(report_path):
                os.makedirs(report_path)

        
        for folder in ['val_predictions', 'val_probs']:
            if not os.path.exists(os.path.join(test_path, folder)):
                os.makedirs(os.path.join(test_path, folder))
        
        # Init validation container
        all_f_preds = []
        all_f_labels = []
        for i, seq_frames in enumerate(val_loader.dataset.frames):
            all_f_preds.append([np.zeros((0,), dtype=np.int32) for _ in seq_frames])
            all_f_labels.append([np.zeros((0,), dtype=np.int32) for _ in seq_frames])

        # Network predictions
        predictions = []
        targets = []
        test_epoch = 0

        

        for i, batch in enumerate(val_loader):
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
                points = sample_gpu['in_pts']
                centers = sample_gpu['in_fts'][:,:,4:8]
                times = sample_gpu['in_fts'][:,:,8]
                # save for visualization
                # viz_pc = sample_gpu['in_fts'][:,:,:3].squeeze(0).cpu().numpy()
                # prepare inputs for PointNet Architecture
                sample_gpu['in_fts'] = sample_gpu['in_fts'][:,:,:3].transpose(2,1) #TODO: fix the feature handling with the dataloader !
                # Forward pass
                outputs, centers_output, var_output, embedding = net(sample_gpu['in_fts'])
                probs = softmax(outputs.view(-1, 19)).cpu().detach().numpy()
                for l_ind, label_value in enumerate(val_loader.dataset.label_values):
                        if label_value in val_loader.dataset.ignored_labels:
                            probs = np.insert(probs, l_ind, 0, axis=1)
                preds = val_loader.dataset.label_values[np.argmax(probs, axis=1)]
                exit()
                preds = torch.from_numpy(preds)
                preds.to(outputs.device)
                sequence = val_loader.dataset.sequences[s_ind]
                pose = val_loader.dataset.poses[s_ind][f_ind]
                if sequence not in self.instances:
                        self.instances[sequence] = {}
                ins_preds, new_instances, ins_id = net.ins_pred_in_time (
                    preds, 
                    centers_output, 
                    var_output, 
                    embedding, 
                    self.instances[sequence],
                    self.next_ins_id,
                    points,
                    times, # (N,1)
                    pose
                )

                self.next_ins_id = ins_id#update next available ins id
                for ins_id, instance in new_instances.items(): #add new instances to history
                    self.instances[sequence][ins_id] = instance

                dont_track_ids = []
                for ins_id in self.instances[sequence].keys():
                    if self.instances[sequence][ins_id]['life'] == 0:
                        dont_track_ids.append(ins_id)
                    self.instances[sequence][ins_id]['life'] -= 1

                for ins_id in dont_track_ids:
                    del self.instances[sequence][ins_id]

                stk_probs = softmax(outputs).cpu().detach().numpy()
                ins_preds = ins_preds.cpu().detach().numpy()
                centers_output = centers_output.cpu().detach().numpy()

                probs = stk_probs[0]
                c_probs = centers_output[0]
                ins_probs = ins_preds[0]

                seq_name = val_loader.dataset.sequences[s_ind]
                folder = 'val_probs'
                pred_folder = 'val_predictions'
                
                filename = '{:s}_{:07d}.npy'.format(seq_name, f_ind)
                filepath = os.path.join(test_path, folder, filename)
                filename_i = '{:s}_{:07d}_i.npy'.format(seq_name, f_ind)
                filename_c = '{:s}_{:07d}_c.npy'.format(seq_name, f_ind)
                filepath_i = os.path.join(test_path, folder, filename_i)
                filepath_c = os.path.join(test_path, folder, filename_c)

                # make size equal to 4k instead of full point cloud size !
                frame_probs_uint8 = np.zeros((preds.shape[0], nc_model), dtype=np.uint8)
                frame_c_probs = np.zeros((preds.shape[0], 1))
                ins_preds = np.zeros((preds.shape[0]))

                frame_probs = frame_probs_uint8.astype(np.float32) / 255 # AB: how come?
                frame_probs = test_smooth * frame_probs + (1 - test_smooth)
                frame_probs_uint8 = (frame_probs * 255).astype(np.uint8)
                ins_preds = ins_probs
                frame_c_probs = c_probs

                np.save(filepath_i, ins_preds)
                np.save(filepath_c, frame_c_probs)

                ins_features = {}
                for ins_id in np.unique(ins_preds):
                    if int(ins_id) in self.instances[sequence]:
                        ins_features[int(ins_id)] = self.instances[sequence][int(ins_id)]['mean']
                filename_f = '{:s}_{:07d}_f.npy'.format(seq_name, f_ind)
                filepath_f = os.path.join(test_path, folder, filename_f)

                frame_probs_uint8_bis = frame_probs_uint8.copy()
                for l_ind, label_value in enumerate(val_loader.dataset.label_values):
                    if label_value in val_loader.dataset.ignored_labels:
                        frame_probs_uint8_bis = np.insert(frame_probs_uint8_bis, l_ind, 0, axis=1)

                frame_preds = val_loader.dataset.label_values[np.argmax(frame_probs_uint8_bis,
                                                                                 axis=1)].astype(np.int32)
                np.save(filepath, frame_preds)

        # results on the whole dataset
        Confs = np.zeros((len(predictions), nc_tot, nc_tot), dtype=np.int32)
        for i, (preds, truth) in enumerate(zip(predictions, targets)):
            # Confusions
            Confs[i, :, :] = fast_confusion(truth, preds, val_loader.dataset.label_values).astype(np.int32)
            # Show vote results
        print('\nCompute confusion')
        val_preds = []
        val_labels = []
        
        for i, seq_frames in enumerate(val_loader.dataset.frames):
            val_preds += [np.hstack(all_f_preds[i])]
            val_labels += [np.hstack(all_f_labels[i])]
        val_preds = np.hstack(val_preds)
        val_labels = np.hstack(val_labels)
        
        C_tot = fast_confusion(val_labels, val_preds, val_loader.dataset.label_values)

        s1 = '\n'
        for cc in C_tot:
            for c in cc:
                s1 += '{:7.0f} '.format(c)
            s1 += '\n'
        print(s1)

        # Remove ignored labels from confusions
        for l_ind, label_value in reversed(list(enumerate(val_loader.dataset.label_values))):
            if label_value in val_loader.dataset.ignored_labels:
                C_tot = np.delete(C_tot, l_ind, axis=0)
                C_tot = np.delete(C_tot, l_ind, axis=1)

        # Objects IoU
        val_IoUs = IoU_from_confusions(C_tot)

        # Compute IoUs
        mIoU = np.mean(val_IoUs)
        s2 = '{:5.2f} | '.format(100 * mIoU)
        for IoU in val_IoUs:
            s2 += '{:5.2f} '.format(100 * IoU)
        print(s2 + '\n')

        # Save a report
        strg = 'Report of the confusion and metrics\n'
        strg += '***********************************\n\n\n'
        strg += 'Confusion matrix:\n\n'
        strg += s1
        strg += '\nIoU values:\n\n'
        strg += s2
        strg += '\n\n'





    

