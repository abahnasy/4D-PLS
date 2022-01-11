""" The aim of this script is to automate the run of the functions in test_models.py, stitch_tracklets.py and evalutate_4dpanoptic.py
and run automatically with different settings to do evaluations

different settins is written in conf_test configurations and do multi run using hydra library

HOW TO USE THIS SCRIPT: change the parameter `config.validation_size` to avoid out of bound error
"""
import os
import time
import logging
from typing import OrderedDict

import numpy as np
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf

from datasets.SemanticKitti import *
from models.architectures import KPFCNN
from utils.tester import ModelTester
from utils.config import Config
from utils.eval_np import Panoptic4DEval
from utils.debugging import d_print

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

#TODO:
"""
get val_seq from config file, assign it to the dataset class, then pass it to FLAGS_1 and then FLAGS
"""
# A logger for this file
log = logging.getLogger(__name__)

@hydra.main(config_path="conf_test", config_name="config")
def my_app(cfg : DictConfig) -> None:
    log.info(OmegaConf.to_yaml(cfg))
    
    # ===================================
    # == replicate test_models.py ==
    # ===================================
    
    d_print(">>>>>>>>>> Replicating test_models.py <<<<<<<<<<")
    chosen_log = 'results/Log_2020-10-06_16-51-05'
    chosen_log = hydra.utils.to_absolute_path(chosen_log)
    chkp_idx = None
    on_val = True
    chkp_path = os.path.join(chosen_log, 'checkpoints')
    chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']
    if chkp_idx is None:
        chosen_chkp = 'current_chkp.tar'
    else:
        chosen_chkp = np.sort(chkps)[chkp_idx]
    chosen_chkp = os.path.join(chosen_log, 'checkpoints', chosen_chkp)

    # Initialize configuration class
    config = Config()
    config.load(chosen_log)
    # Change parameters for the test here. For example, you can stop augmenting the input data.
    config.global_fet = False
    config.validation_size = 101 #AB: manual chnage to avoid out of bound error
    # config.input_threads = 16 #AB: useless, cannot run multithreading during validation and testing
    config.n_frames = 4
    config.n_test_frames = 4 #it should be smaller than config.n_frames
    if config.n_frames < config.n_test_frames:
        config.n_frames = config.n_test_frames
    config.big_gpu = True
    config.dataset_task = '4d_panoptic'
    #config.sampling = 'density'
    config.sampling = 'importance'
    config.decay_sampling = 'None'
    config.stride = 1
    config.first_subsampling_dl = 0.061
    # prepare data
    log.info("\n")
    log.info('Data Preparation')
    log.info('****************')
    if on_val:
        set = 'validation'
    else:
        set = 'test'
    test_dataset = SemanticKittiDataset(
        config, 
        set=set, 
        balance_classes=False, 
        seqential_batch=True,
        rot_x = cfg.rot_x,
        rot_y = cfg.rot_y,
        rot_z = cfg.rot_z,
        )
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
    test_sampler.calibration(test_loader, verbose=True)
    log.info('\nModel Preparation')
    log.info('*****************')
    # Define network model
    t1 = time.time()
    net = KPFCNN(config, test_dataset.label_values, test_dataset.ignored_labels)
    # Define a visualizer class
    tester = ModelTester(net, chkp_path=chosen_chkp)
    log.info('Done in {:.1f}s\n'.format(time.time() - t1))
    log.info('\nStart test')
    log.info('**********\n')
    config.dataset_task = '4d_panoptic'
    # thet test path will be used in the following sections
    test_path = tester.panoptic_4d_test(net, test_loader, config)
    d_print("returned path from test_models is {}".format(test_path))
    d_print("passed test 4dpanoptic")
    
    
    # ===================================
    # == replicate stitch tracking ==
    # ===================================
    d_print(">>>>>>>>>> Replicating stitch_tracklets.py <<<<<<<<<<")
    # python stitch_tracklets.py --predictions test/model_dir --n_test_frames 4
    import argparse
    parser = argparse.ArgumentParser("./stitch_tracklets.py")
    parser.add_argument(
        '--n_test_frames',
        '-n',
        type=int,
        default=4
    )

    parser.add_argument(
        '--3d', '-3d',
        dest='iou_3d',
        type=float,
        default=0
    )

    parser.add_argument(
        '--2d', '-2d',
        dest='iou_2d',
        type=float,
        default=0
    )

    parser.add_argument(
        '--center', '-c',
        dest='center',
        type=float,
        default=0
    )

    parser.add_argument(
        '--feature', '-f',
        dest='feature',
        type=float,
        default=0
    )

    parser.add_argument(
        '--sequences', '-s',
        dest='sequences',
        type=str,
        default='8'
    )

    parser.add_argument(
        '--predictions', '-p',
        dest='predictions',
        type=str,
        required=False,
        default = None
    )
    FLAGS_1, unparsed = parser.parse_known_args()
    FLAGS_1.predictions = test_path
    FLAGS_1.sequences = [506]

    
    

    from stitch_tracklets import main
    main(FLAGS_1, use_hydra=True)
    d_print("passed stitch tracklets")
    
    # ===================================
    # == replicate evaluate_4dpanoptic ==
    # ===================================
    d_print(">>>>>>>>>> Replicating evaluate_4dpanoptic.py <<<<<<<<<<")
    # FLAGS = {
    #     "dataset": "{}".format(hydra.utils.to_absolute_path("./data")),
    #     "predictions": "{}/stitch4".format(test_path),
    #     "split": "valid", # choices: ["train", "valid", "test"],
    #     "data_cfg": "{}".format(hydra.utils.to_absolute_path('./data/semantic-kitti.yaml')),
    #     "limit": None,
    #     "min_inst_points": 50,
    #     "output": None,
    # }
    FLAGS = OrderedDict()
    FLAGS.dataset = "{}".format(hydra.utils.to_absolute_path("./data"))
    FLAGS.predictions = "{}/stitch4".format(test_path)
    FLAGS.split = "valid"
    FLAGS.data_cfg = "{}".format(hydra.utils.to_absolute_path('./data/semantic-kitti.yaml'))
    FLAGS.limit = None
    FLAGS.min_inst_points = 50
    FLAGS.output= None
    FLAGS.sequences = [506] #TODO: remove it after getting it from the config file
    
    start_time = time.time()
    # possible splits
    splits = ["train", "valid", "test"]
    # python evaluate_4dpanoptic.py \ 
    # --dataset=SemanticKITTI_dir \
    # --predictions=output_of_stitch_tracket_dir \
    # --data_cfg=semantic-kitti.yaml

    # print summary of what we will do
    print("*" * 80)
    print("INTERFACE:")
    print("Data: ", FLAGS.dataset)
    print("Predictions: ", FLAGS.predictions)
    print("Split: ", FLAGS.split)
    print("Config: ", FLAGS.data_cfg)
    print("Limit: ", FLAGS.limit)
    print("Min instance points: ", FLAGS.min_inst_points)
    print("Output directory", FLAGS.output)
    print("*" * 80)
    # assert split
    assert (FLAGS.split in splits)
    # open data config file
    DATA = yaml.safe_load(open(FLAGS.data_cfg, 'r'))
    # get number of interest classes, and the label mappings
    # class
    class_remap = DATA["learning_map"]
    class_inv_remap = DATA["learning_map_inv"]
    class_ignore = DATA["learning_ignore"]
    nr_classes = len(class_inv_remap)
    class_strings = DATA["labels"]
    # make lookup table for mapping
    # class
    maxkey = max(class_remap.keys())
    # +100 hack making lut bigger just in case there are unknown labels
    class_lut = np.zeros((maxkey + 100), dtype=np.int32)
    class_lut[list(class_remap.keys())] = list(class_remap.values())
    # class
    ignore_class = [cl for cl, ignored in class_ignore.items() if ignored]
    log.info("Ignoring classes: {}".format(ignore_class))
    # get test set
    # test_sequences = DATA["split"][FLAGS.split]
    test_sequences = FLAGS.sequences
    # create evaluator
    class_evaluator = Panoptic4DEval(nr_classes, None, ignore_class, offset = 2 ** 32, min_points=FLAGS.min_inst_points)
    # get label paths
    label_names = []
    for sequence in test_sequences:
        sequence = '{0:02d}'.format(int(sequence))
        label_paths = os.path.join(FLAGS.dataset, "sequences", sequence, "labels")
        # populate the label names
        seq_label_names = sorted([os.path.join(label_paths, fn) for fn in os.listdir(label_paths) if fn.endswith(".label")])
        label_names.extend(seq_label_names)
    # print(label_names)

    # get predictions paths
    pred_names = []
    for sequence in test_sequences:
        sequence = '{0:02d}'.format(int(sequence))
        pred_paths = os.path.join(FLAGS.predictions, "sequences", sequence, "predictions")
        # populate the label names
        seq_pred_names = sorted([os.path.join(pred_paths, fn) for fn in os.listdir(pred_paths) if fn.endswith(".label")])
        pred_names.extend(seq_pred_names)
    # print(pred_names)

    # check that I have the same number of files
    assert (len(label_names) == len(pred_names))
    print("Evaluating sequences: ", end="", flush=True)
    # open each file, get the tensor, and make the iou comparison

    complete = len(label_names)

    count = 0
    percent = 10
    for label_file, pred_file in zip(label_names, pred_names):
        count = count + 1
        if 100 * count / complete > percent:
            print("{}% ".format(percent), end="", flush=True)
            percent = percent + 10
            # log.info("evaluating label ", label_file, "with", pred_file)
            # open label

        label = np.fromfile(label_file, dtype=np.uint32)

        u_label_sem_class = class_lut[label & 0xFFFF]  # remap to xentropy format
        u_label_inst = label >> 16
        if FLAGS.limit is not None:
            u_label_sem_class = u_label_sem_class[:FLAGS.limit]
            u_label_inst = u_label_inst[:FLAGS.limit]

        label = np.fromfile(pred_file, dtype=np.uint32)

        u_pred_sem_class = class_lut[label & 0xFFFF]  # remap to xentropy format
        u_pred_inst = label >> 16
        if FLAGS.limit is not None:
            u_pred_sem_class = u_pred_sem_class[:FLAGS.limit]
            u_pred_inst = u_pred_inst[:FLAGS.limit]

        class_evaluator.addBatch(label_file.split('/')[-3], u_pred_sem_class, u_pred_inst, u_label_sem_class, u_label_inst)

    log.info("100%")

    complete_time = time.time() - start_time
    LSTQ, LAQ_ovr, LAQ, AQ_p, AQ_r,  iou, iou_mean, iou_p, iou_r = class_evaluator.getPQ4D()
    things_iou = iou[1:9].mean()
    stuff_iou = iou[9:].mean()
    log.info ("=== Results ===")
    log.info ("LSTQ:{}".format(LSTQ))
    log.info("S_assoc (LAQ): {}".format(LAQ_ovr))
    float_formatter = "{:.2f}".format
    np.set_printoptions(formatter={'float_kind': float_formatter})
    log.info ("Assoc: {}".format(LAQ))
    log.info ("iou: {}".format(iou))
    log.info("things_iou: {}".format(things_iou))
    log.info("stuff_iou: {}".format(stuff_iou))

    log.info ("S_cls (LSQ): {}".format(iou_mean))


if __name__ == '__main__':
    my_app()
    