import open3d as o3d
import numpy as np

from utils.ply import write_ply
from utils.config import Config
from datasets.SemanticKitti import *
from torch.utils.data import DataLoader

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# pcd = o3d.io.read_point_cloud("./test/_importance_None_str1_4_no_aug/val_probs/04_0000000_probs.ply")
# print(pcd)
# print(np.asarray(pcd.points))
# o3d.visualization.draw_geometries([pcd])


config = Config()
##################################
# Change model parameters for test
##################################

# Change parameters for the test here. For example, you can stop augmenting the input data.

config.global_fet = False
config.validation_size = 200
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

test_dataset = SemanticKittiDataset(config, set='val', balance_classes=False, seqential_batch=True)
test_sampler = SemanticKittiSampler(test_dataset)
collate_fn = SemanticKittiCollate
test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=collate_fn,
                             num_workers=0,#config.input_threads,
                             pin_memory=True)

for i, file_path in enumerate(test_loader.dataset.files):
    visual_path = "./visuals/"
    points = test_loader.dataset.load_evaluation_points(file_path)
    targets = test_loader.dataset.validation_labels[i]
    write_ply(visual_path,
                [points, targets],
                ['x', 'y', 'z', 'labels'])
    break