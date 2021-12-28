""" Generate input point clouds from the dataloader
sanity check of the coordinates of the point clouds from the dataloader
"""

# load the dataloader with shuffle=False
# loop over the dataloader and get the point cloud, write it in plyfile
# name format: seq_main_frame_prev_frames
import os

from datasets.semantic_kitti_dataset import SemanticKittiDataSet
from torch.utils.data import DataLoader

from utils.debugging import d_print, write_pc, bcolors

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATASET_PATH = './data'


train_set = SemanticKittiDataSet(path=DATASET_PATH, set='train')
train_loader = DataLoader(train_set, shuffle=False, batch_size= 1, num_workers=4, pin_memory=True)

d_print(BASE_DIR, bcolors.OKBLUE)
for idx, batch in enumerate(train_loader):
    # get the point clouds
    # write it in ply file
    volume_id = batch['volume_name'][0]
    vol_pts = batch['in_fts'][0][:,:3].cpu().numpy()
    write_pc(vol_pts, "{}/viz/{}".format(BASE_DIR, volume_id))
    # d_print("Vol name: {} and its size {}".format((volume_id), vol_pts.shape))