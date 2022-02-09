"""
This script is used as overfitting experiment for different models on semantic kitti dataset 
on the task of semantic segmentation
"""
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from models.blocks import global_average


from models.vn_layers import *
from datasets.semantic_kitti_sem_seg import SemanticKittiSegDataset
from utils.debugging import d_print

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import sklearn.metrics as metrics



def calculate_sem_IoU(pred_np, seg_np, visual=False):
    I_all = np.zeros(20)
    U_all = np.zeros(20)
    for sem_idx in range(seg_np.shape[0]):
        for sem in range(20):
            I = np.sum(np.logical_and(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            U = np.sum(np.logical_or(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            I_all[sem] += I
            U_all[sem] += U
    if visual:
        for sem in range(20):
            if U_all[sem] == 0:
                I_all[sem] = 1
                U_all[sem] = 1
    return I_all / U_all 


CLASS_WEIGHTS = np.array(
    [3.1501833e-02, 4.2607829e-02, 1.6609539e-04, 3.9838615e-04, 2.1649399e-03,
    1.8070553e-03, 3.3758327e-04, 1.2711105e-04, 3.7461064e-05, 1.9879648e-01,
    1.4717169e-02, 1.4392298e-01, 3.9048553e-03, 1.3268620e-01, 7.2359227e-02,
    2.6681501e-01, 6.0350122e-03, 7.8142218e-02, 2.8554981e-03, 6.1559578e-04]
    )


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean', weight=CLASS_WEIGHTS)

    return loss


# create model
def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, x_coord=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if x_coord is None: # dynamic knn graph
            idx = knn(x, k=k)
        else:          # fixed knn graph with input point coordinates
            idx = knn(x_coord, k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3) 
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 4, 1, 2).contiguous()
  
    return feature

class get_model(nn.Module):
    def __init__(self, num_class=20, normal_channel=False):
        super(get_model, self).__init__()
        # self.args = args
        self.n_knn = 20
        self.pooling = 'mean'
        
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)
        
        self.conv1 = VNLinearLeakyReLU(2, 64//3)
        self.conv2 = VNLinearLeakyReLU(64//3, 64//3)
        self.conv3 = VNLinearLeakyReLU(64//3*2, 64//3)
        self.conv4 = VNLinearLeakyReLU(64//3, 64//3)
        self.conv5 = VNLinearLeakyReLU(64//3*2, 64//3)
        
        if self.pooling == 'max':
            self.pool1 = VNMaxPool(64//3)
            self.pool2 = VNMaxPool(64//3)
            self.pool3 = VNMaxPool(64//3)
        elif self.pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
        
        self.conv6 = VNLinearLeakyReLU(64//3*3, 1024//3, dim=4, share_nonlinearity=True)
        self.std_feature = VNStdFeature(1024//3*2, dim=4, normalize_frame=False)
        self.conv8 = nn.Sequential(nn.Conv1d(2235, 256, kernel_size=1, bias=False),
                               self.bn8,
                               nn.LeakyReLU(negative_slope=0.2))
        
        # self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
        #                            self.bn7,
        #                            nn.LeakyReLU(negative_slope=0.2))
        
        self.dp1 = nn.Dropout(p=0.5)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=0.5)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn10,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, num_class, kernel_size=1, bias=False)


    def forward(self, x):
        """
        x: shape [B, feats, num_points]
        """
        batch_size = x.size(0)
        num_points = x.size(2) 
        
        x = x.unsqueeze(1) # [B, feats, num_points] -> [B, 1, feats, num_points]
        
        x = get_graph_feature(x, k=self.n_knn)
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = self.pool1(x)
        
        x = get_graph_feature(x1, k=self.n_knn)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = self.pool2(x)
        
        x = get_graph_feature(x2, k=self.n_knn)
        x = self.conv5(x)
        x3 = self.pool3(x)
        
        x123 = torch.cat((x1, x2, x3), dim=1)
        
        x = self.conv6(x123)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x, z0 = self.std_feature(x)
        x123 = torch.einsum('bijm,bjkm->bikm', x123, z0).view(batch_size, -1, num_points)
        x = x.view(batch_size, -1, num_points)
        x = x.max(dim=-1, keepdim=True)[0]

        # l = l.view(batch_size, -1, 1)
        # l = self.conv7(l)

        # x = torch.cat((x, l), dim=1)
        x = x.repeat(1, 1, num_points)

        x = torch.cat((x, x123), dim=1)

        x = self.conv8(x)
        x = self.dp1(x)
        x = self.conv9(x)
        x = self.dp2(x)
        x = self.conv10(x)
        x = self.conv11(x)
        
        trans_feat = None
        return x.transpose(1, 2).contiguous()


@hydra.main(config_path="conf/sem_seg_vn_dgcnn", config_name="config")
def my_app(cfg : DictConfig) -> None:
    
    
    from addict import Dict
    args = Dict()
    args.lr = 0.001/2
    args.momentum = 0.9
    args.epochs = 100
    args.scheduler = 'cos'

    train_logger = SummaryWriter(log_dir="vn_dgcnn_train")
    val_logger = SummaryWriter(log_dir="vn_dgcnn_val")



    model = get_model().to(device)
    train_set = SemanticKittiSegDataset(mode='train')
    test_set = SemanticKittiSegDataset(mode='val')
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8)
    
    opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    
    # opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, 20, 0.5, args.epochs)

    criterion = cal_loss

    best_test_iou = 0
    global_step = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []
        train_label_seg = []

        for i, batch in enumerate(train_loader):
            data, seg = batch
            seg = seg.long()
            data, seg = data.to(device), seg.to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            seg_pred = model(data)            
            # seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = criterion(seg_pred.view(-1, 20).cpu(), seg.view(-1,1).squeeze().cpu(), smoothing=True)
            train_logger.add_scalar("loss", loss.item(), global_step)
            print("loss: {}".format(loss.item()))
            loss.backward()
            opt.step()
            pred = seg_pred.max(dim=2)[1]               # (batch_size, num_points)
            count += batch_size
            train_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()                  # (batch_size, num_points)
            pred_np = pred.detach().cpu().numpy()       # (batch_size, num_points)
            train_true_cls.append(seg_np.reshape(-1))       # (batch_size * num_points)
            train_pred_cls.append(pred_np.reshape(-1))      # (batch_size * num_points)
            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)
            global_step += 1
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)
        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
        train_true_seg = np.concatenate(train_true_seg, axis=0)
        train_pred_seg = np.concatenate(train_pred_seg, axis=0)
        train_ious = calculate_sem_IoU(train_pred_seg, train_true_seg)
        for i, iou in enumerate(train_ious):
            train_logger.add_scalar("metrics/iou/class_{}".format(i), iou, global_step)
        train_logger.add_scalar("train/iou/mean".format(i), np.mean(train_ious), global_step)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f, train iou: %.6f' % (epoch, 
                                                                                                  train_loss*1.0/count,
                                                                                                  train_acc,
                                                                                                  avg_per_class_acc,
                                                                                                  np.mean(train_ious))
        d_print(outstr)

        ####################
        # Test
        ####################
        d_print(">>>> test run")
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_true_cls = []
        test_pred_cls = []
        test_true_seg = []
        test_pred_seg = []
        with torch.no_grad():
            for data, seg in test_loader:
                data, seg = data.to(device), seg.to(device)
                seg = seg.long()
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                seg_pred = model(data)
                # seg_pred = seg_pred.permute(0, 2, 1).contiguous()
                loss = criterion(seg_pred.reshape(-1, 20), seg.view(-1,1).squeeze())
                pred = seg_pred.max(dim=2)[1]
                count += batch_size
                test_loss += loss.item() * batch_size
                seg_np = seg.cpu().numpy()
                pred_np = pred.detach().cpu().numpy()
                test_true_cls.append(seg_np.reshape(-1))
                test_pred_cls.append(pred_np.reshape(-1))
                test_true_seg.append(seg_np)
                test_pred_seg.append(pred_np)
            test_true_cls = np.concatenate(test_true_cls)
            test_pred_cls = np.concatenate(test_pred_cls)
            test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
            avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
            test_true_seg = np.concatenate(test_true_seg, axis=0)
            test_pred_seg = np.concatenate(test_pred_seg, axis=0)
            test_ious = calculate_sem_IoU(test_pred_seg, test_true_seg)
            outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (epoch,
                                                                                                test_loss*1.0/count,
                                                                                                test_acc,
                                                                                                avg_per_class_acc,
                                                                                                np.mean(test_ious))
            d_print(outstr)
            if np.mean(test_ious) >= best_test_iou:
                best_test_iou = np.mean(test_ious)
                torch.save(model.state_dict(), 'outputs/%s/models/model_%s.t7' % (args.exp_name, args.test_area))


if __name__ == '__main__':
    my_app()