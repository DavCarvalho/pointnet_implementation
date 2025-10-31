# 1- Setup Libraries 

# Math library
import numpy as np
# Drawing library
import matplotlib.pyplot as plt
import open3d as o3d

# Deep Learning Libraries
import torch
import torch.nn.functional as nnf
import torch.nn as nn
import torchnet as tnt
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix

# Utility libraries
import copy
from glob import glob
import os
import functools
import mock
from tqdm.auto import tqdm
import time

# Check if CUDA is available and set Pytorch to use CPU or GPU accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device: ", device)


# 2- Point Cloud Preparation

def cloud_loader(tile_name, features_used):
    """
    Loads and normalizes point cloud data
    """
    cloud_data = np.loadtxt(tile_name).transpose()
    min_f = np.min(cloud_data, axis=1)
    mean_f = np.mean(cloud_data, axis=1)
    features = []
    
    if 'xyz' in features_used:
        n_coords = cloud_data[0:3]
        n_coords[0] -= mean_f[0]
        n_coords[1] -= mean_f[1]
        n_coords[2] -= min_f[2]
        features.append(n_coords)
    
    if 'rgb' in features_used:
        colors = cloud_data[3:6]
        features.append(colors)
    
    if 'i' in features_used:
        IQR = np.quantile(cloud_data[-2], 0.75) - np.quantile(cloud_data[-2], 0.25)
        n_intensity = ((cloud_data[-2] - np.median(cloud_data[-2])) / IQR)
        n_intensity -= np.min(n_intensity)
        features.append(n_intensity)
    
    gt = cloud_data[-1]
    gt = torch.from_numpy(gt).long()
    cloud_data = torch.from_numpy(np.vstack(features))
    
    return cloud_data, gt


def cloud_collate(batch):
    """
    Collects a list of samples into a batch list for clouds
    and a single array for labels
    """
    clouds, labels = list(zip(*batch))
    labels = torch.cat(labels, 0)
    return clouds, labels


class Tnet(nn.Module):
    def __init__(self, dim, num_points=2500):
        super(Tnet, self).__init__()
        self.dim = dim 
        self.conv1 = nn.Conv1d(dim, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, dim**2)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.max_pool = nn.MaxPool1d(kernel_size=num_points)

    def forward(self, x):
        bs = x.shape[0]
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.max_pool(x).view(bs, -1)
        x = self.bn4(F.relu(self.linear1(x)))
        x = self.bn5(F.relu(self.linear2(x)))
        x = self.linear3(x)
        iden = torch.eye(self.dim, requires_grad=True).repeat(bs, 1, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x.view(-1, self.dim, self.dim) + iden
        return x

# 3- Defining the PointNet Architecture

class PointNet(nn.Module):
    """PointNet for semantic segmentation with T-nets"""
    
    def __init__(self, MLP_1, MLP_2, MLP_3, n_classes=3, input_feat=3, subsample_size=51, cuda=1):
        """
        Initialization function
        MLP_1, MLP_2 and MLP_3: list of integers = width of multilayer perceptron layers
        n_classes: int = number of classes
        input_feat: int = number of input features
        subsample_size: int = number of points to which blocks are subsampled
        cuda: int = if 0, run on CPU, if 1, run on GPU
        """
        super(PointNet, self).__init__()
        self.is_cuda = cuda
        self.subsample_size = subsample_size

        m1 = MLP_1[-1]
        m2 = MLP_2[-1]

        # Add T-nets for spatial transformations
        self.tnet1 = Tnet(dim=input_feat, num_points=subsample_size)
        self.tnet2 = Tnet(dim=m1, num_points=subsample_size)

        # Create MLP_1 layers
        modules = []
        for i in range(len(MLP_1)):
            modules.append(
                nn.Conv1d(
                    in_channels=MLP_1[i-1] if i > 0 else input_feat,
                    out_channels=MLP_1[i],
                    kernel_size=1
                )
            )
            modules.append(nn.BatchNorm1d(MLP_1[i]))
            modules.append(nn.ReLU())
        self.mlp_1 = nn.Sequential(*modules)

        # Create MLP_2 layers
        modules = []
        for i in range(len(MLP_2)):
            modules.append(nn.Conv1d(MLP_2[i-1] if i > 0 else m1, MLP_2[i], 1))
            modules.append(nn.BatchNorm1d(MLP_2[i]))
            modules.append(nn.ReLU(True))
        self.mlp_2 = nn.Sequential(*modules)

        # Create MLP_3 layers
        modules = []
        for i in range(len(MLP_3)):
            modules.append(nn.Conv1d(MLP_3[i-1] if i > 0 else m1 + m2, MLP_3[i], 1))
            modules.append(nn.BatchNorm1d(MLP_3[i]))
            modules.append(nn.ReLU(True))
        modules.append(nn.Conv1d(MLP_3[-1], n_classes, 1))
        self.mlp_3 = nn.Sequential(*modules)

        # Maxpooling
        self.maxpool = nn.MaxPool1d(subsample_size)

        if cuda:
            self = self.cuda()

    def forward(self, input):
        """
        Forward function that produces embeddings for each point in 'input'
        input: [n_batch, input_feat, subsample_size] array of floats
        output: [n_batch, n_classes, subsample_size] array of floats
        """
        if self.is_cuda:
            input = input.cuda()
        
        # Apply first T-net for spatial transformation of input
        A_input = self.tnet1(input)
        input = torch.bmm(input.transpose(2, 1), A_input).transpose(2, 1)
        
        # Point embedding, equation (1)
        f1 = self.mlp_1(input)
        
        # Apply second T-net for feature space transformation
        A_feat = self.tnet2(f1)
        f1 = torch.bmm(f1.transpose(2, 1), A_feat).transpose(2, 1)
        
        # Second point embedding, equation (2)
        f2 = self.mlp_2(f1)
        
        # Maxpooling, equation (3)
        G = self.maxpool(f2)
        
        # Repeat G so that its dimension 2 has the same size as f1
        G = G.repeat(1, 1, f1.size(2))
        
        # Concatenate G and f1 in dimension 1 (the channels)
        Gf1 = torch.cat((G, f1), 1)
        
        # Global + local featuring, equation (4)
        out = self.mlp_3(Gf1)
        
        return out

# 5- Segmentation Definition

class PointCloudClassifier:
    """
    The main point cloud classifier class
    handles subsampling of tiles to a fixed number of points
    and interpolation to the original point cloud
    """

    def __init__(self, args):
        self.subsample_size = args.subsample_size
        self.n_inputs_feats = 3
        if 'i' in args.input_feats:
            self.n_inputs_feats += 1
        if 'rgb' in args.input_feats:
            self.n_inputs_feats += 3
        self.n_class = args.n_class
        self.is_cuda = args.cuda

    def run(self, model, clouds):
        """
        INPUT:
        model = the neural network
        clouds = list of n_batch tensors of size [n_feat, n_points_i]
        OUTPUT:
        pred = float tensor [sum_i n_points_i, n_class]
        """
        n_batch = len(clouds)
        prediction_batch = torch.zeros((self.n_class, 0))
        sampled_clouds = torch.Tensor(n_batch, self.n_inputs_feats, self.subsample_size)

        if self.is_cuda:
            prediction_batch = prediction_batch.cuda()

        for i_batch in range(n_batch):
            cloud = clouds[i_batch][:, :]
            n_points = cloud.shape[1]
            selected_points = np.random.choice(n_points, self.subsample_size)
            sampled_cloud = cloud[:, selected_points]
            sampled_clouds[i_batch, :, :] = sampled_cloud

        sampled_prediction = model(sampled_clouds)

        for i_batch in range(n_batch):
            cloud = clouds[i_batch][:3, :]
            sampled_cloud = sampled_clouds[i_batch, :3, :]

            knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(sampled_cloud.cpu().permute(1, 0))
            dump, closest_point = knn.kneighbors(cloud.permute(1, 0).cpu())
            closest_point = closest_point.squeeze()

            prediction_full_cloud = sampled_prediction[i_batch, :, closest_point]
            prediction_batch = torch.cat((prediction_batch, prediction_full_cloud), 1)

        return prediction_batch.permute(1, 0)


# 7- Metrics Definition

class ConfusionMatrix:
    """Class to calculate performance metrics"""
    
    def __init__(self, n_class, class_names):
        self.CM = np.zeros((n_class, n_class))
        self.n_class = n_class
        self.class_names = class_names
    
    def clear(self):
        self.CM = np.zeros((self.n_class, self.n_class))

    def add_batch(self, gt, pred):
        self.CM += confusion_matrix(gt, pred, labels=list(range(self.n_class)))

    def overall_accuracy(self):
        return 100 * self.CM.trace() / self.CM.sum()
    
    def class_IoU(self, show=1):
        ious = np.diag(self.CM) / (np.sum(self.CM, 1) + np.sum(self.CM, 0) - np.diag(self.CM))
        if show:
            print(' / '.join('{} : {:3.2f}%'.format(name, 100 * iou) for name, iou in zip(self.class_names, ious)))
        return 100 * np.nansum(ious) / (np.logical_not(np.isnan(ious))).sum()


# 8- Training Functions Definition

def train(model, PCC, optimizer, args, train_set, class_names):
    """Train for one epoch"""
    model.train()
    loader = torch.utils.data.DataLoader(train_set, collate_fn=cloud_collate, batch_size=args.batch_size, 
                                        shuffle=True, drop_last=True)
    loader = tqdm(loader, ncols=100, desc='Training Epoch', leave=False)
    loss_meter = tnt.meter.AverageValueMeter()
    cm = ConfusionMatrix(args.n_class, class_names=class_names[1:])
    
    for index_batch, (cloud, gt) in enumerate(loader):
        if PCC.is_cuda:
            gt = gt.cuda()
        
        optimizer.zero_grad()
        pred = PCC.run(model, cloud)
        labeled = gt != 0
        
        if labeled.sum() == 0:
            continue
        
        loss = nn.functional.cross_entropy(pred[labeled], gt[labeled] - 1)
        loss.backward()
        optimizer.step()
        
        loss_meter.add(loss.item())
        gt_labeled = gt[labeled].cpu().numpy() - 1
        pred_labeled = pred[labeled].argmax(1).cpu().detach().numpy()
        cm.add_batch(gt_labeled, pred_labeled)
    
    return cm, loss_meter.value()[0]


def eval(model, PCC, test, args, test_set, valid_set, class_names):
    """Evaluate the model on the test set/valid set"""
    model.eval()
    
    if test:
        loader = torch.utils.data.DataLoader(test_set, collate_fn=cloud_collate, batch_size=args.batch_size,
                                            shuffle=False)
        loader = tqdm(loader, ncols=500, leave=False, desc="Test")
    else:
        loader = torch.utils.data.DataLoader(valid_set, collate_fn=cloud_collate, batch_size=60, shuffle=False,
                                            drop_last=False)
        loader = tqdm(loader, ncols=500, leave=False, desc="Val")

    loss_meter = tnt.meter.AverageValueMeter()
    cm = ConfusionMatrix(args.n_class, class_names=class_names[1:])
    
    for index_batch, (cloud, gt) in enumerate(loader):
        if PCC.is_cuda:
            gt = gt.cuda()
        
        with torch.no_grad():
            pred = PCC.run(model, cloud)
        
        labeled = gt != 0
        if labeled.sum() == 0:
            continue
        
        loss = nn.functional.cross_entropy(pred[labeled], gt[labeled] - 1)
        loss_meter.add(loss.item())
        cm.add_batch((gt[labeled] - 1).cpu(), pred[labeled].argmax(1).cpu().detach().numpy())

    return cm, loss_meter.value()[0]


def train_full(args, train_set, valid_set, test_set, class_names):
    """The full training Loop"""
    model = PointNet(args.MLP_1, args.MLP_2, args.MLP_3, args.n_class, input_feat=args.n_input_feats,
                     subsample_size=args.subsample_size)
    
    print('Total number of parameters: {}'.format(sum([p.numel() for p in model.parameters()])))

    best_model = None
    best_mIoU = 0

    PCC = PointCloudClassifier(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    TESTCOLOR = '\033[104m'
    TRAINCOLOR = '\033[100m'
    VALIDCOLOR = '\033[45m'
    NORMALCOLOR = '\033[0m'

    metrics_pn = {}
    metrics_pn['definition'] = [['train_oa', 'train_mIoU', 'train_loss'], 
                                ['valid_oa', 'valid_mIoU', 'valid_loss'], 
                                ['test_oa', 'test_mIoU', 'test_loss']]

    for i_epoch in tqdm(range(args.n_epoch), desc='Training'):
        cm_train, loss_train = train(model, PCC, optimizer, args, train_set, class_names)
        mIoU = cm_train.class_IoU()

        tqdm.write(TRAINCOLOR + 'Epoch %3d -> Train overall accuracy : %3.2f%%, Train mIoU : %3.2f%%, Train Loss : %3.2f' % 
                  (i_epoch, cm_train.overall_accuracy(), mIoU, loss_train) + NORMALCOLOR)

        metrics_pn[i_epoch] = [[cm_train.overall_accuracy(), mIoU, loss_train]]

        cm_valid, loss_valid = eval(model, PCC, False, args, test_set, valid_set, class_names)
        mIoU_valid = cm_valid.class_IoU()
        metrics_pn[i_epoch].append([cm_valid.overall_accuracy(), mIoU_valid, loss_valid])

        best_valid = False
        if mIoU_valid > best_mIoU:
            best_valid = True
            best_mIoU = mIoU_valid
            best_model = copy.deepcopy(model)
            tqdm.write(VALIDCOLOR + 'Best performance achieved at epoch %3d -> Valid overall accuracy : %3.2f%%, Valid mIoU : %3.2f%%, Valid Loss : %3.2f' % 
                      (i_epoch, cm_valid.overall_accuracy(), mIoU_valid, loss_valid) + NORMALCOLOR)
        else:
            tqdm.write(VALIDCOLOR + 'Epoch %3d -> Valid overall accuracy : %3.2f%%, Valid mIoU : %3.2f%%, Valid Loss : %3.2f' % 
                      (i_epoch, cm_valid.overall_accuracy(), mIoU_valid, loss_valid) + NORMALCOLOR)
        
        if i_epoch == args.n_epoch - 1 or best_valid:
            cm_test, loss_test = eval(best_model, PCC, True, args, test_set, valid_set, class_names)
            mIoU = cm_test.class_IoU()
            tqdm.write(TESTCOLOR + 'Epoch %3d -> Test overall accuracy : %3.2f%%, Test mIoU : %3.2f%%, Test Loss : %3.2f' % 
                      (i_epoch, cm_test.overall_accuracy(), mIoU, loss_test) + NORMALCOLOR)
            
            metrics_pn[i_epoch].append([cm_test.overall_accuracy(), mIoU, loss_test])
    
    return best_model, metrics_pn


# 11- PointNet Prediction Visualization

def tile_prediction(tile_name, model=None, PCC=None, Visualization=True, features_used='xyzrgbi'):
    """Performs prediction on a point cloud and visualizes the result"""
    cloud, gt = cloud_loader(tile_name, features_used)
    labels = PCC.run(model, [cloud])
    labels = labels.argmax(1).cpu() + 1
    
    xyz = np.array(cloud[0:3]).transpose()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    if Visualization == True:
        max_label = labels.max()
        colors = plt.get_cmap("tab10")(labels / (max_label if max_label > 0 else 1))
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        pcd.estimate_normals(fast_normal_computation=True)
        o3d.visualization.draw_geometries([pcd])
    
    return pcd, labels


# Main function to run the complete pipeline

def main():
    """Main function to execute the training pipeline"""
    
    # 2- Prepare data
    project_dir = "./DATA/"
    pointcloud_train_files = glob(os.path.join(project_dir, "train/*.txt"))
    pointcloud_test_files = glob(os.path.join(project_dir, "test/*.txt"))

    valid_index = np.random.choice(len(pointcloud_train_files), int(len(pointcloud_train_files) / 5), replace=False)
    valid_list = [pointcloud_train_files[i] for i in valid_index]
    train_list = [pointcloud_train_files[i] for i in np.setdiff1d(list(range(len(pointcloud_train_files))), valid_index)]
    test_list = pointcloud_test_files

    print("%d tiles in train set, %d tiles in validation set, %d tiles in test set" % 
          (len(train_list), len(valid_list), len(test_list)))

    cloud_features = "xyzrgbi"
    test_set = tnt.dataset.ListDataset(test_list, functools.partial(cloud_loader, features_used=cloud_features))
    train_set = tnt.dataset.ListDataset(train_list, functools.partial(cloud_loader, features_used=cloud_features))
    valid_set = tnt.dataset.ListDataset(valid_list, functools.partial(cloud_loader, features_used=cloud_features))

    # 9- Define parameters
    args = mock.Mock()
    class_names = ['unclassified', 'vegetation', 'ground', 'buildings']
    args.n_epoch = 42
    args.subsample_size = 2048
    args.batch_size = 8
    args.n_class = len(class_names) - 1
    args.input_feats = 'xyzrgbi'
    args.n_input_feats = len(args.input_feats)
    args.MLP_1 = [32, 32]
    args.MLP_2 = [32, 64, 256]
    args.MLP_3 = [128, 64, 32]
    args.lr = 5e-3
    args.wd = 0
    args.cuda = 1

    # 10- Train model
    t0 = time.time()
    trained_model, metrics_pn = train_full(args, train_set, valid_set, test_set, class_names)
    t1 = time.time()

    print(trained_model)
    print(f"{'-'*50}")
    print(f"Total training time: {t1-t0} seconds")
    print(f"{'='*50}")

    # 12- Export model
    torch.save(trained_model.state_dict(), './pointnet_model_' + project_dir.split("DATA/")[1] + '.torch')

    with open("./metrics_" + project_dir.split("DATA/")[1] + ".csv", "w") as f:
        for key, value in metrics_pn.items():
            f.write("%s,%s\n" % (key, value))

    return trained_model, metrics_pn


if __name__ == "__main__":
    main()