

#1 Setup ambiente

import numpy as np

import matplotlib.pyplot as plt
import open3d as o3d

import torch
from pointnet_arquitetura import cloud_loader, PointNet, PointCloudClassifier

from glob import glob
import os
import mock
import time


#2 data check e qualidade check
"""
Aqui, carregaremos previamente os dados (tiles, seguuindo a mesma estrtura da pointnet)
*usamos a mesam função que foi usada para preparar os dados, para carregar as nuvems
"""
project_dir = "./DATA/"
#gather either the path to the point cloud you want predict on, or a list of paths
inference_list = glob(os.path.join(project_dir, "test/*.txt"))


#3 seettings os parametros e hyperparametros
cloud_features = "xyzrgbi"
class_names = ['unclassified', 'vegetation', 'ground', 'buildings']
gpu_use = 0

# definindo os argumentos como usado no treinamento
args = mock.Mock()
args.n_class = len(class_names)-1
args.input_feats = cloud_features
args.subsample_size = 2048
args.cuda = gpu_use

#4 Incializando PointCloudClassifier 
loaded_PCC = PointCloudClassifier(args)

#5 loading the training model
loaded_trained_model = PointNet([32,32], [32, 64, 256], [128, 64, 32], n_classes=len(class_names)-1, input_feat=7, subsample_size=args.subsample_size, cuda=gpu_use)

loaded_trained_model.load_state_dict(torch.load("./pointnet_model_.torch"))

#6 PointNet prediction visualization

def tile_prediction(tile_name, model=None, PCC= None, Visualization=True, features_used="xyzrgbi"):
    #load the tile
    cloud, gt = cloud_loader(tile_name, features_used)

    t0 = time.time()
    labels = PCC.run(model,[cloud])
    t1 = time.time()
    print(f"Prediction of Poitn cloud with {np.shape(cloud)[-1]} pts in> {round(t1-t0, 5)}s")

    labels = labels.argmax(1).cpu() + 1

    #prepare the data for export
    xyz = np.array(cloud[0:3]).transpose()

    #data for open3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    #visualizar no pen3d
    if Visualization == True:
        max_label = labels.max()
        colors = plt.get_cmap("tab10")(labels / (max_label if max_label > 0 else 1))
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        pcd.estimate_normals(fast_normal_computation=True)
        o3d.visualization.draw_geometries([pcd])

    return pcd, labels

#selection = inference_list[3]
#pcd,labels = tile_prediction(selection, model=loaded_trained_model, PCC=loaded_PCC)

# 7 Analise da qualidade da prediçao

for i in range(5):
    selection = inference_list[i]
    pcd, labels = tile_prediction(selection, model=loaded_trained_model, PCC=loaded_PCC)

# 8 Pointnet Single Generalization Test


# XX 9. 3D Point Cloud Export with the predictions
pcd_np = np.hstack((np.asarray(pcd.points), (labels).reshape(-1, 1)))
np.savetxt("RESULTS/" + inference_list[1].split("test\\")[1], pcd_np, fmt='%1.4f', delimiter=' ')

