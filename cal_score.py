'''Train CIFAR with PyTorch.
e.g.
    python3 cifar.py --netName=PreActResNet18 --cifar=10 --bs=512
'''
from __future__ import print_function

import libmr

import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import scipy.spatial.distance as spd
from sklearn.metrics import f1_score, accuracy_score

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from alexnet import AlexNet
from utils import *
from openmax import *


os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--bs', default=512, type=int, help='batch size')
parser.add_argument('--es', default=160, type=int, help='epoch size')
args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
known_index = torch.tensor(np.where(np.array(trainset.targets)<5)[0])
targets = torch.tensor(trainset.targets)[known_index]
data = torch.tensor(trainset.data)[known_index,:,:,:]
trainset.data = data.numpy()
trainset.targets = targets.tolist()
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=4)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
targets = torch.tensor(testset.targets)
targets[targets >= 5] = 5
testset.targets = targets.tolist()
testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')

net = AlexNet(5)

checkpoint_path = 'cifar_10_alexnet.t7'
checkpoint = torch.load(checkpoint_path,map_location=torch.device('cpu'))
try:
    net.load_state_dict(checkpoint['net'])
except:
    new_check_point = OrderedDict()
    for k, v in checkpoint['net'].items():
        name = k[7:]  # remove `module.`
        # name = k[9:]  # remove `module.1.`
        new_check_point[name] = v
    net.load_state_dict(new_check_point)


net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True



criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

#
# # Training
# def main():
#     AV = [[] for _ in range(5)]
#     MAV = [[] for _ in range(5)]
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(trainloader):
#             print("Processing {} batches in {}".format(batch_idx, 25000//args.bs))
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = net(inputs)
#             for score, t in zip(outputs, targets):
#                 if np.argmax(score) == t:
#                     AV[t].append(score.unsqueeze(dim=0))
#     for i in range(5):
#         MAV[i] = torch.cat(AV[i],dim=0).mean(dim=0).unsqueeze(dim=0)
#     MAV = torch.cat(MAV,dim=0) # shape: [C,C]
#     distance = [[] for _ in range(5)]

def compute_channel_distances(mavs, features, eu_weight=0.5):
    """
    Input:
        mavs (channel, C)
        features: (N, channel, C)
    Output:
        channel_distances: dict of distance distribution from MAV for each channel.
    """
    eucos_dists, eu_dists, cos_dists = [], [], []
    for channel, mcv in enumerate(mavs):  # Compute channel specific distances
        eu_dists.append([spd.euclidean(mcv, feat[channel]) for feat in features])
        cos_dists.append([spd.cosine(mcv, feat[channel]) for feat in features])
        eucos_dists.append([spd.euclidean(mcv, feat[channel]) * eu_weight +
                            spd.cosine(mcv, feat[channel]) for feat in features])

    return {'eucos': np.array(eucos_dists), 'cosine': np.array(cos_dists), 'euclidean': np.array(eu_dists)}


def main():
    scores = [[] for _ in range(5)]
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            print("Processing {} batches in {}".format(batch_idx, 25000 // args.bs))
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            for score, t in zip(outputs, targets):
                if np.argmax(score) == t:
                    scores[t].append(score.unsqueeze(dim=0).unsqueeze(dim=0))
    scores = [torch.cat(x).numpy() for x in scores]  # (N_c, 1, C) * C
    mavs = np.array([np.mean(x, axis=0) for x in scores])  # (C, 1, C)

    dists = [compute_channel_distances(mcv, score) for mcv, score in zip(mavs, scores)]

    scores, labels = [], []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            print("Predicting {} batches in {}".format(batch_idx, 10000 // args.bs))
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            for score, t in zip(outputs, targets):
                if np.argmax(score) == t:
                    scores.append(score.numpy())
                    labels.append(t.numpy())
    scores = np.array(scores)[:, np.newaxis, :]
    labels = np.array(labels)
    categories = ["airplane", "automobile", "bird", "cat", "deer"] # First five cates
    weibull_model = fit_weibull(mavs, dists, categories, 80, "euclidean")
    alpha = 3
    threshold = 0.9
    pred_y, pred_y_o = [], []
    for score in scores:
        so, ss = openmax(weibull_model, categories, score,
                         0.5, alpha, "euclidean") # openmax_prob, softmax_prob
        pred_y.append(np.argmax(ss) if np.max(ss) >= threshold else 5)
        pred_y_o.append(np.argmax(so) if np.max(so) >= threshold else 5)

    print(accuracy_score(labels, pred_y), accuracy_score(labels, pred_y_o))
    openmax_score = f1_score(labels, pred_y_o, average="macro")
    print(f1_score(labels, pred_y, average="macro"), openmax_score)

if __name__ == "__main__":
    main()


