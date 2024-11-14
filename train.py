# -*- coding: utf-8 -*-
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from sklearn.metrics import confusion_matrix, roc_auc_score
import numpy as np
import os
import tqdm
from torch.autograd import Variable
from PIL import Image
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from data_loader import *
import argparse
import warnings
from draw import *
from metrics import *
from focal_loss import FocalLoss

warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

torch.manual_seed(42)

df = pd.read_csv('冠心病患者认知评测记录_内部_截至23年7月.csv', encoding='gbk')

img_list = []
label_list = []

label = 'psqi'
print(len(df))
df.dropna(subset=[label], inplace=True)
df[label] = df[label].apply(lambda x: x if len(str(x).split(',')) == 1 else max(str(x).split(',')))
if label == 'mmse':
    df[label] = df[label].apply(lambda x: 1 if float(x) < 27 else 0) # mmse
elif label == 'moca':
    df[label] = df[label].apply(lambda x: 1 if float(x) < 26 else 0) # moca
elif label == 'psqi':
    df[label] = df[label].apply(lambda x: 1 if float(x) >= 7 else 0) # psqi

def func_path(x):
    if type(x) == list:
        x = [i.replace('ikang/','img/') for i in x]
        x = ['img/'+i for i in x if not i.startswith('img')]
    elif type(x) == str:
        x = x.replace('ikang/','img/').replace('[','').replace(']','').replace("'","").replace(' ','').split(',')
    return x
df['path'] = df['path'].apply(func_path)

list_1 = df['path'].values.tolist()
list_2 = df[label].values.tolist()
print('0:{} 1:{}'.format(df[label].values.tolist().count(0), df[label].values.tolist().count(1)))
X_train, X_test, y_train, y_test = train_test_split(list_1, list_2, test_size=0.2, random_state=42)

print('train num:{} 0:{} 1:{}'.format(len(y_train), y_train.count(0), y_train.count(1)))
print('test num:{} 0:{} 1:{}'.format(len(y_test), y_test.count(0), y_test.count(1)))

def list_func(X, y):
    img_list = []
    label_list = []
    for i in range(len(X)):
        for j in range(len(X[i])):
            label_list.append(y[i])
        img_list.extend(X[i])
    return img_list, label_list

img_list_train, label_list_train = list_func(X_train, y_train)
img_list_test, label_list_test = list_func(X_test, y_test)

print('train img num: {}'.format(len(img_list_train)))
print('test img num: {}'.format(len(img_list_test)))

data_transforms = {
    'train': transforms.Compose([
                                    transforms.Resize((512,512)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ]),
    'test': transforms.Compose([
                                   transforms.Resize((512,512)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                   ])
                    }

train_dataset = customData(img_list_train, label_list_train, '', dataset = 'train', data_transforms=data_transforms)
test_dataset = customData(img_list_test, label_list_test, '', dataset = 'test', data_transforms=data_transforms)

"""
batch size
resnet18 128
resnet50 50
resnet101 32
inceptionv3 58
densenet121 32
"""
train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=50,
                                            shuffle=True
                                               )

test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=50
                                              )

model_name = 'resnet50'
if model_name == 'resnet101':
    net = models.resnet101(num_classes = 2)
    pretrained_dict = torch.load('/mnt/workdir/xiapeng/pretrained_models/resnet101-5d3b4d8f.pth')
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'fc' not in k)}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    # net.load_state_dict(torch.load('/mnt/workdir/xiapeng/cognitive/model/clean/resnet101_valid_chd_mmse.pth'))

elif model_name == 'resnet18':# resnet18: (fc): Linear(in_features=512, out_features=2, bias=True)
    net = models.resnet18(num_classes = 2)
    pretrained_dict = torch.load('/mnt/workdir/xiapeng/pretrained_models/resnet18-5c106cde.pth')
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'fc' not in k)}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    # net.load_state_dict(torch.load('/mnt/workdir/xiapeng/cognitive/model/clean/resnet18_valid_chd_mmse.pth'))

elif model_name == 'resnet50': # (fc): Linear(in_features=2048, out_features=2, bias=True)
    net = models.resnet50(num_classes = 2)
    pretrained_dict = torch.load('/mnt/workdir/xiapeng/pretrained_models/resnet50-19c8e357.pth')
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'fc' not in k)}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    # net.load_state_dict(torch.load('/mnt/workdir/xiapeng/cognitive/model/clean/resnet50_valid_chd_mmse.pth'))

elif model_name == 'inceptionv3': # (fc): Linear(in_features=2048, out_features=2, bias=True)
    net = models.inception_v3(num_classes = 2, aux_logits=False)
    pretrained_dict = torch.load('/mnt/workdir/xiapeng/pretrained_models/inception_v3_google-1a9a5a14.pth')
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'fc' not in k)}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    # net.load_state_dict(torch.load('/mnt/workdir/xiapeng/cognitive/model/clean/inceptionv3_valid_chd_mmse.pth'))

elif model_name == 'vgg16': # classifier[6]: Linear(in_features=4096, out_features=1000, bias=True)
    net = models.vgg16(pretrained=False)
    pretrained_dict = torch.load('/mnt/workdir/xiapeng/pretrained_models/vgg16-397923af.pth')
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'fc' not in k)}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    net.classifier[6] = nn.Linear(4096,2)
    # net.load_state_dict(torch.load('/mnt/workdir/xiapeng/cognitive/model/clean/vgg16_valid_chd_mmse.pth'))

elif model_name == 'densenet121': # (classifier): Linear(in_features=1024, out_features=1000, bias=True)
    net = models.densenet121()
    pretrained_dict = torch.load('/mnt/workdir/xiapeng/pretrained_models/densenet121-a639ec97.pth')
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'fc' not in k)}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    in_features = net.classifier.in_features
    net.classifier = nn.Linear(in_features,2)
    # net.load_state_dict(torch.load('/mnt/workdir/xiapeng/cognitive/model/clean/densenet121_valid_chd_mmse.pth'))


net.cuda()
# loss_function = nn.CrossEntropyLoss()
loss_function = FocalLoss(gamma=2, alpha=0.25)

params = [p for p in net.parameters() if p.requires_grad]
optimizer = optim.Adam(params, lr=1e-4)
exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience = 5, mode = 'max', verbose = True, min_lr = 1e-7)

train_losses = []
train_aucs = []

test_losses = []
test_aucs = []

best_auc = 0.0

train_steps = len(train_loader)
test_steps = len(test_dataset)

epochs = 30
patience = 10

for epoch in range(epochs):
    cnt = 0
    if cnt >= patience:
        break
    net.train()
    running_loss = 0.0
    gt_labels = []
    predict_p = []
    for data in tqdm.tqdm(train_loader) :
        optimizer.zero_grad()
        img_name, inputs, labels = data
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        outputs = net(inputs)
        gt_labels.extend(labels.cpu().numpy().tolist())
        predict_p.extend(nn.functional.softmax(outputs).cpu().detach().numpy()[:,1])
        loss = loss_function(outputs, labels)
        running_loss += loss.data.item()
        loss.backward()
        optimizer.step()

    print("train epoch[{}/{}] loss:{:.3f} train AUC:{}".format(epoch + 1, epochs, loss, roc_auc_score(gt_labels, predict_p)))
    train_epoch_loss = running_loss/train_steps
    train_losses.append(train_epoch_loss)
    train_aucs.append(roc_auc_score(gt_labels,predict_p))

    net.eval()

    with torch.no_grad():
        gt_labels = []
        predict_p = []
        running_loss = 0.0
        for data in tqdm.tqdm(test_loader):
            img_name, inputs, labels = data
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
            outputs = net(inputs)
            gt_labels.extend(labels.cpu().numpy().tolist())
            predict_p.extend(nn.functional.softmax(outputs).cpu().detach().numpy()[:,1])
            loss = loss_function(outputs, labels)
            running_loss += loss.data.item()
        print("test epoch[{}/{}] loss:{:.3f} test AUC:{}".format(epoch + 1, epochs, loss, roc_auc_score(gt_labels, predict_p)))
        test_epoch_loss = running_loss / test_steps
        test_losses.append(test_epoch_loss)
        test_aucs.append(roc_auc_score(gt_labels,predict_p))
        current_auc = roc_auc_score(gt_labels,predict_p)
        exp_lr_scheduler.step(current_auc)
        if current_auc > best_auc:
            best_auc = current_auc
            torch.save(net.state_dict(), './model/'+label+'/'+model_name+'_chd_2307_07.pth')
        else:
            cnt += 1

print('Finished Training')
