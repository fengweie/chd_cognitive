# -*- coding: utf-8 -*-
# +
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
import argparse
import copy
import ast
warnings.filterwarnings('ignore')

# os.environ["CUDA_VISIBLE_DEVICES"] = '3'
train_arg_parser = argparse.ArgumentParser(description="parser")
train_arg_parser.add_argument('--arc', type=str,
                    default='resnet50', help='select one model')
train_arg_parser.add_argument('--label', type=str,
                    default='mmse', help='select one model')
train_arg_parser.add_argument('--batch_size', default=128, type=int)
args = train_arg_parser.parse_args()
torch.manual_seed(42)

df_internal = pd.read_csv('/mnt/workdir/fengwei/xiapeng/SOP/anzhen_chd_cognitive/datasets/冠心病患者认知评测记录_内部_截至23年7月.csv', encoding='gbk')
df_external = pd.read_csv('/mnt/workdir/fengwei/xiapeng/SOP/anzhen_chd_cognitive/datasets/冠心病患者认知评测记录_外部_截至23年7月.csv', encoding='gbk')

df_all = pd.concat([df_internal,df_external])
df_all["测评日期"] = pd.to_datetime(df_all["测评日期"])
df_train = df_all[df_all['测评日期']<=pd.datetime(2023,5,31)]
df_train = df_train.reset_index(drop=True) 
df_test = df_all[df_all['测评日期']>pd.datetime(2023,5,31)]
df_test = df_test.reset_index(drop=True) 
print("总数量:{}, 内部所有:{}, 外部所有:{}".format(len(df_all),len(df_train),len(df_test)))
# df_all.drop_duplicates(subset=['患者id'], keep='first', inplace=True)
# print("总数量:{}".format(len(df_all)))
def func_path(x):
    x = x.replace('img/','/mnt/workdir/fengwei/xiapeng/cognitive_guanxin/img_resize/').replace('ikang/','/mnt/workdir/fengwei/xiapeng/cognitive_guanxin/img_resize/')

#     x = ast.literal_eval(x)
#     x = [i.replace('img/','/mnt/workdir/fengwei/xiapeng/cognitive_guanxin/img_resize/') for i in x]
#     x = [i.replace('ikang/','/mnt/workdir/fengwei/xiapeng/cognitive_guanxin/img_resize/') for i in x]    
#     if type(x) == list:
#         x = [i.replace('img/','/mnt/workdir/fengwei/xiapeng/cognitive_guanxin/img/') for i in x]
#         x = [i.replace('ikang/','/mnt/workdir/fengwei/xiapeng/cognitive_guanxin/img/') for i in x]

# #         x = ['/mnt/workdir/fengwei/xiapeng/cognitive_guanxin/img/'+i for i in x if not i.startswith('/mnt/workdir/fengwei/xiapeng/cognitive_guanxin/img')]
#     elif type(x) == str:
# #         x = x.replace('img/','/mnt/workdir/fengwei/xiapeng/cognitive_guanxin/img/').replace('[','').replace(']','').replace("'","").replace(' ','').split(',')
#         x = x.replace('ikang/','/mnt/workdir/fengwei/xiapeng/cognitive_guanxin/img/').replace('[','').replace(']','').replace("'","").replace(' ','').split(',')
    return x
list_2_3_4_set = []
for row in df_train.index:
    danhao_one = ast.literal_eval(df_train["path"][row])
    for img_path_one in danhao_one:
        list_2_3_4_set.append(df_train.iloc[row].values.tolist() +  [img_path_one])

df_train = pd.DataFrame(list_2_3_4_set,
                        columns=df_train.columns.tolist()+['path_one'])
df_train["path_one"] = df_train["path_one"].apply(func_path)

list_2_3_4_set = []
for row in df_test.index:
    danhao_one = ast.literal_eval(df_test["path"][row])
    for img_path_one in danhao_one:
        list_2_3_4_set.append(df_test.iloc[row].values.tolist() +  [img_path_one])

df_test = pd.DataFrame(list_2_3_4_set,
                        columns=df_test.columns.tolist()+['path_one'])
df_test["path_one"] = df_test["path_one"].apply(func_path)


# # ###qc
qc_df = pd.DataFrame(pd.read_csv("/mnt/workdir/fengwei/xiapeng/cognitive_guanxin/data_231008_qc.csv"))
qc_df["path"] = qc_df["path"].apply(lambda x: x.replace('img/','/mnt/workdir/fengwei/xiapeng/cognitive_guanxin/img_resize/'))
qc_df = qc_df.rename(columns={'path':'path_one'})
qc_df = qc_df.drop(["check_id","uuid","patient_id"],axis=1)
df_test = pd.merge(df_test, qc_df,on=['path_one'],how='left')
df_train = pd.merge(df_train, qc_df,on=['path_one'],how='left')
print("drop qc img前df_train:{}".format(len(df_train)))
print("drop qc img前df_test:{}".format(len(df_test)))
df_train = df_train.drop(df_train[df_train['label']==295].index)
df_train = df_train.drop(df_train[df_train['label']==5].index)
df_test = df_test.drop(df_test[df_test['label']==295].index)
df_test = df_test.drop(df_test[df_test['label']==5].index)
df_train.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)
print("drop qc img后df_train:{}".format(len(df_train)))
print("drop qc img后df_test:{}".format(len(df_test)))
# # ###qc


print("drop path img前df_train:{}".format(len(df_train)))
df_train.drop_duplicates(subset=['path_one'], keep='first', inplace=True)
df_train.reset_index(drop=True, inplace=True)
print("drop path img后df_train:{}".format(len(df_train)))

print("drop path img前df_test:{}".format(len(df_test)))
df_test.drop_duplicates(subset=['path_one'], keep='first', inplace=True)
df_test.reset_index(drop=True, inplace=True)
print("drop path img后df_test:{}".format(len(df_test)))

print("内部所有img:{}, 外部测试img:{}".format(len(df_train),len(df_test)))

df = copy.copy(df_train)

img_list = []
label_list = []

label = args.label
print("drop 前:{}".format(len(df)))
df.dropna(subset=[label], inplace=True)
print("drop 后:{}".format(len(df)))
df[label] = df[label].apply(lambda x: x if len(str(x).split(',')) == 1 else max(str(x).split(',')))
if label == 'mmse':
    df[label] = df[label].apply(lambda x: 1 if float(x) < 27 else 0) # mmse
elif label == 'moca':
    df[label] = df[label].apply(lambda x: 1 if float(x) < 26 else 0) # moca
elif label == 'psqi':
    df[label] = df[label].apply(lambda x: 1 if float(x) >= 7 else 0) # psqi


# print(df['path'])
# list_1 = df['path_one'].values.tolist()
# list_2 = df[label].values.tolist()

# # danhao_unique_id_second = df['患者id'].drop_duplicates().values.tolist()
# danhao_unique_id_second = df['患者id'].values.tolist()
# print("人数一共有:{}".format(len(danhao_unique_id_second)))
# import random
# random.seed(42)
# test_idx = random.sample(danhao_unique_id_second, int(0.2*len(danhao_unique_id_second)))
# '''取list中不包含的元素'''
# train_idx = [x for x in danhao_unique_id_second if x not in test_idx]
# test_data = df[df['患者id'].isin(test_idx)]
# # danhao_unique_id_test_data = test_data['患者id'].drop_duplicates().values.tolist()
# danhao_unique_id_test_data = test_data['患者id'].values.tolist()
# print("测试集人数以及数据量",len(danhao_unique_id_test_data),len(test_data))
# train_data = df[df['患者id'].isin(train_idx)]
# # danhao_unique_id_train_data = train_data['患者id'].drop_duplicates().values.tolist()
# danhao_unique_id_train_data = train_data['患者id'].values.tolist()
# print("训练集人数以及数据量",len(danhao_unique_id_train_data),len(train_data))

# print('0:{} 1:{}'.format(df[label].values.tolist().count(0), df[label].values.tolist().count(1)))
# # X_train, X_test, y_train, y_test = train_test_split(list_1, list_2, test_size=0.2, random_state=42)
train_data,test_data = train_test_split(df, test_size=0.2, random_state=42)
print("内部训练img:{}, 内部测试img:{}".format(len(train_data),len(test_data)))
X_train, y_train = train_data['path_one'].values.tolist(),train_data[label].values.tolist()
X_test, y_test = test_data['path_one'].values.tolist(),test_data[label].values.tolist()
print('train num:{} 0:{} 1:{}'.format(len(y_train), y_train.count(0), y_train.count(1)))
print('test num:{} 0:{} 1:{}'.format(len(y_test), y_test.count(0), y_test.count(1)))

# def list_func(X, y):
#     img_list = []
#     label_list = []
#     for i in range(len(X)):
#         for j in range(len(X[i])):
#             label_list.append(y[i])
#         img_list.extend(X[i])
#     return img_list, label_list
img_list_train, label_list_train = X_train, y_train
img_list_test, label_list_test = X_test, y_test
# img_list_train, label_list_train = list_func(X_train, y_train)
# img_list_test, label_list_test = list_func(X_test, y_test)

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
                                            batch_size=args.batch_size,
                                            shuffle=True
                                               )

test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=args.batch_size
                                              )

model_name = args.arc
if model_name == 'resnet101':
    net = models.resnet101(num_classes = 2)
    pretrained_dict = torch.load('/mnt/workdir/fengwei/ROP/pretrain/resnet101-5d3b4d8f.pth')
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'fc' not in k)}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    # net.load_state_dict(torch.load('/mnt/workdir/xiapeng/cognitive/model/clean/resnet101_valid_chd_mmse.pth'))

elif model_name == 'resnet18':# resnet18: (fc): Linear(in_features=512, out_features=2, bias=True)
    net = models.resnet18(num_classes = 2)
    pretrained_dict = torch.load('/mnt/workdir/fengwei/ROP/pretrain/resnet18-5c106cde.pth')
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'fc' not in k)}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    # net.load_state_dict(torch.load('/mnt/workdir/xiapeng/cognitive/model/clean/resnet18_valid_chd_mmse.pth'))

elif model_name == 'resnet50': # (fc): Linear(in_features=2048, out_features=2, bias=True)
    net = models.resnet50(num_classes = 2)
    pretrained_dict = torch.load('/mnt/workdir/fengwei/ROP/pretrain/resnet50-19c8e357.pth')
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'fc' not in k)}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    # net.load_state_dict(torch.load('/mnt/workdir/xiapeng/cognitive/model/clean/resnet50_valid_chd_mmse.pth'))

elif model_name == 'inceptionv3': # (fc): Linear(in_features=2048, out_features=2, bias=True)
    net = models.inception_v3(num_classes = 2, aux_logits=False)
    pretrained_dict = torch.load('/mnt/workdir/fengwei/ROP/pretrain/inception_v3_google-1a9a5a14.pth')
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'fc' not in k)}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    # net.load_state_dict(torch.load('/mnt/workdir/xiapeng/cognitive/model/clean/inceptionv3_valid_chd_mmse.pth'))

elif model_name == 'vgg16': # classifier[6]: Linear(in_features=4096, out_features=1000, bias=True)
    net = models.vgg16(pretrained=False)
    pretrained_dict = torch.load('/mnt/workdir/fengwei/ROP/pretrain/vgg16-397923af.pth')
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'fc' not in k)}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    net.classifier[6] = nn.Linear(4096,2)
    # net.load_state_dict(torch.load('/mnt/workdir/xiapeng/cognitive/model/clean/vgg16_valid_chd_mmse.pth'))

elif model_name == 'densenet121': # (classifier): Linear(in_features=1024, out_features=1000, bias=True)
    net = models.densenet121()
    pretrained_dict = torch.load('/mnt/workdir/fengwei/ROP/pretrain/densenet121-a639ec97.pth')
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'fc' not in k)}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    in_features = net.classifier.in_features
    net.classifier = nn.Linear(in_features,2)
    # net.load_state_dict(torch.load('/mnt/workdir/xiapeng/cognitive/model/clean/densenet121_valid_chd_mmse.pth'))
elif model_name == "efficient":
    model_config = [[3, 3, 1, 1, 24, 24, 0, 0],
                    [5, 3, 2, 4, 24, 48, 0, 0],
                    [5, 3, 2, 4, 48, 80, 0, 0],
                    [7, 3, 2, 4, 80, 160, 1, 0.25],
                    [14, 3, 1, 6, 160, 176, 1, 0.25],
                    [18, 3, 2, 6, 176, 304, 1, 0.25],
                    [5, 3, 1, 6, 304, 512, 1, 0.25]]

    net = EfficientNetV2(model_cnf=model_config,
                           num_mainclass=2,
                           dropout_rate=0.3)

net.cuda()
net = torch.nn.DataParallel(net)
loss_function = nn.CrossEntropyLoss()
# loss_function = FocalLoss(gamma=2, alpha=0.25)

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
save_path = './model/yingwen/'+label+'/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

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
        
        test_epoch_loss = running_loss / test_steps
        test_losses.append(test_epoch_loss)
        test_aucs.append(roc_auc_score(gt_labels,predict_p))
        current_auc = roc_auc_score(gt_labels,predict_p)
        exp_lr_scheduler.step(current_auc)
        if current_auc > best_auc:
            print("====================================")
            print("in best model, validating on testing subset")
            best_auc = current_auc
            print("test epoch[{}/{}] loss:{:.3f} test AUC:{}".format(epoch + 1, epochs, loss, roc_auc_score(gt_labels, predict_p)))
            torch.save(net.state_dict(), save_path+model_name+'_chd_2409.pth')
        else:
            cnt += 1

print('Finished Training')
