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
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import numpy as np
import os
import tqdm
from torch.autograd import Variable
from PIL import Image
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import warnings

# -

warnings.filterwarnings('ignore')

# +
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# -

torch.manual_seed(42)

# +

df_clean = pd.read_excel('/mnt/workdir/fengwei/yifuying/成都爱康/data_chengdu_all_ikang.xlsx')
# -

img_list = []


# +

def func_path(x):
    if type(x) == list:
        x = [i.replace('ikang/','/mnt/workdir/fengwei/yifuying/成都爱康/img/') for i in x]
        x = ['img/'+i for i in x if not i.startswith('img')]
    elif type(x) == str:
        x = x.replace('ikang/','/mnt/workdir/fengwei/yifuying/成都爱康/img/')
        # x = ['img/'+i for i in x if not i.startswith('img')]
    return x
df_clean['img_path'] = df_clean['img_path'].apply(func_path)
img_list_test = df_clean['img_path'].values.tolist()


# -

def list_func(X, y):
    img_list = []
    label_list = []
    for i in range(len(X)):
        for j in range(len(X[i])):
            label_list.append(y[i])
        img_list.extend(X[i])
    return img_list, label_list
print('test num:{}'.format(len(img_list_test)))

data_transforms = transforms.Compose([
                               transforms.Resize((512,512)),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


# +
net_mmse = models.resnet50(num_classes = 2)
net_mmse.cuda()
net_mmse.load_state_dict(torch.load('./model/resnet50_mmse.pth'))

net_moca = models.resnet50(num_classes = 2)
net_moca.cuda()
net_moca.load_state_dict(torch.load('./model/resnet50_moca.pth'))

net_moca_mmse = models.resnet50(num_classes = 2)
net_moca_mmse.cuda()
net_moca_mmse.load_state_dict(torch.load('./model/resnet50_vision.pth'))

df_clean["mmse"] = pd.Series(0,index=df_clean.index).astype(float)
df_clean["moca"] = pd.Series(0,index=df_clean.index).astype(float)
df_clean["moca_mmse"] = pd.Series(0,index=df_clean.index).astype(float)
for row in df_clean.index:
    img_path = df_clean['img_path'][row]
    # Image
    img_input = Image.open(img_path).convert('RGB')
    img_ori = data_transforms(img_input)
#     print(img_ori.shape)
    img_ori = img_ori.unsqueeze(0).cuda()
    with torch.no_grad():
        outputs_mmse = net_mmse(img_ori)
        outputs_moca = net_moca(img_ori)
        outputs_moca_mmse = net_moca_mmse(img_ori)
        
        predict_mmse = nn.functional.softmax(outputs_mmse).cpu().detach().numpy()[:,1]
        
        predict_moca = nn.functional.softmax(outputs_moca).cpu().detach().numpy()[:,1]
        predict_moca_mmse = nn.functional.softmax(outputs_moca_mmse).cpu().detach().numpy()[:,1]  
        print(predict_mmse,predict_moca,predict_moca_mmse)
        df_clean["mmse"][row] = predict_mmse[0]
        df_clean["moca"][row] = predict_moca[0]
        df_clean["moca_mmse"][row] = predict_moca_mmse[0]

df_clean.to_excel('data_chengdu_all_ikang_moca_mmse.xlsx', index=False)
df_clean  
# -



# +
# with torch.no_grad():
#     gt_labels = []
#     predict_p = []
#     predict_1 = []
#     predict_2 = []
#     predict_3 = []
#     predict_4 = []
#     predict_5 = []
#     predict_6 = []
#     for data in tqdm.tqdm(test_loader):
#         img_name, inputs, labels = data
#         inputs = Variable(inputs.cuda())
#         labels = Variable(labels.cuda())
#         outputs_1 = net1(inputs)
#         outputs_2 = net2(inputs)
#         outputs_3 = net3(inputs)
#         # outputs_4 = net4(inputs)
#         # outputs_5 = net5(inputs)
#         outputs_6 = net6(inputs)
#         gt_labels.extend(labels.cpu().numpy().tolist())
#         predict_1.extend(nn.functional.softmax(outputs_1).cpu().detach().numpy()[:,1])
#         predict_2.extend(nn.functional.softmax(outputs_2).cpu().detach().numpy()[:,1])
#         predict_3.extend(nn.functional.softmax(outputs_3).cpu().detach().numpy()[:,1])
#         # predict_4.extend(nn.functional.softmax(outputs_4).cpu().detach().numpy()[:,1])
#         # predict_5.extend(nn.functional.softmax(outputs_5).cpu().detach().numpy()[:,1])
#         predict_6.extend(nn.functional.softmax(outputs_6).cpu().detach().numpy()[:,1])
#     # predict_p.extend((np.array(predict_1)+np.array(predict_2)+np.array(predict_3)+np.array(predict_6)+np.array(predict_5))/5)
#     predict_p.extend((np.array(predict_1)+np.array(predict_2)+np.array(predict_3)+np.array(predict_6))/4)
#     # predict_p.extend((np.array(predict_1)+np.array(predict_2)+np.array(predict_3)+np.array(predict_4)+np.array(predict_5)+np.array(predict_6))/6)
