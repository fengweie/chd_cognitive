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
# from data_loader import *
import warnings
from draw import *
from metrics_full import *

warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

torch.manual_seed(42)

df_internal = pd.read_csv('/mnt/workdir/fengwei/xiapeng/SOP/anzhen_chd_cognitive/datasets/冠心病患者认知评测记录_内部_截至23年7月.csv', encoding='gbk')
df_external = pd.read_csv('/mnt/workdir/fengwei/xiapeng/SOP/anzhen_chd_cognitive/datasets/冠心病患者认知评测记录_外部_截至23年7月.csv', encoding='gbk')

df_all = pd.concat([df_internal,df_external])
df_all["测评日期"] = pd.to_datetime(df_all["测评日期"])
df_train = df_all[df_all['测评日期']<=pd.datetime(2022,10,1)]
df_train = df_train.reset_index(drop=True) 
df_test = df_all[df_all['测评日期']>pd.datetime(2022,10,1)]
df_test = df_test.reset_index(drop=True) 
print(len(df_all),len(df_train),len(df_test))

label = 'mmse'
df_train.dropna(subset=[label], inplace=True)
df_train = df_train.reset_index(drop=True) 
df_train[label] = df_train[label].apply(lambda x: x if len(str(x).split(',')) == 1 else max(str(x).split(',')))
df_train[label] = df_train[label].apply(lambda x: x if len(str(x).split(' ')) == 1 else max(str(x).split(' ')))
   
train, test = train_test_split(df_train, test_size=0.2, random_state=42)
print(len(train),len(test))
if label == 'mmse':
    train.to_csv('/mnt/workdir/fengwei/xiapeng/SOP/anzhen_chd_cognitive/datasets/冠心病患者认知评测记录MMSE_内部训练_截至23年7月.csv', index=False, encoding='gbk')
    test.to_csv('/mnt/workdir/fengwei/xiapeng/SOP/anzhen_chd_cognitive/datasets/冠心病患者认知评测记录MMSE_内部测试_截至23年7月.csv', index=False, encoding='gbk')
elif label == 'moca':
    train.to_csv('/mnt/workdir/fengwei/xiapeng/SOP/anzhen_chd_cognitive/datasets/冠心病患者认知评测记录MOCA_内部训练_截至23年7月.csv', index=False, encoding='gbk')
    test.to_csv('/mnt/workdir/fengwei/xiapeng/SOP/anzhen_chd_cognitive/datasets/冠心病患者认知评测记录MOCA_内部测试_截至23年7月.csv', index=False, encoding='gbk')
elif label == 'psqi':
    train.to_csv('/mnt/workdir/fengwei/xiapeng/SOP/anzhen_chd_cognitive/datasets/冠心病患者认知评测记录PSQI_内部训练_截至23年7月.csv', index=False, encoding='gbk')
    test.to_csv('/mnt/workdir/fengwei/xiapeng/SOP/anzhen_chd_cognitive/datasets/冠心病患者认知评测记录PSQI_内部测试_截至23年7月.csv', index=False, encoding='gbk')

df_test.to_csv('/mnt/workdir/fengwei/xiapeng/SOP/anzhen_chd_cognitive/datasets/冠心病患者认知评测记录_外部测试_截至23年7月.csv', index=False, encoding='gbk')
# -

img_list = []
label_list = []

label = 'mmse'
df.dropna(subset=[label], inplace=True)
df[label] = df[label].apply(lambda x: x if len(str(x).split(',')) == 1 else max(str(x).split(',')))
df[label] = df[label].apply(lambda x: x if len(str(x).split(' ')) == 1 else max(str(x).split(' ')))
if label == 'mmse':
    df[label] = df[label].apply(lambda x: 1 if float(x) < 27 else 0)
elif label == 'moca':
    df[label] = df[label].apply(lambda x: 1 if float(x) < 26 else 0)
elif label == 'psqi':
    df[label] = df[label].apply(lambda x: 1 if float(x) >= 7 else 0)
def func_path(x):
    if type(x) == list:
        x = [i.replace('ikang/','img/') for i in x]
        x = ['img/'+i for i in x if not i.startswith('img')]
    elif type(x) == str:
        x = x.replace('ikang/','img/').replace('[','').replace(']','').replace("'","").replace(' ','').split(',')
        # x = ['img/'+i for i in x if not i.startswith('img')]
    return x
df['path'] = df['path'].apply(func_path)
list_1 = df['path'].values.tolist()
list_2 = df[label].values.tolist()
print('{} 0:{} 1:{}'.format(label, df[label].values.tolist().count(0), df[label].values.tolist().count(1)))

def list_func(X, y):
    img_list = []
    label_list = []
    for i in range(len(X)):
        for j in range(len(X[i])):
            label_list.append(y[i])
        img_list.extend(X[i])
    return img_list, label_list

# internal
X_train, X_test, y_train, y_test = train_test_split(list_1, list_2, test_size=0.2, random_state=42)
print('train num:{} 0:{} 1:{}'.format(len(y_train), y_train.count(0), y_train.count(1)))
print('test num:{} 0:{} 1:{}'.format(len(y_test), y_test.count(0), y_test.count(1)))
img_list_test, label_list_test = list_func(X_test, y_test)
external
img_list, label_list = list_func(list_1, list_2)

data_transforms = {
    'train': transforms.Compose([
                                 transforms.Resize((512,512)),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
     'test': transforms.Compose([
                               transforms.Resize((512,512)),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
}

test_dataset = customData(img_list_test, label_list_test, './', dataset = 'test', data_transforms=data_transforms)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=256)

net1 = models.resnet101(num_classes = 2)
net1.cuda()
net1.load_state_dict(torch.load('./model/'+label+'/resnet101_chd_07.pth'))
# net1.load_state_dict(torch.load('/mnt/workdir/xiapeng/cognitive/model/clean/resnet101_all_dir_mmse.pth'))
# net1.load_state_dict(torch.load('./model/'+label+'/resnet101_clean_valid_chd_mmse.pth'))

net2 = models.resnet50(num_classes = 2)
net2.cuda()
net2.load_state_dict(torch.load('./model/'+label+'/resnet50_chd_07.pth'))
# net2.load_state_dict(torch.load('/mnt/workdir/xiapeng/cognitive/model/clean/resnet50_all_dir_mmse.pth'))
# net2.load_state_dict(torch.load('./model/'+label+'/resnet50_clean_valid_chd_mmse.pth'))

net3 = models.inception_v3(num_classes = 2, aux_logits=False)
net3.cuda()
net3.load_state_dict(torch.load('./model/'+label+'/inceptionv3_chd_07.pth'))
# net3.load_state_dict(torch.load('/mnt/workdir/xiapeng/cognitive/model/clean/inceptionv3_all_dir_mmse.pth'))
# net3.load_state_dict(torch.load('./model/'+label+'/inceptionv3_clean_valid_chd_mmse.pth'))

net6 = models.resnet18(num_classes = 2)
net6.cuda()
net6.load_state_dict(torch.load('./model/'+label+'/resnet18_chd_07.pth'))
# net6.load_state_dict(torch.load('/mnt/workdir/xiapeng/cognitive/model/clean/resnet18_all_dir_mmse.pth'))
# net6.load_state_dict(torch.load('./model/'+label+'/resnet18_clean_valid_chd_mmse.pth'))

net1.eval()
net2.eval()
net3.eval()
# net4.eval()
# net5.eval()
net6.eval()

with torch.no_grad():
    gt_labels = []
    predict_p = []
    predict_1 = []
    predict_2 = []
    predict_3 = []
    predict_4 = []
    predict_5 = []
    predict_6 = []
    for data in tqdm.tqdm(test_loader):
        img_name, inputs, labels = data
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        outputs_1 = net1(inputs)
        outputs_2 = net2(inputs)
        outputs_3 = net3(inputs)
        # outputs_4 = net4(inputs)
        # outputs_5 = net5(inputs)
        outputs_6 = net6(inputs)
        gt_labels.extend(labels.cpu().numpy().tolist())
        predict_1.extend(nn.functional.softmax(outputs_1).cpu().detach().numpy()[:,1])
        predict_2.extend(nn.functional.softmax(outputs_2).cpu().detach().numpy()[:,1])
        predict_3.extend(nn.functional.softmax(outputs_3).cpu().detach().numpy()[:,1])
        # predict_4.extend(nn.functional.softmax(outputs_4).cpu().detach().numpy()[:,1])
        # predict_5.extend(nn.functional.softmax(outputs_5).cpu().detach().numpy()[:,1])
        predict_6.extend(nn.functional.softmax(outputs_6).cpu().detach().numpy()[:,1])
    # predict_p.extend((np.array(predict_1)+np.array(predict_2)+np.array(predict_3)+np.array(predict_6)+np.array(predict_5))/5)
    predict_p.extend((np.array(predict_1)+np.array(predict_2)+np.array(predict_3)+np.array(predict_6))/4)
    # predict_p.extend((np.array(predict_1)+np.array(predict_2)+np.array(predict_3)+np.array(predict_4)+np.array(predict_5)+np.array(predict_6))/6)

FPR_1, TPR_1, th1 = roc_curve(gt_labels, predict_1, pos_label = 1)
CI_1 = AUC_CI(roc_auc_score(gt_labels, predict_1), gt_labels, 0.05)

FPR_2, TPR_2, th2 = roc_curve(gt_labels, predict_2, pos_label = 1)
CI_2 = AUC_CI(roc_auc_score(gt_labels, predict_2), gt_labels, 0.05)

FPR_3, TPR_3, th3 = roc_curve(gt_labels, predict_3, pos_label = 1)
CI_3 = AUC_CI(roc_auc_score(gt_labels, predict_3), gt_labels, 0.05)
FPR_4, TPR_4, th4 = roc_curve(gt_labels, predict_4, pos_label = 1)
CI_4 = AUC_CI(roc_auc_score(gt_labels, predict_4), gt_labels, 0.05)
FPR_5, TPR_5, th5 = roc_curve(gt_labels, predict_5, pos_label = 1)
CI_5 = AUC_CI(roc_auc_score(gt_labels, predict_5), gt_labels, 0.05)

FPR_6, TPR_6, th6 = roc_curve(gt_labels, predict_6, pos_label = 1)
CI_6 = AUC_CI(roc_auc_score(gt_labels, predict_6), gt_labels, 0.05)

FPR_p, TPR_p, thp = roc_curve(gt_labels, predict_p, pos_label = 1)
CI_p = AUC_CI(roc_auc_score(gt_labels, predict_p), gt_labels, 0.05)

pre1, sen1, spe1, acc1, f11, ppv1, npv1 = cal_metrics(gt_labels, predict_1, find_optimal_cutoff(FPR_1, TPR_1, th1))
pre2, sen2, spe2, acc2, f12, ppv2, npv2 = cal_metrics(gt_labels, predict_2, find_optimal_cutoff(FPR_2, TPR_2, th2))
pre3, sen3, spe3, acc3, f13, ppv3, npv3 = cal_metrics(gt_labels, predict_3, find_optimal_cutoff(FPR_3, TPR_3, th3))
# pre4, sen4, spe4, acc4, f14, ppv4, npv4 = cal_metrics(gt_labels, predict_4, find_optimal_cutoff(FPR_1, TPR_1, th4))
# pre5, sen5, spe5, acc5, f15, ppv5, npv5 = cal_metrics(gt_labels, predict_5, find_optimal_cutoff(FPR_5, TPR_5, th5))
pre6, sen6, spe6, acc6, f16, ppv6, npv6 = cal_metrics(gt_labels, predict_6, find_optimal_cutoff(FPR_6, TPR_6, th6))
prep, senp, spep, accp, f1p, ppvp, npvp = cal_metrics(gt_labels, predict_p, find_optimal_cutoff(FPR_p, TPR_p, thp))

plt.style.use('ggplot')
fig, ax = plt.subplots()
ax = plot_AUC(ax, FPR_6, TPR_6, roc_auc_score(gt_labels, predict_6), CI_6, label = 'ResNet-18')
ax = plot_AUC(ax, FPR_2, TPR_2, roc_auc_score(gt_labels, predict_2), CI_2, label = 'ResNet-50')
ax = plot_AUC(ax, FPR_1, TPR_1, roc_auc_score(gt_labels, predict_1), CI_1, label = 'ResNet-101')
ax = plot_AUC(ax, FPR_3, TPR_3, roc_auc_score(gt_labels, predict_3), CI_3, label = 'Inception-v3')
# ax = plot_AUC(ax, FPR_4, TPR_4, roc_auc_score(gt_labels, predict_4), CI_4, label = 'VGG-16')
# ax = plot_AUC(ax, FPR_5, TPR_5, roc_auc_score(gt_labels, predict_5), CI_5, label = 'DenseNet-121')
ax = plot_AUC(ax, FPR_p, TPR_p, roc_auc_score(gt_labels, predict_p), CI_p, label = 'Ensemble')
ax.plot((0, 1), (0, 1), ':', color = 'grey')
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.01, 1.01)
ax.set_aspect('equal')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend()
plt.legend(loc='lower right',fontsize='small',framealpha=0.5)
plt.savefig('./pic/'+label+'/internal_2307.svg', format="svg") # , bbox_inches='tight'

print("resnet101: test AUC:{:.4f} Precision:{:.4f} Recall:{:.4f} Specificity:{:.4f} ACC:{:.4f} F1:{:.4f} PPV:{:.4f} NPV:{:.4f}".format(roc_auc_score(gt_labels, predict_1), pre1, sen1, spe1, acc1, f11, ppv1, npv1))
print("resnet50: test AUC:{:.4f} Precision:{:.4f} Recall:{:.4f} Specificity:{:.4f} ACC:{:.4f} F1:{:.4f} PPV:{:.4f} NPV:{:.4f}".format(roc_auc_score(gt_labels, predict_2), pre2, sen2, spe2, acc2, f12, ppv2, npv2))
print("resnet18: test AUC:{:.4f} Precision:{:.4f} Recall:{:.4f} Specificity:{:.4f} ACC:{:.4f} F1:{:.4f} PPV:{:.4f} NPV:{:.4f}".format(roc_auc_score(gt_labels, predict_6), pre6, sen6, spe6, acc6, f16, ppv6, npv6))
print("inceptionv3: test AUC:{:.4f} Precision:{:.4f} Recall:{:.4f} Specificity:{:.4f} ACC:{:.4f} F1:{:.4f} PPV:{:.4f} NPV:{:.4f}".format(roc_auc_score(gt_labels, predict_3), pre3, sen3, spe3, acc3, f13, ppv3, npv3))
# print("vgg16: test AUC:{:.4f} Precision:{:.4f} Recall:{:.4f} Specificity:{:.4f} ACC:{:.4f} F1:{:.4f} PPV:{:.4f} NPV:{:.4f}".format(roc_auc_score(gt_labels, predict_4), pre4, sen4, spe4, acc4, f14, ppv4, npv4))
# print("densenet121: test AUC:{:.4f} Precision:{:.4f} Recall:{:.4f} Specificity:{:.4f} ACC:{:.4f} F1:{:.4f} PPV:{:.4f} NPV:{:.4f}".format(roc_auc_score(gt_labels, predict_5), pre5, sen5, spe5, acc5, f15, ppv5, npv5))
print("model ensemble: test AUC:{:.4f} Precision:{:.4f} Recall:{:.4f} Specificity:{:.4f} ACC:{:.4f} F1:{:.4f} PPV:{:.4f} NPV:{:.4f}".format(roc_auc_score(gt_labels, predict_p), prep, senp, spep, accp, f1p, ppvp, npvp))

print("resnet101: AUC:{} Recall:{} Specificity:{} ACC:{}".format(AUC_CI(roc_auc_score(gt_labels, predict_1), gt_labels, alpha = 0.05), AUC_CI(sen1, gt_labels, alpha = 0.05), AUC_CI(spe1, gt_labels, alpha = 0.05), AUC_CI(acc1, gt_labels, alpha = 0.05)))
print("resnet50: AUC:{} Recall:{} Specificity:{} ACC:{}".format(AUC_CI(roc_auc_score(gt_labels, predict_2), gt_labels, alpha = 0.05), AUC_CI(sen2, gt_labels, alpha = 0.05), AUC_CI(spe2, gt_labels, alpha = 0.05), AUC_CI(acc2, gt_labels, alpha = 0.05)))
print("resnet18: AUC:{} Recall:{} Specificity:{} ACC:{}".format(AUC_CI(roc_auc_score(gt_labels, predict_6), gt_labels, alpha = 0.05), AUC_CI(sen6, gt_labels, alpha = 0.05), AUC_CI(spe6, gt_labels, alpha = 0.05), AUC_CI(acc6, gt_labels, alpha = 0.05)))
print("inceptionv3: AUC:{} Recall:{} Specificity:{} ACC:{}".format(AUC_CI(roc_auc_score(gt_labels, predict_3), gt_labels, alpha = 0.05), AUC_CI(sen3, gt_labels, alpha = 0.05), AUC_CI(spe3, gt_labels, alpha = 0.05), AUC_CI(acc3, gt_labels, alpha = 0.05)))
print("model ensemble: AUC:{} Recall:{} Specificity:{} ACC:{}".format(AUC_CI(roc_auc_score(gt_labels, predict_p), gt_labels, alpha = 0.05), AUC_CI(senp, gt_labels, alpha = 0.05), AUC_CI(spep, gt_labels, alpha = 0.05), AUC_CI(accp, gt_labels, alpha = 0.05)))
