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
from data_loader import *
from draw import *
from metrics_full import *
import copy
warnings.filterwarnings('ignore')
import argparse
import ast
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as st
from sklearn import metrics

class DelongTest():
    def __init__(self,preds1,preds2,label,threshold=0.05):
        '''
        preds1:the output of model1
        preds2:the output of model2
        label :the actual label
        '''
        self._preds1=preds1
        self._preds2=preds2
        self._label=label
        self.threshold=threshold
        self._show_result()

    def _auc(self,X, Y)->float:
        return 1/(len(X)*len(Y)) * sum([self._kernel(x, y) for x in X for y in Y])

    def _kernel(self,X, Y)->float:
        '''
        Mann-Whitney statistic
        '''
        return .5 if Y==X else int(Y < X)

    def _structural_components(self,X, Y)->list:
        V10 = [1/len(Y) * sum([self._kernel(x, y) for y in Y]) for x in X]
        V01 = [1/len(X) * sum([self._kernel(x, y) for x in X]) for y in Y]
        return V10, V01

    def _get_S_entry(self,V_A, V_B, auc_A, auc_B)->float:
        return 1/(len(V_A)-1) * sum([(a-auc_A)*(b-auc_B) for a,b in zip(V_A, V_B)])
    
    def _z_score(self,var_A, var_B, covar_AB, auc_A, auc_B):
        return (auc_A - auc_B)/((var_A + var_B - 2*covar_AB )**(.5)+ 1e-8)

    def _group_preds_by_label(self,preds, actual)->list:
        X = [p for (p, a) in zip(preds, actual) if a]
        Y = [p for (p, a) in zip(preds, actual) if not a]
        return X, Y

    def _compute_z_p(self):
        X_A, Y_A = self._group_preds_by_label(self._preds1, self._label)
        X_B, Y_B = self._group_preds_by_label(self._preds2, self._label)

        V_A10, V_A01 = self._structural_components(X_A, Y_A)
        V_B10, V_B01 = self._structural_components(X_B, Y_B)

        auc_A = self._auc(X_A, Y_A)
        auc_B = self._auc(X_B, Y_B)

        # Compute entries of covariance matrix S (covar_AB = covar_BA)
        var_A = (self._get_S_entry(V_A10, V_A10, auc_A, auc_A) * 1/len(V_A10)+ self._get_S_entry(V_A01, V_A01, auc_A, auc_A) * 1/len(V_A01))
        var_B = (self._get_S_entry(V_B10, V_B10, auc_B, auc_B) * 1/len(V_B10)+ self._get_S_entry(V_B01, V_B01, auc_B, auc_B) * 1/len(V_B01))
        covar_AB = (self._get_S_entry(V_A10, V_B10, auc_A, auc_B) * 1/len(V_A10)+ self._get_S_entry(V_A01, V_B01, auc_A, auc_B) * 1/len(V_A01))

        # Two tailed test
        z = self._z_score(var_A, var_B, covar_AB, auc_A, auc_B)
        p = st.norm.sf(abs(z))*2

        return z,p

    def _show_result(self):
        z,p=self._compute_z_p()
        print(f"z score = {z:.5f};\np value = {p:.8f};")
        if p < self.threshold :print("There is a significant difference")
        else:        print("There is NO significant difference")




# os.environ["CUDA_VISIBLE_DEVICES"] = '3'
train_arg_parser = argparse.ArgumentParser(description="parser")
train_arg_parser.add_argument('--arc', type=str,
                    default='resnet50', help='select one model')
train_arg_parser.add_argument('--label', type=str,
                    default='mmse', help='select one model')
train_arg_parser.add_argument('--cohort', type=str,
                    default='internal', help='select one model')
train_arg_parser.add_argument('--batch_size', default=128, type=int)
args = train_arg_parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

torch.manual_seed(42)

df_internal = pd.read_csv('/mnt/workdir/fengwei/xiapeng/SOP/anzhen_chd_cognitive/datasets/冠心病患者认知评测记录_内部_截至23年7月.csv', encoding='gbk')
df_external = pd.read_csv('/mnt/workdir/fengwei/xiapeng/SOP/anzhen_chd_cognitive/datasets/冠心病患者认知评测记录_外部_截至23年7月.csv', encoding='gbk')

df_all = pd.concat([df_internal,df_external])
df_all["测评日期"] = pd.to_datetime(df_all["测评日期"])
df_train = df_all[df_all['测评日期']<=pd.datetime(2023,5,31)]
df_train = df_train.reset_index(drop=True) 
df_test = df_all[df_all['测评日期']>pd.datetime(2023,5,31)]
# df_test = df_test[df_test['测评日期']<pd.datetime(2023,6,30)]
df_test = df_test.reset_index(drop=True) 
print("总数量:{}, 内部训练:{}, 外部测试:{}".format(len(df_all),len(df_train),len(df_test)))

label = args.label
# df_train.dropna(subset=[label], inplace=True)
# df_train = df_train.reset_index(drop=True) 

   
# internal_train, internal_test = train_test_split(df_train, test_size=0.2, random_state=42)
# print(len(internal_train),len(internal_test))


def func_path(x):
    x = x.replace('img/','/mnt/workdir/fengwei/xiapeng/cognitive_guanxin/img_resize/').replace('ikang/','/mnt/workdir/fengwei/xiapeng/cognitive_guanxin/img_resize/')

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

if args.cohort == "external":
    df = copy.copy(df_test)

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


    internal_test = copy.copy(df)
    internal_test = internal_test[:400]
    
    danhao_unique_id_train_data = internal_test['患者id'].drop_duplicates().values.tolist()
  
    print("训练集人数以及数据量",len(danhao_unique_id_train_data),len(internal_test))
    print("外部测试img:{}".format(len(internal_test)))
elif args.cohort == "internal":

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


    internal_train, internal_test = train_test_split(df, test_size=0.2, random_state=42)

    print("内部训练img:{}, 内部测试img:{}".format(len(internal_train),len(internal_test)))
    
# #     if label == 'mmse':
# #         internal_train.to_csv('/mnt/workdir/fengwei/xiapeng/SOP/anzhen_chd_cognitive/datasets/冠心病患者认知评测记录MMSE_内部训练_截至23年7月.csv', index=False, encoding='gbk')
# #         internal_test.to_csv('/mnt/workdir/fengwei/xiapeng/SOP/anzhen_chd_cognitive/datasets/冠心病患者认知评测记录MMSE_内部测试_截至23年7月.csv', index=False, encoding='gbk')
# #     elif label == 'moca':
# #         internal_train.to_csv('/mnt/workdir/fengwei/xiapeng/SOP/anzhen_chd_cognitive/datasets/冠心病患者认知评测记录MOCA_内部训练_截至23年7月.csv', index=False, encoding='gbk')
# #         internal_test.to_csv('/mnt/workdir/fengwei/xiapeng/SOP/anzhen_chd_cognitive/datasets/冠心病患者认知评测记录MOCA_内部测试_截至23年7月.csv', index=False, encoding='gbk')
# # #     elif label == 'psqi':
# # #         internal_train.to_csv('/mnt/workdir/fengwei/xiapeng/SOP/anzhen_chd_cognitive/datasets/冠心病患者认知评测记录PSQI_内部训练_截至23年7月.csv', index=False, encoding='gbk')
# # #         internal_test.to_csv('/mnt/workdir/fengwei/xiapeng/SOP/anzhen_chd_cognitive/datasets/冠心病患者认知评测记录PSQI_内部测试_截至23年7月.csv', index=False, encoding='gbk')
# if label == 'mmse':
#     internal_test.to_csv('/mnt/workdir/fengwei/xiapeng/SOP/anzhen_chd_cognitive/datasets/冠心病患者认知评测记录MMSE_外部测试_截至23年7月.csv', index=False, encoding='gbk')
# elif label == 'moca':
#     internal_test.to_csv('/mnt/workdir/fengwei/xiapeng/SOP/anzhen_chd_cognitive/datasets/冠心病患者认知评测记录MOCA_外部测试_截至23年7月.csv', index=False, encoding='gbk')

# +



test_data = copy.copy(internal_test)
X_test, y_test = test_data['path_one'].values.tolist(),test_data[label].values.tolist()
print('test num:{} 0:{} 1:{}'.format(len(y_test), y_test.count(0), y_test.count(1)))
img_list_test, label_list_test = X_test, y_test


# X_test, y_test = list_1, list_2
# print('test num:{} 0:{} 1:{}'.format(len(y_test), y_test.count(0), y_test.count(1)))
# img_list_test, label_list_test = list_func(X_test, y_test)
# # external
# # img_list, label_list = list_func(list_1, list_2)
# -

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

test_dataset = customData(img_list_test, label_list_test, '', dataset = 'test', data_transforms=data_transforms)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=args.batch_size)

save_path = './model/yingwen/'+label+'/'+"resnet101"+'_chd_2409.pth'
net1 = models.resnet101(num_classes = 2)
net1.cuda()
net1 = torch.nn.DataParallel(net1)
net1.load_state_dict(torch.load(save_path))
# net1.load_state_dict(torch.load('/mnt/workdir/xiapeng/cognitive/model/clean/resnet101_all_dir_mmse.pth'))
# net1.load_state_dict(torch.load('./model/'+label+'/resnet101_clean_valid_chd_mmse.pth'))

save_path = './model/yingwen/'+label+'/'+"resnet50"+'_chd_2409.pth'
net2 = models.resnet50(num_classes = 2)
net2.cuda()
net2 = torch.nn.DataParallel(net2)
net2.load_state_dict(torch.load(save_path))
# net2.load_state_dict(torch.load('/mnt/workdir/xiapeng/cognitive/model/clean/resnet50_all_dir_mmse.pth'))
# net2.load_state_dict(torch.load('./model/'+label+'/resnet50_clean_valid_chd_mmse.pth'))

save_path = './model/yingwen/'+label+'/'+"inceptionv3"+'_chd_2409.pth'
net3 = models.inception_v3(num_classes = 2, aux_logits=False)
net3.cuda()
net3 = torch.nn.DataParallel(net3)
net3.load_state_dict(torch.load(save_path))
# net3.load_state_dict(torch.load('/mnt/workdir/xiapeng/cognitive/model/clean/inceptionv3_all_dir_mmse.pth'))
# net3.load_state_dict(torch.load('./model/'+label+'/inceptionv3_clean_valid_chd_mmse.pth'))

# +
# save_path = './model/'+label+'/'+"densenet121"+'_chd_2409.pth'
# net4 = models.densenet121(num_classes = 2)
# net4.cuda()
# net4 = torch.nn.DataParallel(net4)
# net4.load_state_dict(torch.load(save_path))
# -


save_path = './model/yingwen/'+label+'/'+"resnet18"+'_chd_2409.pth'
net6 = models.resnet18(num_classes = 2)
net6.cuda()
net6 = torch.nn.DataParallel(net6)
net6.load_state_dict(torch.load(save_path))
# net6.load_state_dict(torch.load('/mnt/workdir/xiapeng/cognitive/model/clean/resnet18_all_dir_mmse.pth'))
# net6.load_state_dict(torch.load('./model/'+label+'/resnet18_clean_valid_chd_mmse.pth'))

# +
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
#     predict_4 = []
#     predict_5 = []
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
# FPR_4, TPR_4, th4 = roc_curve(gt_labels, predict_4, pos_label = 1)
# CI_4 = AUC_CI(roc_auc_score(gt_labels, predict_4), gt_labels, 0.05)
# FPR_5, TPR_5, th5 = roc_curve(gt_labels, predict_5, pos_label = 1)
# CI_5 = AUC_CI(roc_auc_score(gt_labels, predict_5), gt_labels, 0.05)

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
plt.savefig('./pic/yingwen/'+label+'/AUC_{}_{}.pdf'.format(label,args.cohort), bbox_inches = 'tight') # , bbox_inches='tight'

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
# # Model A (random) vs. "good" model B
# preds_A = np.array([.5, .5, .5, .5, .5, .5, .5, .5, .5, .5])
# preds_B = np.array([.2, .5, .1, .4, .9, .8, .7, .5, .9, .8])
# actual=    np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])
DelongTest(predict_1,predict_p,gt_labels)
DelongTest(predict_2,predict_p,gt_labels)
DelongTest(predict_3,predict_p,gt_labels)
DelongTest(predict_6,predict_p,gt_labels)
