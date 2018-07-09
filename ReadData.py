# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 13:49:06 2018

@author: 徐嘉诚
"""

import numpy as np
import pandas as pd
import scipy.stats as sts 
from sklearn.preprocessing import MinMaxScaler


# 参数
print('start, loading...')
Th1 = 10
Th2 = 0.001

# 读取数据
print('reading data, loading...')
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
sample_submission = pd.read_csv('../data/sample_submission.csv')

# 删除低方差特征
print('eliminate var feature')
columns_name = [f for f in train.columns if f not in ['ID','target'] ]
scaler_data = MinMaxScaler().fit_transform(train[columns_name])
scaler_data = pd.DataFrame(data = scaler_data,columns = columns_name)
STD = scaler_data.std()
low_var_threshold = np.percentile(STD,Th1)
low_var_feature = [f for f in STD[STD<=low_var_threshold].index]

# 剔除训练测试集不同分布的特征
columns_name = list(set(columns_name) - set(low_var_feature))
ks_pvalue = np.zeros([len(columns_name)])
for i,cn in enumerate(columns_name):
    _,ks_pvalue[i] = sts.ks_2samp(train[cn],test[cn] )
ks_pvalue = pd.Series(ks_pvalue,index = columns_name)
same_distribute_feature = [f for f in ks_pvalue[ks_pvalue<=Th2].index]

# 数据合并
columns_name = ['ID','target']
[columns_name.append(f) for f in same_distribute_feature]
merge = pd.concat([train,test],axis = 0).reset_index(drop =True)[columns_name]
train_set = merge[merge['ID'].isin(train['ID'])].reset_index(drop=True)
test_set = merge[merge['ID'].isin(sample_submission['ID'])].reset_index(drop=True)

# 保存到csv
train_set.to_csv('../temp/train_th{0}.csv'.format(Th1+Th2) ,index=False)
test_set.to_csv('../temp/test_th{0}.csv'.format(Th1+Th2),index=False )