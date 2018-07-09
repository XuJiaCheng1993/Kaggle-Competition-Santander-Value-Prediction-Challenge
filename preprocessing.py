# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 11:04:06 2018

@author: 徐嘉诚
"""

import numpy as np
import pandas as pd
import scipy.stats as sts 
import lightgbm as lgb
from sklearn.decomposition import PCA
from sklearn.random_projection import SparseRandomProjection

print(' "preprocess" start...')
## 参数设置
VER = 1.1
TOP_COLUMN_NUMS = 1000
THRESHOLD_CORR = 0.9
THRESHOLD_D_VALUE = 0.3
THRESHOLD_P_VALUE = 0.01
LGB_PARAMS = dict(boosting_type='gbdt', objective='regression', metric='rmse', learning_rate=0.01, num_leaves=255,
            sub_feature=0.25, subsample=0.9, subsample_freq=1, lambda_l1=2.5, lambda_l2=5, random_state=0) 
DR_RADIO = 0.8
print('version {0}'.format(VER) )

## 读取数据并
print('*'*38,' #1 ','*'*38)
print('reading data, loading...')
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
sample_submission = pd.read_csv('../data/sample_submission.csv')
columns_name = [f for f in train.columns if f not in ['ID','target'] ]
print('train set nums:{0} ,test set nums:{1} ,columns nums:{2}'.format(train.shape[0],test.shape[0],len(columns_name)))

## 数据进行log1p变换
train[columns_name] = np.log1p(train[columns_name])
train['target'] = np.log1p(train['target'])
test[columns_name] = np.log1p(test[columns_name])
merge = pd.concat([train,test],axis = 0).reset_index(drop = True)


## 删除 单值 特征
print('*'*38,' #2 ','*'*38)
print('eliminate signle-value columns, loading...')
nunique = train[columns_name].nunique()
single_value_name = [f for f in nunique[nunique==1].index]
columns_name = list(set(columns_name) - set(single_value_name) )
print('signle-value column nums:{0}, remaining column nums:{1}'.format(len(single_value_name),len(columns_name)))

## 剔除冗余特征
def eliminate_redundant_feature(df,feature_name,Threshold=0.9,feature_score=None):
	corr = df.corr().abs()
	redundancy_set = []
	total = set(feature_name)
	for i in feature_name:
		if i in redundancy_set:
			continue
		tmp = corr[i]
		tmp_name = [ f for f in tmp[tmp >= Threshold ].index ]
		if len(tmp_name) == 1:
			pass
		else:
			if feature_score is not None:
				select_feature = feature_score[tmp_name].sort_values(ascending=False).index[0]
			else:
				select_feature = i
			[redundancy_set.append(f) for f in tmp_name if f not in select_feature ]
	remaining_set = list(total-set(redundancy_set))
	return remaining_set,set(redundancy_set)

print('*'*38,' #3 ','*'*38)
print('eliminate redundant columns, loading...')
columns_name, redundancy_name = eliminate_redundant_feature(train,columns_name,Threshold=THRESHOLD_CORR)
print('redundancy column nums:{0}, remaining column nums:{1}'.format(len(redundancy_name),len(columns_name)))

## 基于gbm模型挑选top特征
print('*'*38,' #4 ','*'*38)
print('select top columns from gbm model, loading...')
Data = lgb.Dataset(train[columns_name],train['target'] )
gbm = lgb.train(LGB_PARAMS, Data, num_boost_round=500, verbose_eval=False)
imps = gbm.feature_importance()
imps = pd.Series(imps,index = columns_name)
imps.sort_values(ascending = False, inplace = True)
columns_name = [f for f in imps.index[:TOP_COLUMN_NUMS]]
print('top column nums:{0}'.format(TOP_COLUMN_NUMS))

## 统计特征
print('*'*38,' #5 ','*'*38)
print('compute statistics columns, loading...')
train_middle = train[columns_name]
wi = 1 - sts.entropy(train_middle)/np.log(train_middle.shape[0])
wi = wi/np.sum(wi)
merge_middle =merge[columns_name]
statistic_data = np.zeros([merge.shape[0],10])
statistic_data[:,0] = np.count_nonzero(merge_middle,axis = 1)
statistic_data[:,1] = np.mean(merge_middle[merge_middle>0],axis = 1)
statistic_data[:,2] = np.std(merge_middle[merge_middle>0],axis = 1 )
statistic_data[:,3] = statistic_data[:,2]/statistic_data[:,1]
statistic_data[:,4] = sts.entropy(merge_middle.T)
statistic_data[:,5] = np.median(merge_middle[merge_middle>0],axis = 1)
statistic_data[:,6] = np.min(merge_middle[merge_middle>0],axis = 1)
statistic_data[:,7] = np.max(merge_middle[merge_middle>0],axis = 1)
statistic_data[:,8] = statistic_data[:,7]-statistic_data[:,6]
statistic_data[:,9] = np.dot(merge_middle,wi)
statistic_column_name = ['NONZERONUM','MEAN','STD','VARCOEF','ENTROPY','MEDIAN','MINIMA','MAXIMA','RANGE','ENTWI']
statistic_data = pd.DataFrame(statistic_data,columns = statistic_column_name)
statistic_data.fillna(0,inplace = True)
print('statistics column nums:{0}'.format(len(statistic_column_name)))

## 降维特征
print('*'*38,' #6 ','*'*38)
print('compute dr columns, loading...')
pca = PCA().fit(train_middle)
contribute = np.cumsum(pca.explained_variance_ratio_)
dims = np.count_nonzero(contribute <= DR_RADIO)
pca_data = pca.fit_transform(merge_middle)[:,:dims] 
srp_data = SparseRandomProjection(n_components=dims,random_state=220).fit(train_middle).transform(merge_middle)
dr_data = np.hstack((pca_data,srp_data))
dr_column_name = [ i+str(j) for i in ['PCA_Dim_','SRP_Dim_'] for j in range(dims)]
dr_data = pd.DataFrame(dr_data,columns = dr_column_name)
print('dr column nums:{0}'.format(len(dr_column_name)))

## 合并数据
columns_name = list(set(columns_name) | set(statistic_column_name) | set(dr_column_name))
merge = pd.concat([merge,statistic_data,dr_data],axis = 1)
train = merge[merge['ID'].isin(train['ID'])].reset_index(drop=True)
test = merge[merge['ID'].isin(sample_submission['ID'])].reset_index(drop=True)

## 剔除 训练测试集不同分布的特征
print('*'*38,' #7 ','*'*38)
print('remove different distribution columns, loading...')
ks_value = np.zeros([len(columns_name),2])
for i,cn in enumerate(columns_name):
    ks_value[i,0],ks_value[i,1] = sts.ks_2samp( train[cn].values,test[cn].values )
ks_value = pd.DataFrame(ks_value,  columns = ['d_value','p_value'],index = columns_name)
different_column_name = [f for f in ks_value.index if ks_value.loc[f,'d_value'] > THRESHOLD_D_VALUE and  ks_value.loc[f,'p_value'] <= THRESHOLD_P_VALUE ]
columns_name = list(set(columns_name)-set(different_column_name) )
print('different distribution column nums:{0}, same distribution column nums:{1}'.format(len(different_column_name),len(columns_name)))

# 保存数据
print('*'*38,' #8 ','*'*38)
print('save result to files, loading...')
feature_name = ['ID','target']
[feature_name.append(f) for f in columns_name]
train_set = train[feature_name]
test_set = test[feature_name]
train_set.to_csv('../temp/train_version_{0}.csv'.format(VER) ,index=False)
test_set.to_csv('../temp/test_version_{0}.csv'.format(VER),index=False )

print('*'*80)
print('done!')














    