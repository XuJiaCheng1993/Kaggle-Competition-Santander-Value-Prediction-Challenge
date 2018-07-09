# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 11:04:06 2018

@author: 徐嘉诚
"""

import numpy as np
import pandas as pd
import scipy.stats as sts 
import lightgbm as lgb

print(' "preprocess" start...')
## 参数设置
VER = 1.0
TOP_COLUMN_NUMS = 1000
THRESHOLD_CORR = 0.9
THRESHOLD_D_VALUE = 0.1
THRESHOLD_P_VALUE = 0.01
LGB_PARAMS = dict(boosting_type='gbdt', objective='regression', metric='rmse', learning_rate=0.01, num_leaves=255,
            sub_feature=0.25, subsample=0.9, subsample_freq=1, lambda_l1=2.5, lambda_l2=5, random_state=0) 
print('version {0}'.format(VER) )

# 读取数据
print('*'*80)
print('reading data, loading...')
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
sample_submission = pd.read_csv('../data/sample_submission.csv')
columns_name = [f for f in train.columns if f not in ['ID','target'] ]
print('train set nums:{0} ,test set nums:{1} ,columns nums:{2}'.format(train.shape[0],test.shape[0],len(columns_name)))

## 删除 单值 特征
print('*'*80)
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

print('*'*80)
print('eliminate redundant columns, loading...')
columns_name, redundancy_name = eliminate_redundant_feature(train,columns_name,Threshold=THRESHOLD_CORR)
print('redundancy column nums:{0}, remaining column nums:{1}'.format(len(redundancy_name),len(columns_name)))

## 基于gbm模型挑选top特征
print('*'*80)
print('select top columns from gbm model, loading...')
Data = lgb.Dataset(train[columns_name],np.log1p(train['target']) )
gbm = lgb.train(LGB_PARAMS, Data, num_boost_round=500, verbose_eval=False)
imps = gbm.feature_importance()
imps = pd.Series(imps,index = columns_name)
imps.sort_values(ascending = False, inplace = True)
columns_name = [f for f in imps.index[:TOP_COLUMN_NUMS]]
print('top column nums:{0}'.format(TOP_COLUMN_NUMS))

### 统计特征
#print('*'*38,' #1 ','*'*38)
#print('compute statistics columns, loading...')



### 剔除 训练测试集不同分布的特征
#print('*'*80)
#print('remove different distribution columns, loading...')
#ks_value = np.zeros([TOP_COLUMN_NUMS,2])
#for i,cn in enumerate(columns_name):
#    ks_value[i,0],ks_value[i,1] = sts.ks_2samp( train[cn].values,test[cn].values )
#ks_value = pd.DataFrame(ks_value,  columns = ['d_value','p_value'],index = columns_name)
#different_column_name = [f for f in ks_value.index if ks_value.loc[f,'d_value'] > THRESHOLD_D_VALUE and  ks_value.loc[f,'p_value'] <= THRESHOLD_P_VALUE ]
#columns_name = list(set(columns_name)-set(different_column_name) )
#print('different distribution column nums:{0}, same distribution column nums:{1}'.format(len(different_column_name),len(columns_name)))

## 保存数据
print('*'*80)
print('save result to files, loading...')
feature_name = ['ID','target']
[feature_name.append(f) for f in columns_name]
merge = pd.concat([train,test],axis = 0).reset_index(drop =True)[feature_name]
train_set = merge[merge['ID'].isin(train['ID'])].reset_index(drop=True)
test_set = merge[merge['ID'].isin(sample_submission['ID'])].reset_index(drop=True)
train_set.to_csv('../temp/train_version_{0}.csv'.format(VER) ,index=False)
test_set.to_csv('../temp/test_version_{0}.csv'.format(VER),index=False )

print('*'*80)
print('done!')














    