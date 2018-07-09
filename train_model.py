# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 15:03:52 2018

@author: 徐嘉诚
"""

import pandas as pd 
import lightgbm as lgb
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import scipy.stats as sts 
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def fevalfunc(pred,df):
    label = df.get_label().values.copy()
    score = np.sqrt(np.mean(np.power(pred-label,2)))
    return ('RMSLE',score,False)

def fevalfunc_forxgb(pred,df):
    label = df.get_label()
    score = np.sqrt(np.mean(np.power(pred-label,2)))
    return ('RMSLE',score)   

def feval(pred,label):
    return np.sqrt(np.mean(np.power(np.log1p(pred)-np.log1p(label),2)))

def LGBCV(train,test,n_kf,feature_name,params,verb=500):
    pred1 = np.zeros([train.shape[0]])
    pred2 = np.zeros([test.shape[0]])
    imps = np.zeros([len(feature_name),2])
    kf = KFold(n_splits=n_kf,shuffle=True,random_state=223).split(train)
    for i,(tridx,teidx) in enumerate(kf):
        lgbtrain = lgb.Dataset( train[feature_name].iloc[tridx,:] ,train['target'].iloc[tridx] )
        lgbtest = lgb.Dataset(train[feature_name].iloc[teidx,:] ,train['target'].iloc[teidx]  )
        rgs = lgb.train(params,
                            lgbtrain,
                            num_boost_round=3000,
                            valid_sets=lgbtest,
                            verbose_eval=verb,
                            feval=None,
                            early_stopping_rounds=100)
        pred1[teidx] = rgs.predict( train[feature_name].iloc[teidx,:] )
        pred2 += rgs.predict( test[feature_name] )
        imps[:,0] += rgs.feature_importance(importance_type='gain')
        imps[:,1] += rgs.feature_importance(importance_type='split')
    pred2 /= n_kf
    imps /= n_kf
    metrics = np.round(feval(np.expm1(pred1),np.expm1(train['target'])),3)
    imps = pd.DataFrame(data = imps,columns=['gain','split'],index=feature_name )
    return metrics,imps,pred1,pred2

def XGBCV(train,test,n_kf,feature_name,params,verb=500):
    pred1 = np.zeros([train.shape[0]])
    pred2 = np.zeros([test.shape[0]])
    imps = np.zeros([len(feature_name),2])
    kf = KFold(n_splits=n_kf,shuffle=True,random_state=223).split(train)
    ttest = xgb.DMatrix(test[feature_name],test['target'])
    for i,(tridx,teidx) in enumerate(kf):
        xgbtrain = xgb.DMatrix(data = train[feature_name].iloc[tridx,:] ,label = train['target'].iloc[tridx])
        xgbtest = xgb.DMatrix(data = train[feature_name].iloc[teidx,:] ,label = train['target'].iloc[teidx]  )
        rgs = xgb.train(params = params,
                        dtrain = xgbtrain,
                        num_boost_round = 5000,
                        evals= [(xgbtest, 'eval'),],
                        feval=fevalfunc_forxgb,
                        maximize = False,
                        early_stopping_rounds=100,
                        verbose_eval=verb)
        pred1[teidx] = rgs.predict( xgbtest,ntree_limit=rgs.best_ntree_limit )
        pred2 += rgs.predict( ttest,ntree_limit=rgs.best_ntree_limit )
#        imps[:,0] += rgs.get_score(importance_type='gain')
#        imps[:,1] += rgs.get_score(importance_type='weight')
    pred2 /= n_kf
    imps /= n_kf
    metrics = np.round(feval(np.expm1(pred1),np.expm1(train['target'])),3)
    imps = pd.DataFrame(data = imps,columns=['gain','split'],index=feature_name )
    return metrics,imps,pred1,pred2



def remove_low_importace_feature(imps,drop_rate = 0.1):   
    Len = int( imps.shape[0] * drop_rate)
    tick_feature_name = [f for f in imps.index[-Len:] ]
    return tick_feature_name

def cross_top_feature(imps,train,test,TopLen = 10):
    top_feature_name = [f for f in imps.index[:TopLen] ]
    new_feature_train = np.zeros([train.shape[0],int(TopLen*(TopLen-1)/2)])
    new_feature_test = np.zeros([test.shape[0],int(TopLen*(TopLen-1)/2)])
    new_feature_name = []
    ct = 0
    ind = []
    for i in range(TopLen-1):
        for j in range(1,TopLen):
            new_feature_train[:,ct] = train[top_feature_name[i]] + train[top_feature_name[j]]
            new_feature_test[:,ct] = test[top_feature_name[i]] + test[top_feature_name[j]]
            _,p = sts.ks_2samp(new_feature_train[:,ct],new_feature_test[:,ct])
            if p < 0.01:
                ind.append(ct)
                new_feature_name.append( '{0}_plus_{1}'.format(top_feature_name[i],top_feature_name[j]))
    ind = np.array(ind)
    new_feature_train = pd.DataFrame(new_feature_train[:,ind],columns = new_feature_name)
    new_feature_test = pd.DataFrame(new_feature_test[:,ind],columns = new_feature_name)
    return new_feature_train,new_feature_test




# 参数
N_KF = 5    
params_lgb = {   'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.005,
            'num_leaves':255,
            'sub_feature': 0.25,
            'subsample':0.9,
            'subsample_freq':1,
            'lambda_l1':2.5,
            'lambda_l2':5,
            'random_state':0,
            'verbose':-1}

params_xgb = {'objective': 'reg:linear',
              'booster':'gbtree',
              'eval_metric': 'rmse',
              'eta': 0.005,
              'max_depth': 20, 
              'subsample': 0.7, 
#             'min_child_weight':57,
              'colsample_bytree': 0.05,
              'alpha':2.5,
              'lambda':5,
              'random_state': 42, 
              'silent': True}


# 读取数据
print('reading data, loading...')
train = pd.read_csv('../temp/train_version_1.1.csv')
test = pd.read_csv('../temp/test_version_1.1.csv')
feature_name = [f for f in train.columns if f not in ['ID','target']]

# 训练gbm模型
print('training model, loading...')

print('1st time train lgb model, loading...')    
metrics_lgb1,imps_lgb1,pred1_lgb1,pred2_lgb1 = LGBCV(train,test,N_KF,feature_name,params_lgb)
print('*'*50)
print('1st time lgb model rmsle:{0}'.format(metrics_lgb1))
print('*'*50)
#imps_lgb1.sort_values(by='split',inplace = True,ascending = False)
#print('remove low important feature, loading...') 
#tick_feature = remove_low_importace_feature(imps_lgb1,drop_rate = 0.2)
#print('generate cross feature, loading...') 
#train_cross,test_cross = cross_top_feature(imps_lgb1,train,test,TopLen=10)
#print('generate pca feature, loading...') 
#train_pca,test_pca = generate_pca_feature(imps_lgb1,train,test,toplen=100,components_radio=0.85)
#new_train = pd.concat([train,train_cross,train_pca],axis = 1)
#new_test = pd.concat([test,test_cross,test_pca])
#new_feature_name = list(set(feature_name) - set(tick_feature))
#print('*'*50)
#print('total feature nums: {0}'.format(len(feature_name)))
#print('*'*50)
#print('2nd time train lgb model, loading...') 
#metrics_lgb,imps_lgb,pred1_lgb,pred2_lgb = LGBCV(train,test,N_KF,new_feature_name,params_lgb)
#print('*'*50)
#print('2nd time lgb model rmsle:{0}'.format(metrics_lgb))
#print('*'*50)
print('train xgb model, loading...') 
metrics_xgb,imps_xgb,pred1_xgb,pred2_xgb = XGBCV(train,test,N_KF,feature_name,params_xgb)
print('*'*50)
print('xgb model rmsle:{0}'.format(metrics_xgb))
print('*'*50)
   
lr = LinearRegression()
lr.fit( np.vstack((pred1_lgb1,pred1_xgb)).T, train['target'] )
pred1 = lr.predict( np.vstack((pred1_lgb1,pred1_xgb)).T  )
pred2 = lr.predict( np.vstack((pred2_lgb1,pred2_xgb)).T  )

metrics = np.round(feval(np.expm1(pred1),np.expm1(train['target'])),3)

print('*'*50)
print('ensemable model rmsle:{0}'.format(metrics))
print('*'*50) 

result = test[['ID','target']]
result['target'] = np.expm1(pred2)

if np.any(pred2 < 0):
    print('result includes negtive value, please check again')
else:
    time_info = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result.to_csv('../submit/submission_{0}_{1}.csv'.format(metrics,time_info), index=False)
    



    

