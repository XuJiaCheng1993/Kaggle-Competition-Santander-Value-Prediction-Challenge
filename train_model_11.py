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



# 读取数据
print('reading data, loading...')
train = pd.read_csv('../temp/train_th10.001.csv')
test = pd.read_csv('../temp/test_th10.001.csv')
feature_name = [f for f in train.columns if f not in ['ID','target']]



# 标签转换
train['target'] = np.log1p(train['target'] )

# 数据归一化
print('propressing data, loading...')
scaler = MinMaxScaler(feature_range=(-1,1))
train_scaler = scaler.fit_transform(train[feature_name])
test_scaler = scaler.transform(test[feature_name])

#train_scaler = np.log1p(train[feature_name])
#test_scaler = np.log1p(test[feature_name])

train[feature_name] = train_scaler
test[feature_name] = test_scaler

# 训练模型
print('training model, loading...')
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

def LGBCV(train,test,n_kf,feature_name,params):
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
                            verbose_eval=100,
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

def XGBCV(train,test,n_kf,feature_name,params):
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
                        verbose_eval=100)
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

def cross_top_feature(imps,train,test,TopLen = 5):
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
    return new_feature_train,new_feature_test,new_feature_name

params = {   'boosting_type': 'gbdt',
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
#          'min_child_weight':57,
          'colsample_bytree': 0.05,
          'alpha':2.5,
          'lambda':5,
          'random_state': 42, 
          'silent': True}

n_kf = 5

def training(imps,train,test,n_kf,feature_name,params,drop_rate=0.01,top_featuren_num = 5):
    # 1st train model
#    metrics,imps,pred1,pred2 = LGBCV(train,test,n_kf,feature_name,params)
    # sort feature by featureimportance
    imps['score'] = imps['gain'] * imps['split']
    imps.sort_values(by='score',inplace = True,ascending = False)
    # remove low importance feature
    tick_feature_name = remove_low_importace_feature(imps,drop_rate)
    # generate poly features
    poly_train,poly_test,poly_feature_name = cross_top_feature(imps,train,test,top_featuren_num)
    # update train and test set
    train = pd.concat([train,poly_train],axis = 1)
    test = pd.concat([test,poly_test],axis = 1)
    # update useful feature name
    feature_name = list(set([f for f in train.columns if f not in ['ID','target']])-set(tick_feature_name))
    print('*'*50)
    print('new feature nums: {0}'.format(len(feature_name)))
    print('*'*50)    
    # 2nd train model
    metrics_2nd,imps_2nd,pred1_2nd,pred2_2nd = LGBCV(train,test,n_kf,feature_name,params)
#    print('1st model metrics:{0} , 2nd model metrics{1}'.format(metrics,metrics_2nd))
#    if metrics< metrics_2nd:
#        return metrics,imps,pred1,pred2
#    else:
    return metrics_2nd,imps_2nd,pred1_2nd,pred2_2nd
    
metrics_lgb1,imps,pred1_lgb1,pred2_lgb1 = LGBCV(train,test,n_kf,feature_name,params)
#imps.sort_values(by='split',inplace = True,ascending = False)
#tick_feature = remove_low_importace_feature(imps,drop_rate = 0.2)
#new_feature_name = list(set(feature_name)-set(tick_feature))
#metrics_lgb,imps,pred1_lgb,pred2_lgb = LGBCV(train,test,n_kf,new_feature_name,params)
#
#metrics_xgb,imps,pred1_xgb,pred2_xgb = XGBCV(train,test,n_kf,feature_name,params_xgb)
#
#lr = LinearRegression()
#lr.fit( np.vstack((pred1_lgb,pred1_xgb)).T, train['target'] )
#pred1 = lr.predict( np.vstack((pred1_lgb,pred1_xgb)).T  )
#pred2 = lr.predict( np.vstack((pred2_lgb,pred2_xgb)).T  )
#metrics = np.round(feval(np.expm1(pred1),np.expm1(train['target'])),3)
#
#result = test[['ID','target']]
#result['target'] = np.expm1(pred2)
#
#if np.any(pred2 < 0):
#    print('result includes negtive value, please check again')
#else:
#    time_info = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#    result.to_csv('../submit/submission_{0}_{1}.csv'.format(metrics,time_info), index=False)
    



    

