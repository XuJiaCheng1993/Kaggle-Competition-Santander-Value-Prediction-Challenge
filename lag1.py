# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 21:20:13 2018

@author: 徐嘉诚
"""
import pandas as pd
import numpy as np
import gc
#import os
import datetime

from sklearn.metrics import mean_squared_error
from tqdm import tqdm, tqdm_notebook
tqdm.pandas(tqdm_notebook)

#filelist = os.listdir('../data')

def get_beautiful_test(test):
    test_rnd = np.round(test.iloc[:, 1:], 2)
    ugly_indexes = []
    non_ugly_indexes = []
    for idx in tqdm(range(len(test))):
        if not np.all(
            test_rnd.iloc[idx, :].values==test.iloc[idx, 1:].values
        ):
            ugly_indexes.append(idx)
        else:
            non_ugly_indexes.append(idx)
    print(len(ugly_indexes), len(non_ugly_indexes))
    np.save('test_ugly_indexes', np.array(ugly_indexes))
    np.save('test_non_ugly_indexes', np.array(non_ugly_indexes))
    test = test.iloc[non_ugly_indexes].reset_index(drop=True)
    return test, non_ugly_indexes, ugly_indexes

def fast_get_leak(df, cols, lag=0):
    d1 = df[cols[:-lag-2]].apply(tuple, axis=1).to_frame().rename(columns={0: 'key'})
    d2 = df[cols[lag+2:]].apply(tuple, axis=1).to_frame().rename(columns={0: 'key'})
    d2['pred'] = df[cols[lag]]
    #d2 = d2[d2.pred != 0] ### to make output consistent with Hasan's function
    d3 = d2[~d2.duplicated(['key'], keep=False)]
    return d1.merge(d3, how='left', on='key').pred.fillna(0)

def compiled_leak_result():
    
    max_nlags = len(cols) - 2
    train_leak = train[["ID", "target"] + cols]
    train_leak["compiled_leak"] = 0
    train_leak["nonzero_mean"] = train[columns_name].apply(
        lambda x: np.expm1(np.log1p(x[x!=0]).mean()), axis=1
    )
    
    scores = []
    leaky_value_counts = []
    leaky_value_corrects = []
    leaky_cols = []
    
    for i in range(max_nlags):
        c = "leaked_target_"+str(i)
        
        print('Processing lag', i)
        train_leak[c] = fast_get_leak(train_leak, cols, i)
        
        leaky_cols.append(c)
        train_leak = train.join(
            train_leak.set_index("ID")[leaky_cols+["compiled_leak", "nonzero_mean"]], 
            on="ID", how="left"
        )
        zeroleak = train_leak["compiled_leak"]==0
        train_leak.loc[zeroleak, "compiled_leak"] = train_leak.loc[zeroleak, c]
        leaky_value_counts.append(sum(train_leak["compiled_leak"] > 0))
        _correct_counts = sum(train_leak["compiled_leak"]==train_leak["target"])
        leaky_value_corrects.append(_correct_counts/leaky_value_counts[-1])
        print("Leak values found in train", leaky_value_counts[-1])
        print(
            "% of correct leaks values in train ", 
            leaky_value_corrects[-1]
        )
        tmp = train_leak.copy()
        tmp.loc[zeroleak, "compiled_leak"] = tmp.loc[zeroleak, "nonzero_mean"]
        scores.append(np.sqrt(mean_squared_error(y, np.log1p(tmp["compiled_leak"]).fillna(14.49))))
        print(
            'Score (filled with nonzero mean)', 
            scores[-1]
        )
    result = dict(
        score=scores, 
        leaky_count=leaky_value_counts,
        leaky_correct=leaky_value_corrects,
    )
    return train_leak, result




train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
sample_submission = pd.read_csv('../data/sample_submission.csv')
columns_name = [f for f in train.columns if f not in ['ID','target'] ]
y = np.log1p(train["target"]).values

try:
    ugly_indexes = np.load('test_ugly_indexes.npy')
    non_ugly_indexes = np.load('test_non_ugly_indexes.npy')
    test = test.iloc[non_ugly_indexes].reset_index(drop=True)
except:
    test, non_ugly_indexes, ugly_indexes = get_beautiful_test(test)

test["target"] = train["target"].mean()

cols = ['f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec', '6eef030c1', '15ace8c9f', 
        'fb0f5dbfe', '58e056e12', '20aa07010', '024c577b9', 'd6bb78916', 'b43a7cfd5', 
        '58232a6fb', '1702b5bf0', '324921c7b', '62e59a501', '2ec5b290f', '241f0f867', 
        'fb49e4212', '66ace2992', 'f74e8f13d', '5c6487af1', '963a49cdc', '26fc93eb7', 
        '1931ccfdd', '703885424', '70feb1494', '491b9ee45', '23310aa6f', 'e176a204a', 
        '6619d81fc', '1db387535', 'fc99f9426', '91f701ba2', '0572565c2', '190db8488', 
        'adb64ff71', 'c47340d97', 'c5a231d81', '0ff32eb98']


train_leak, result = compiled_leak_result()

best_lag = np.argmin(result['score'])

def rewrite_compiled_leak(leak_df, lag):
    leak_df["compiled_leak"] = 0
    for i in range(lag):
        c = "leaked_target_"+str(i)
        zeroleak = leak_df["compiled_leak"]==0
        leak_df.loc[zeroleak, "compiled_leak"] = leak_df.loc[zeroleak, c]
    return leak_df

leaky_cols = [c for c in train_leak.columns if 'leaked_target_' in c]
train_leak = rewrite_compiled_leak(train_leak, best_lag)
train_leak[['ID']+leaky_cols+['compiled_leak']].head()

train_res = train_leak[leaky_cols+['compiled_leak']].replace(0.0, np.nan)

def compiled_leak_result_test(max_nlags):
    test_leak = test[["ID", "target"] + cols]
    test_leak["compiled_leak"] = 0
    test_leak["nonzero_mean"] = test[columns_name].apply(
        lambda x: np.expm1(np.log1p(x[x!=0]).mean()), axis=1
    )
    
    scores = []
    leaky_value_counts = []
    # leaky_value_corrects = []
    leaky_cols = []
    
    for i in range(max_nlags):
        c = "leaked_target_"+str(i)
        
        print('Processing lag', i)
        test_leak[c] = fast_get_leak(test_leak, cols, i)
        
        leaky_cols.append(c)
        test_leak = test.join(
            test_leak.set_index("ID")[leaky_cols+["compiled_leak", "nonzero_mean"]], 
            on="ID", how="left"
        )[["ID", "target"] + cols + leaky_cols+["compiled_leak", "nonzero_mean"]]
        zeroleak = test_leak["compiled_leak"]==0
        test_leak.loc[zeroleak, "compiled_leak"] = test_leak.loc[zeroleak, c]
        leaky_value_counts.append(sum(test_leak["compiled_leak"] > 0))
        #_correct_counts = sum(train_leak["compiled_leak"]==train_leak["target"])
        #leaky_value_corrects.append(_correct_counts/leaky_value_counts[-1])
        print("Leak values found in test", leaky_value_counts[-1])
        #print(
        #    "% of correct leaks values in train ", 
        #    leaky_value_corrects[-1]
        #)
        #tmp = train_leak.copy()
        #tmp.loc[zeroleak, "compiled_leak"] = tmp.loc[zeroleak, "nonzero_mean"]
        #scores.append(np.sqrt(mean_squared_error(y, np.log1p(tmp["compiled_leak"]).fillna(14.49))))
        #print(
        #    'Score (filled with nonzero mean)', 
        #    scores[-1]
        #)
    result = dict(
        # score=scores, 
        leaky_count=leaky_value_counts,
        # leaky_correct=leaky_value_corrects,
    )
    return test_leak, result
test_leak, test_result = compiled_leak_result_test(max_nlags=38)
test_leak = rewrite_compiled_leak(test_leak, best_lag)
test_leak.loc[test_leak["compiled_leak"]==0, "compiled_leak"] = test_leak.loc[test_leak["compiled_leak"]==0, "nonzero_mean"]


sample_submission['target'] = 0
sample_submission.iloc[non_ugly_indexes, 1] = test_leak["compiled_leak"].values

time_info = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
sample_submission.to_csv('../submit/submission_{0}_{1}.csv'.format(np.min(result['score']),time_info), index=False)
