from gini import *
import pandas as pd
import numpy as np 
from sklearn import preprocessing
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
import csv
from numpy import *
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.cross_validation import KFold


def getFeatureUsed(df_train):
    #feature_used = list(set(df_train.columns.values)) - set(['Id', 'Hazard']))
    feature_used = ['T1_V1', 'T1_V2', 'T1_V3', 'T1_V4', 'T1_V5',
       'T1_V6', 'T1_V7', 'T1_V8', 'T1_V9', 'T1_V10', 'T1_V11', 'T1_V12',
       'T1_V13', 'T1_V14', 'T1_V15', 'T1_V16', 'T1_V17', 'T2_V1', 'T2_V2',
       'T2_V3', 'T2_V4', 'T2_V5', 'T2_V6', 'T2_V7', 'T2_V8', 'T2_V9',
       'T2_V10', 'T2_V11', 'T2_V12', 'T2_V13', 'T2_V14', 'T2_V15']
    return feature_used

def getFeatureCat(df_train):
    feature_used = getFeatureUsed(df_train)
    feature_cat = []
    for feat in feature_used:
        if type(df_train.loc[1,feat]) is str:
           feature_cat = feature_cat + [feat]
    return feature_cat


def NumericLabel(df_train, df_test):
    feature_cat = getFeatureCat(df_train)
    for feat in feature_cat:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(df_train[feat]) + list(df_test[feat]))
        df_train[feat] = lbl.transform(df_train[feat])
        df_test[feat] = lbl.transform(df_test[feat])
    return {'train': df_train, 'test:': df_test}



def OneHotLabel(df_train, df_test):
    # the input argument is the raw data
    feature_cat = getFeatureCat(df_train)
    ntrain = df_train.shape[0]
    ntest = df_test.shape[0]
    for feat in feature_cat:
        print feat
        #print df_train[feat]
        feat_unique = unique(list(df_train[feat]) + list(df_test[feat]))
        #print feat_unique
        nfeat = len(feat_unique)
        arr_train = np.zeros((ntrain, nfeat+1))
        arr_test = np.zeros((ntest, nfeat+1))
        d = dict(zip(feat_unique,range(len(feat_unique))))
        #print d
        for i in range(ntrain):
            #print "train: ", i , 'out of', ntrain
            #print d[df_train.loc[i,feat]]
            arr_train[i,0] = df_train.loc[i, 'Id']
            arr_train[i, d[df_train.loc[i,feat]] + 1] = 1
        for j in range(ntest):
            #print "test: ", j, 'out of', ntest
            arr_test[j,0] = df_test.loc[j, 'Id']
            arr_test[j, d[df_test.loc[j,feat]] + 1] = 1
        df_train_add = pd.DataFrame(arr_train)
        df_test_add = pd.DataFrame(arr_test)
        feataddname = []
        for j in range(nfeat):
            feataddname = feataddname + [feat + '_'+str(j)]
        df_train_add.columns = ['Id'] + feataddname
        df_test_add.columns = ['Id'] + feataddname
        
        df_train = pd.merge(df_train_add, df_train, how = 'inner', on = ['Id'])
        df_test = pd.merge(df_test_add, df_test, how = 'inner', on = ['Id'])
        
        #df_train = df_train.add(df_train_add, axis=1)
        #df_test = df_test.add(df_test_add, axis=1)
        #df_train = df_train + df_train_add
        #df_test = df_test + df_test_add
    df_train = df_train.drop(feature_cat, axis = 1)
    df_test = df_test.drop(feature_cat, axis = 1)
    df_train.to_csv('../../data/train/train_onehot_temp_v3.csv',header = True, index = False, index_label = False)
    df_train.to_csv('../../data/test/test_onehot_temp_v3.csv',header = True, index = False, index_label = False)


def StatLabel(df_train, df_test):
    feature_cat = getFeatureCat(df_train)
    ntrain = df_train.shape[0]
    ntest = df_test.shape[0]
    for feat in feature_cat:
        print feat
        #print df_train[feat]
        feat_unique = unique(list(df_train[feat]) + list(df_test[feat]))
        #print feat_unique
        #print feat_unique
        nfeat = len(feat_unique)
        arr_train = np.zeros((ntrain, 3 + 1))
        arr_test = np.zeros((ntest, 3 + 1))
        d_mean = dict(zip(feat_unique,range(len(feat_unique))))
        d_std = dict(zip(feat_unique,range(len(feat_unique))))
        d_median = dict(zip(feat_unique,range(len(feat_unique))))
        for lev in feat_unique:
            data = df_train.loc[df_train[feat] == lev ,'Hazard']
            d_mean[lev] = mean(data)
            d_std[lev] = std(data)
            d_median[lev] = median(data)
        #print d_mean
        for i in range(ntrain):
            #print feat, "train", i, 'out of', ntrain
            feat_val = df_train.loc[i,feat]
            #print feat_val
            #print "train: ", i , 'out of', ntrain
            #print d[df_train.loc[i,feat]]
            arr_train[i,0] = df_train.loc[i, 'Id']
            arr_train[i,1] = d_mean[feat_val]
            arr_train[i,2] = d_std[feat_val]
            arr_train[i,3] = d_median[feat_val]
        for j in range(ntest):
            feat_val = df_test.loc[j,feat]
            #print feat, "test", j, 'out of', ntest
            #print "test: ", j, 'out of', ntest
            arr_test[j,0] = df_test.loc[j, 'Id']
            arr_test[j,1] = d_mean[feat_val]
            arr_test[j,2] = d_std[feat_val]
            arr_test[j,3] = d_median[feat_val]
        
        df_train_add = pd.DataFrame(arr_train)
        df_test_add = pd.DataFrame(arr_test)
        feataddname = []
        feataddname = [feat+'_'+'mean'] + [feat+'_'+'std'] + [feat+'_'+'median']

        df_train_add.columns = ['Id'] + feataddname
        df_test_add.columns = ['Id'] + feataddname
        
        df_train = pd.merge(df_train_add, df_train, how = 'inner', on = ['Id'])
        df_test = pd.merge(df_test_add, df_test, how = 'inner', on = ['Id'])
        
        #df_train = df_train.add(df_train_add, axis=1)
        #df_test = df_test.add(df_test_add, axis=1)
        #df_train = df_train + df_train_add
        #df_test = df_test + df_test_add
    df_train = df_train.drop(feature_cat, axis = 1)
    df_test = df_test.drop(feature_cat, axis = 1)
    df_train.to_csv('../../data/train/train_stat_v0.csv',header = True, index = False, index_label = False)
    df_test.to_csv('../../data/test/test_stat_v0.csv',header = True, index = False, index_label = False)


def mergeData():
    #df_train = pd.read_csv('../../data/train/train.csv', header = 0)
    df_train_onehot = pd.read_csv('../../data/train/train_onehot_v2.csv', header = 0)
    df_train_stat = pd.read_csv('../../data/train/train_stat_v0.csv', header = 0)
    lscol1 = set(df_train_onehot.columns.values) - set(['Id','Hazard'])
    lscol2 = set(df_train_stat.columns.values) - set(['Id','Hazard'])
    lscolcom = lscol1.intersection(lscol2)
    #print list(lscolcom)
    df_train_onehot = df_train_onehot.drop(list(lscolcom) + ['Hazard'], axis = 1)
    df_train_merge = pd.merge(df_train_onehot, df_train_stat, how = 'inner', on = ['Id'])
    df_train_merge.to_csv('../../data/train/train_onehot_stat_v0.csv', header = True, index = False, index_label = False)

    df_test_onehot = pd.read_csv('../../data/test/test_onehot_v2.csv', header = 0)
    df_test_stat = pd.read_csv('../../data/test/test_stat_v0.csv', header = 0)
    lscol1 = set(df_test_onehot.columns.values) - set(['Id'])
    lscol2 = set(df_test_stat.columns.values) - set(['Id'])
    lscolcom = lscol1.intersection(lscol2)
    df_test_onehot = df_test_onehot.drop(list(lscolcom), axis = 1)
    df_test_merge = pd.merge(df_test_onehot, df_test_stat, how = 'inner', on = ['Id'])
    df_test_merge.to_csv('../../data/test/test_onehot_stat_v0.csv', header = True, index = False, index_label = False)


if __name__ == '__main__':
    df_train = pd.read_csv('../../data/train/train.csv', header = 0)
    df_test = pd.read_csv('../../data/test/test.csv', header = 0)
    #df_train = df_train[0:1000]
    #df_test = df_test[0:1000]
    #print getFeatureCat(df_train)
    #OneHotLabel(df_train, df_test)
    StatLabel(df_train, df_test)
    mergeData()








