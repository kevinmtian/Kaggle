from KgModels import Models
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

class Regressor:

        def __init__(self):
              self.dataMat = []
              self.labelMat = []
              self.testData = []
              self.params_best = []
              self.params_cv = []
              self.feature_used = []
        
        def setFeature(self):
              #self.feature_used = ['T1_V1', 'T1_V2', 'T1_V3', 'T1_V4', 'T1_V5',
       #'T1_V6', 'T1_V7', 'T1_V8', 'T1_V9', 'T1_V10', 'T1_V11', 'T1_V12',
       #'T1_V13', 'T1_V14', 'T1_V15', 'T1_V16', 'T1_V17', 'T2_V1', 'T2_V2',
       #'T2_V3', 'T2_V4', 'T2_V5', 'T2_V6', 'T2_V7', 'T2_V8', 'T2_V9',
       #'T2_V10', 'T2_V11', 'T2_V12', 'T2_V13', 'T2_V14', 'T2_V15']
       
              self.feature_used = ['T1_V1', 'T1_V2', 'T1_V3', 'T1_V10', 'T1_V13',
       'T1_V14', 'T2_V1', 'T2_V2', 'T2_V4', 'T2_V6', 'T2_V7', 'T2_V8',
       'T2_V9', 'T2_V10', 'T2_V14', 'T2_V15',
       'T1_V4_0', 'T1_V4_1',
       'T1_V4_2', 'T1_V4_3', 'T1_V4_4', 'T1_V4_5', 'T1_V4_6', 'T1_V4_7',
       'T1_V5_0', 'T1_V5_1', 'T1_V5_2', 'T1_V5_3', 'T1_V5_4', 'T1_V5_5',
       'T1_V5_6', 'T1_V5_7', 'T1_V5_8', 'T1_V5_9', 'T1_V6_0', 'T1_V6_1',
       'T1_V7_0', 'T1_V7_1', 'T1_V7_2', 'T1_V7_3', 'T1_V8_0', 'T1_V8_1',
       'T1_V8_2', 'T1_V8_3', 'T1_V9_0', 'T1_V9_1', 'T1_V9_2', 'T1_V9_3',
       'T1_V9_4', 'T1_V9_5', 'T1_V11_0', 'T1_V11_1', 'T1_V11_2',
       'T1_V11_3', 'T1_V11_4', 'T1_V11_5', 'T1_V11_6', 'T1_V11_7',
       'T1_V11_8', 'T1_V11_9', 'T1_V11_10', 'T1_V11_11', 'T1_V12_0',
       'T1_V12_1', 'T1_V12_2', 'T1_V12_3', 'T1_V15_0', 'T1_V15_1',
       'T1_V15_2', 'T1_V15_3', 'T1_V15_4', 'T1_V15_5', 'T1_V15_6',
       'T1_V15_7', 'T1_V16_0', 'T1_V16_1', 'T1_V16_2', 'T1_V16_3',
       'T1_V16_4', 'T1_V16_5', 'T1_V16_6', 'T1_V16_7', 'T1_V16_8',
       'T1_V16_9', 'T1_V16_10', 'T1_V16_11', 'T1_V16_12', 'T1_V16_13',
       'T1_V16_14', 'T1_V16_15', 'T1_V16_16', 'T1_V16_17', 'T1_V17_0',
       'T1_V17_1', 'T2_V3_0', 'T2_V3_1', 'T2_V5_0', 'T2_V5_1', 'T2_V5_2',
       'T2_V5_3', 'T2_V5_4', 'T2_V5_5', 'T2_V11_0', 'T2_V11_1', 'T2_V12_0',
       'T2_V12_1', 'T2_V13_0', 'T2_V13_1', 'T2_V13_2', 'T2_V13_3',
       'T2_V13_4']

        def setParamsCv(self):
              params = {}
              params["objective"] = "reg:linear"
              params["gamma"] = 0
              params["eta"] = 0.0005
              params["min_child_weight"] = 5
              params["subsample"] = 0.6
              params["colsample_bytree"] = 0.6
              params["scale_pos_weight"] = 1.0
              params["silent"] = 1
              params["max_depth"] = 12
              #plst = list(params.items())
              self.params_cv = params
              self.num_rounds_cv = 8000
              self.early_stopping_rounds_cv = 5

        def setParamsBest(self):
              params = {}
              params["objective"] = "reg:linear"
              params["gamma"] = 0
              params["eta"] = 0.001
              params["min_child_weight"] = 5
              params["subsample"] = 0.6
              params["colsample_bytree"] = 0.6
              params["scale_pos_weight"] = 1.0
              params["silent"] = 1
              params["max_depth"] = 12
              #plst = list(params.items())
              self.params_best = params
              self.num_rounds_best = 4000
              self.early_stopping_rounds_best = 5

        def loadDataSet(self,trainfile):
              df_train = pd.read_csv(trainfile, header = 0)
              self.dftrain = df_train
              self.dataMat = np.array(df_train[self.feature_used])
              self.labelMat = np.array(df_train['Hazard'])
              self.trainid = np.array(df_train['Id'])

              #print df_train.head()
              print self.dataMat.shape
              #print self.dataMat[0:2,0:5]
              #print self.labelMat[0:10]

        def loadTestSet(self,testfile):
              df_test = pd.read_csv(testfile, header = 0)
              self.dftest = df_test
              df_test = df_test.sort(columns = 'Id', ascending = True)
              self.testData = np.array(df_test[self.feature_used])
              self.testid = np.array(df_test['Id'])
              
              #print df_test.head()
              #print self.testData[0:2,:]
              print self.testData.shape
        
        def getCatFeat(self):
              train = self.dftrain
              test = self.dftest
              self.feature_cat = []
              for feat in self.feature_used:
                  if type(train.loc[1,feat]) is str:
                     self.feature_cat = self.feature_cat + [feat]
                  
        def OneHotData(self):
              # run this only after running transformData
              train = self.dftrain
              test = self.dftest
              for feat in self.feature_cat:
                  code_num = unique(list(train[feat]) + list(test[feat]))
                  code_intarr = []
                  for i in code_num:
                      code_intarr = code_intarr + [[int(i)]]
                  enc = OneHotEncoder()
                  enc.fit(code_intarr)
                  for i in range(train.shape[0]):
                      print feat,'-', i
                      for j in code_num:
                          #print enc.transform([[int(train.loc[i,feat])]]).toarray()[0][j]
                          train.loc[i,feat + '_'+str(j)] = enc.transform([[int(train.loc[i,feat])]]).toarray()[0][j]
                  for i in range(test.shape[0]):
                      print feat,'-', i
                      for j in code_num:
                          #print enc.transform([[int(train.loc[i,feat])]]).toarray()[0][j]
                          test.loc[i,feat + '_'+str(j)] = enc.transform([[int(test.loc[i,feat])]]).toarray()[0][j]
              #self.dataMat = train.astype(float)
              #self.testData = test.astype(float)
              
              train = train.drop(self.feature_cat, axis = 1)
              test = test.drop(self.feature_cat, axis = 1)
              train.to_csv('../../data/train/train_onehot_v2.csv',header = True, index = False, index_label = False)
              test.to_csv('../../data/test/test_onehot_v2.csv',header = True, index = False, index_label = False)
              
        def transformDfData(self):
              train = self.dftrain
              test = self.dftest
              for feat in self.feature_cat:
                  lbl = preprocessing.LabelEncoder()
                  lbl.fit(list(train[feat]) + list(test[feat]))
                  train[feat] = lbl.transform(train[feat])
                  test[feat] = lbl.transform(test[feat])
              self.dftrain = train
              self.dftest = test
              #self.dataMat = train.astype(float)
              #self.testData = test.astype(float)
              
              

        def transformData(self):
              train = self.dataMat
              test = self.testData
              for i in range(train.shape[1]):
                  if type(train[1,i]) is str:
                     lbl = preprocessing.LabelEncoder()
                     lbl.fit(list(train[:,i]) + list(test[:,i]))
                     train[:,i] = lbl.transform(train[:,i])
                     test[:,i] = lbl.transform(test[:,i])
              self.dataMat = train.astype(float)
              self.testData = test.astype(float)
              #print self.dataMat
              
        #def XgbData(self):
        #      offset = 5000
        #      self.xgtest = xgb.DMatrix(self.testData)
        #      self.xgtrain = xgb.DMatrix(self.dataMat, label=self.labelMat)
        #      self.xgtrain_train = xgb.DMatrix(self.dataMat[offset:,:], label=self.labelMat[offset:])
        #      self.xgtrain_val = xgb.DMatrix(self.dataMat[:offset,:], label=self.labelMat[:offset])
        
        
        
        
        def XgbTrain(self, submitfile):
              offset = 5000
              X_train, y_train = self.dataMat, self.labelMat
              X_test = self.testData
              xgtest = xgb.DMatrix(X_test)
              
              xgtrain_train = xgb.DMatrix(X_train[offset:,:], label=y_train[offset:])
              xgtrain_val = xgb.DMatrix(X_train[:offset,:], label=y_train[:offset])
              
                      
              watchlist = [(xgtrain_train, 'train'),(xgtrain_val, 'val')]
              model = xgb.train(self.params_best, xgtrain_train, self.num_rounds_best, watchlist,early_stopping_rounds=self.early_stopping_rounds_best)
              preds1 = model.predict(xgtest)
                      
              X_train = X_train[::-1,:]
              y_train = y_train[::-1]

              xgtrain_train = xgb.DMatrix(X_train[offset:,:], label=y_train[offset:])
              xgtrain_val = xgb.DMatrix(X_train[:offset,:], label=y_train[:offset])

              watchlist = [(xgtrain_train, 'train'),(xgtrain_val, 'val')]
              model = xgb.train(self.params_best, xgtrain_train, self.num_rounds_best, watchlist, early_stopping_rounds=self.early_stopping_rounds_best)
              preds2 = model.predict(xgtest)
                      
              preds = preds1 + preds2
              #preds = pd.DataFrame({"Id": self.testid, "Hazard": preds})
              if submitfile!='':
                writer=csv.writer(open(submitfile,'wb'))
                writer.writerow(['ID','Hazard'])
                for i in range(len(preds)):
                    line = [self.testid[i], preds[i]]
                    writer.writerow(line)
              #preds.to_csv(submitfile)
        

        def XgbCv(self, cvfile):
              offset = round(5000 * 2/3)
              X, y = self.dataMat, self.labelMat
              kf = KFold(len(y), n_folds=3, shuffle = True, random_state = 25)
              scores = []
              #df_cvtrain = pd.DataFrame(columns = ['enrollment_id','prob'])
              #df_cvtrain['enrollment_id'] = self.trainid
              for Itr,Its in kf:
                      X_test, y_test = X[np.array(Its),:], y[np.array(Its)]
                      X_train, y_train = X[np.array(Itr),:], y[np.array(Itr)]
                      #xgtrain = xgb.DMatrix(X_train, label = y_train)
                      #print X_test
                      xgtest = xgb.DMatrix(X_test)
                      
                      xgtrain_train = xgb.DMatrix(X_train[offset:,:], label=y_train[offset:])
                      xgtrain_val = xgb.DMatrix(X_train[:offset,:], label=y_train[:offset])
                      
                      watchlist = [(xgtrain_train, 'train'),(xgtrain_val, 'val')]
                      model = xgb.train(self.params_cv, xgtrain_train, self.num_rounds_cv, watchlist, early_stopping_rounds=self.early_stopping_rounds_cv,feval = None)
                      preds1 = model.predict(xgtest)
                      
                      X_train = X_train[::-1,:]
                      y_train = y_train[::-1]

                      xgtrain_train = xgb.DMatrix(X_train[offset:,:], label=y_train[offset:])
                      xgtrain_val = xgb.DMatrix(X_train[:offset,:], label=y_train[:offset])

                      watchlist = [(xgtrain_train, 'train'),(xgtrain_val, 'val')]
                      model = xgb.train(self.params_cv, xgtrain_train, self.num_rounds_cv, watchlist, early_stopping_rounds=self.early_stopping_rounds_cv,feval = None)
                      preds2 = model.predict(xgtest)
                      
                      preds = preds1 + preds2
                      score = Gini(y_test, preds)
                      
                      print ': GINI SCORE :', score
                      
                      scores = scores + [score]
              scores = np.array(scores)
              print "gini_mean: ", scores.mean(), "gini_std:", scores.std()
              if cvfile != '':
                 writer = csv.writer(open(cvfile, 'wb'))
                 #f = open(cvfile,'a')
                 #writer = csv.writer(f)
                 linehead = self.params_cv.keys() + ['num_rounds'] + ['early_stopping_rounds'] + ['score_mean'] + ['score_std']
                 writer.writerow(linehead)
                 #writer.writerow('\n')
                 line = self.params_cv.values() + [self.num_rounds_cv] + [self.early_stopping_rounds_cv] + [scores.mean()] + [scores.std()]
                 writer.writerow(line)
                 #f.close()


if __name__=='__main__':
             version = 21
             CLF = Regressor()
             CLF.setFeature()
             CLF.setParamsCv()
             CLF.setParamsBest()
             
             print "version:", version
             print "xgb"
             #print CLF.feature_used

             trainData = '../../data/train/train_onehot_v2.csv'
             testData = '../../data/test/test_onehot_v2.csv'

             print "traindata from: ", trainData
             print "testdata from: ", testData
             
             submitfile = '../../data/pred/pred_'+ 'xgb' + '_v' + str(version) + '.csv'
             cvfile = '../../data/cv/cv_' + 'xgb' + '_v' + str(version) + '.csv'
             logfile = '../../data/cvlog/log_' + 'xgb' + '_v' + str(version) + '.txt'
             
             writer=csv.writer(open(logfile,'wb'))
             writer.writerow(['xgb'])
             writer.writerow(':::::::::::::::::::::::::')
             writer.writerow(CLF.feature_used)
             writer.writerow(':::::::::::::::::::::::::')
             writer.writerow(CLF.params_best.keys())
             writer.writerow(CLF.params_best.values())
             writer.writerow(':::::::::::::::::::::::::')
             writer.writerow([trainData])
             
             
             CLF.loadDataSet(trainData)
             CLF.loadTestSet(testData)
             #CLF.getCatFeat()
             #CLF.transformDfData()
             #CLF.OneHotData()
             #CLF.XgbTrain(submitfile)
             CLF.XgbCv(cvfile)

