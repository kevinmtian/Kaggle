from KgModels import Models
from gini import *

import csv
import pandas as pd
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.cross_validation import KFold
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import *
from sklearn import grid_search
from sklearn.grid_search import *
from sklearn import cross_validation

from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.cross_validation import cross_val_score,ShuffleSplit
from sklearn.datasets import load_iris
from sklearn.preprocessing import scale
from sklearn.metrics import zero_one_loss, classification_report, accuracy_score, log_loss, roc_curve, auc, roc_auc_score

class Classifier:
         def __init__(self):
              self.dataMat = []
              self.labelMat = []
              self.testData = []
              self.model = []
              self.model_best = []
              self.params_best = []
              self.params_cv = []
              self.feature_used = []
         

         
         def setModel(self, model):
              self.modelname = model.modelname
              self.model = model.model
              self.model_best =model.model_best
              self.params_cv = model.params_cv
              self.params_best = model.params_best
              self.feature_used = model.feature_used
              #self.feature_dict = model.feature_dict

         def loadDataSet(self,trainfile):
              df_train = pd.read_csv(trainfile, header = 0)
              self.dftrain = df_train
              self.dataMat = np.array(df_train[self.feature_used])
              self.labelMat = np.array(df_train['Hazard'])
              self.labelMat = map(float, self.labelMat)
              self.labelMat = np.array(self.labelMat)
              
              ### apply label transformation
              #self.labelMat2 = pow(self.labelMat, 2)
              ###
              self.trainid = np.array(df_train['Id'])
              print self.dataMat.shape
              print self.labelMat
         


         def loadTestSet(self,testfile):
              df_test = pd.read_csv(testfile, header = 0)
              self.dftest = df_test
              df_test = df_test.sort(columns = 'Id', ascending = True)
              self.testData = np.array(df_test[self.feature_used])
              self.testid = np.array(df_test['Id'])
              
              #print df_test.head()
              #print self.testData[0:2,:]
              #print self.testData.shape

              #print df_test.head()
              #print self.testData[0:2,:]
              print self.testData.shape

         def modelTrain(self,submitfile):
             X, y = self.dataMat, self.labelMat
             X_test = self.testData
             clf = self.model_best
             print clf
             clf.fit(X, y)
             preds = clf.predict(X_test)
             if submitfile!='':
                writer=csv.writer(open(submitfile,'wb'))
                writer.writerow(['ID','Hazard'])
                for i in range(len(prob_pos)):
                    line = [self.testid[i], preds[i]]
                    writer.writerow(line)
         
         def modelValArtLazy(self, CvTrainFile, cvfile):
              X, y = self.dataMat, self.labelMat
              kf = KFold(len(y), n_folds=3, shuffle = True, random_state = 25)
              clf = self.model_best
              print clf
              scores = []
              scores_train = []
              #df_cvtrain = pd.DataFrame(columns = ['enrollment_id','prob'])
              #df_cvtrain['enrollment_id'] = self.trainid
              count = 0
              for Itr,Its in kf:
                      if count > 0: break
                      X_test, y_test = X[np.array(Its),:], y[np.array(Its)]
                      X_train, y_train = X[np.array(Itr),:], y[np.array(Itr)]
                      #if self.pca != '':
                      #   X_train = self.pca.fit_transform(X_train)
                      #   X_test = self.pca.fit_transform(X_test)
                      clf.fit(X_train, y_train)
                      preds_train = clf.predict(X_train)
                      preds = clf.predict(X_test)
                      #df_cvtrain['prob'][Its] = prob_pos
                      
                      score_train = Gini(y_train, preds_train)
                      score = Gini(y_test, preds)
                      
                      #print ':ROC_AUC_TRAIN_:', score_train
                      print ': GINI :', score
                      
                      scores_train = scores_train + [score_train]
                      scores = scores + [score]
                      count += 1
              scores = np.array(scores)
              scores_train = np.array(scores_train)
              #scores_train = np.array(scores_train)
              print "train_GINI_mean: ", scores_train.mean(), "train_GINI_std:", scores_train.std()
              print "GINI_mean: ", scores.mean(), "GINI_std:", scores.std()
              #if CvTrainFile != '':
                 #df_cvtrain.to_csv(CvTrainFile, header = True, index = False, index_label = False)
              if cvfile != '':
                 writer = csv.writer(open(cvfile, 'wb'))
                 linehead = self.params_best.keys() + ['score_mean'] + ['score_std'] + ['train_score_mean'] + ['train_score_std']
                 writer.writerow(linehead)
                 line = self.params_best.values() + [scores.mean()] + [scores.std()] + [scores_train.mean()] + [scores_train.std()]
                 writer.writerow(line)

         def modelValArt(self, CvTrainFile, cvfile):
              X, y = self.dataMat, self.labelMat
              kf = KFold(len(y), n_folds=3, shuffle = True, random_state = 25)
              clf = self.model_best
              scores = []
              scores_train = []
              #df_cvtrain = pd.DataFrame(columns = ['enrollment_id','prob'])
              #df_cvtrain['enrollment_id'] = self.trainid
              for Itr,Its in kf:
                      X_test, y_test = X[np.array(Its),:], y[np.array(Its)]
                      X_train, y_train = X[np.array(Itr),:], y[np.array(Itr)]
                      if self.pca != '':
                         X_train = self.pca.fit_transform(X_train)
                         X_test = self.pca.fit_transform(X_test)
                      clf.fit(X_train, y_train)
                      prob_pos_train = clf.predict(X_train)
                      prob_pos = clf.predict(X_test)
                      #df_cvtrain['prob'][Its] = prob_pos
                      
                      score_train = Gini(y_train, prob_pos_train)
                      score = Gini(y_test, prob_pos)
                      
                      print ':GINI_TRAIN_:', score_train
                      print ': GINI :', score
                      
                      scores_train = scores_train + [score_train]
                      scores = scores + [score]
              scores = np.array(scores)
              scores_train = np.array(scores_train)
              print "train_GINI_mean: ", scores_train.mean(), "train_GINI_std:", scores_train.std()
              print "GINI_mean: ", scores.mean(), "GINI_std:", scores.std()
              if CvTrainFile != '':
                 df_cvtrain.to_csv(CvTrainFile, header = True, index = False, index_label = False)
              if cvfile != '':
                 writer = csv.writer(open(cvfile, 'wb'))
                 #f = open(cvfile,'a')
                 #writer = csv.writer(f)
                 linehead = self.params_best.keys() + ['score_mean'] + ['score_std'] + ['train_score_mean'] + ['train_score_std']
                 writer.writerow(linehead)
                 #writer.writerow('\n')
                 line = self.params_best.values() + [scores.mean()] + [scores.std()] + [scores_train.mean()] + [scores_train.std()]
                 writer.writerow(line)
                 #f.close()




         def getFeatureImportance(self):
             if self.modelname != 'random_forest':
                raise ValueError("No Feature Importance will be generated")
             else:
                X, y = self.dataMat, self.labelMat
                clf = self.model_best
                clf.fit(X,y)
                writer = csv.writer(open('../../data/cv/feature_importance.csv','wb'))
                writer.writerow(['feature','importance'])
                for i in range(len(self.feature_used)):
                    line = [self.feature_used[i], clf.feature_importances_[i]]
                    writer.writerow(line)

if __name__=='__main__':
             version = 3
             CLF = Classifier()
             ModelDef  = Models()
             ModelDef.def_models('random_forest')
             
             
             CLF.setModel(ModelDef)
             CLF.setPca('')
             print "version:", version
             print CLF.modelname
             print CLF.feature_used
             
             
             #trainData = '../data/train/kddtrainall_v3_pct_part1.csv'
             #trainData = '../data/train/kddtrainall_v1_trans.csv'
             trainData = '../../data/train/train_onehot_v2.csv'
             #trainData = '../data/train/kddtrain_ext_low.csv'
             testData = '../../data/test/test_onehot_v2.csv'
             #trainData = '../data/train/kddtrainall_v9_drop_ext_v0.csv'
             #testData = '../data/test/kddtestall_v9_drop_ext_v0.csv'
             print "traindata from: ", trainData
             print "testdata from: ", testData
             
             submitfile = '../../data/pred/pred_'+ CLF.modelname + '_v' + str(version) + '.csv'
             cvfile = '../../data/cv/cv_' + CLF.modelname + '_v' + str(version) + '.csv'
             logfile = '../../data/cvlog/log_' + CLF.modelname + '_v' + str(version) + '.txt'
             
             writer=csv.writer(open(logfile,'wb'))
             writer.writerow([CLF.modelname])
             writer.writerow(':::::::::::::::::::::::::')
             writer.writerow(CLF.feature_used)
             writer.writerow(':::::::::::::::::::::::::')
             writer.writerow(CLF.params_best.keys())
             writer.writerow(CLF.params_best.values())
             writer.writerow(':::::::::::::::::::::::::')
             writer.writerow([str(CLF.model)])
             writer.writerow(':::::::::::::::::::::::::')
             writer.writerow([trainData])
             
             
             CLF.loadDataSet(trainData)
             #CLF.feature_selection()
             #CLF.getFeatureImportance()
             #CLF.loadTestSet(testData)
             CLF.modelValArt('',cvfile)
             #CLF.modelValArtLazy('',cvfile)
             #CLF.modelTrain(submitfile)
             #CLF.modelVal(cvfile)
