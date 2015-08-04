import csv
import numpy as np
from numpy import *
from pandas import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import *

class FeatureEda:

    def __init__(self):
          self.feature = []
    
    def loadData(self, trainfile, testfile):
          df_train = read_csv(trainfile, header = 0)
          #df_test = read_csv(testfile, header = 0)
          self.train = df_train
          #self.test = df_test
    
    def setFeature(self):
          self.feature_used = ['T1_V1', 'T1_V2', 'T1_V3', 'T1_V4', 'T1_V5',
       'T1_V6', 'T1_V7', 'T1_V8', 'T1_V9', 'T1_V10', 'T1_V11', 'T1_V12',
       'T1_V13', 'T1_V14', 'T1_V15', 'T1_V16', 'T1_V17', 'T2_V1', 'T2_V2',
       'T2_V3', 'T2_V4', 'T2_V5', 'T2_V6', 'T2_V7', 'T2_V8', 'T2_V9',
       'T2_V10', 'T2_V11', 'T2_V12', 'T2_V13', 'T2_V14', 'T2_V15']
    
    def getCatFeat(self):
          train = self.train
          #test = self.test
          self.feature_cat = []
          for feat in self.feature_used:
              if type(train.loc[1,feat]) is str:
                 self.feature_cat = self.feature_cat + [feat]


    def getCatFeatStat(self, feature):
          df_train = self.train
          d = dict()
          feat_all = array(list(df_train[feature]))
          haz_all = array(list(df_train['Hazard']))
          feat_cats = unique(feat_all)
          for cat in feat_cats:
              d[cat] = mean(df_train.loc[df_train[feature] == cat ,'Hazard'])
          print d
          self.meandic = d
    

    def loadFeature(self, feature):
          # feature here needs to be categorical
          df_train = self.train
          #df_test = self.test
          feat_all = array(list(df_train[feature]))
          haz_all = array(list(df_train['Hazard']))
          feat_cats = unique(feat_all)
          d = dict()
          for cat in feat_cats:
              id = where(feat_all == cat)[0]
              haz = haz_all[id]
              d[cat] = haz
          self.feature = DataFrame(dict([(k,Series(v)) for k,v in d.iteritems()]))
          #print self.feature['class_0'].dropna().shape
          #print self.feature['class_1'].dropna().shape
          #print self.feature.shape
          
    def boxplotFeature(self,savefig,feature):
          fig = plt.figure(1)
          fig.suptitle(feature)
          _=self.feature.boxplot(return_type='axes')
          fig.savefig(savefig)
          plt.close(fig)


if __name__ == '__main__':

          trainfile = '../../data/train/train.csv'
          testfile = '../../data/test/test.csv'
          eda = FeatureEda()
          eda.loadData(trainfile, testfile)
          eda.setFeature()
          eda.getCatFeat()
          featureList = eda.feature_cat
          #eda.transFeature(featureList,transfile)
          count = 0
          for feature in featureList:
              print count
              eda.loadFeature(feature)
              eda.getCatFeatStat(feature)
              #feature = 'page_close'
              savefig = '../../data/eda/box_' + feature + '.png'
              savefile = '../../data/eda/mean_' + feature + '.txt'
              f = open(savefile, 'w')
              f.write(str(eda.meandic))
              f.close()
              
              eda.boxplotFeature(savefig,feature)
              count = count + 1
                  
          #count = 0
          #for feature in featureList:
          #    print count
          #    savefig = '../data/eda_train/hist_' + feature + '.png'
          #    eda.loadFeature(feature)
          #    eda.histFeature(savefig,feature)
          #    count = count + 1



