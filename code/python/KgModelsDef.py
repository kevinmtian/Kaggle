
from itertools import product

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from nolearn.dbn import DBN
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA

class Models:
       def __init__(self):
           self.modelname = []
           self.model_best = []
           self.model = []
           self.params_best = []
           self.params_cv = []
           self.feature_used = []
       
       def def_feature_all(self):
           self.feature_all = ['T1_V1', 'T1_V2', 'T1_V3', 'T1_V10', 'T1_V13',
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


       def def_models(self, modelname):
           self.modelname = modelname
           if modelname == 'random_forest':
                self.def_random_forest()

           elif modelname == 'dbn':
                self.def_dbn()

           else:
               print "Model is Not Definied!"

       def def_random_forest(self):
           self.params_cv = {'max_features': [40],
                        'max_depth': [80],
                        'min_samples_split': [5],
                        'min_samples_leaf': [60],
                        'n_estimators': [1000]}
           
           self.params_best = {'max_features': 100,
                        'max_depth': 30,
                        'min_samples_split': 5,
                        'min_samples_leaf': 60,
                        'n_estimators': 1000}
           self.model = RandomForestRegressor()
           self.model_best = RandomForestRegressor(max_features = self.params_best['max_features'],
                                                    max_depth = self.params_best['max_depth'],
                                                    min_samples_split = self.params_best['min_samples_split'],
                                                    min_samples_leaf = self.params_best['min_samples_leaf'],
                                                    n_estimators=self.params_best['n_estimators'])
           
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
       


       def def_dbn(self):
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
           self.params_cv = {'layer_sizes' : [[len(self.feature_used),100,200,1]],
                             'scales' : [0.3],
                             'output_act_funct': ['Softmax'],
                             'use_re_lu': [False],
                             'learn_rates': [0.05],
                             'learn_rate_decays': [0.9],
                             'learn_rate_minimums':[0.001],
                             'momentum': [0.9],
                             'l2_costs': [0.0001],
                             'dropouts': [0,0.1],
                             'epochs': [200],
                             'minibatch_size': [64]}
           
           #### when doing (artificial CV): tune the params_best! not the params_cv!!!
           self.params_best = {'layer_sizes' : [len(self.feature_used),100,200,1],
                             'scales' : 1.0,
                             'output_act_funct': 'Linear',
                             'use_re_lu': False,
                             'learn_rates': 0.01,
                             'learn_rate_decays': 0.9,
                             'learn_rate_minimums': 0.0001,
                             'momentum': 0.9,
                             'l2_costs': 0.0001,
                             'dropouts': 0.0,
                             'epochs': 200,
                             'minibatch_size':64}
           self.model = DBN()
           self.model_best = DBN(layer_sizes = self.params_best['layer_sizes'],
                                 scales = self.params_best['scales'],
                                 output_act_funct=self.params_best['output_act_funct'],
                                 use_re_lu=self.params_best['use_re_lu'],
                                 learn_rates=self.params_best['learn_rates'],
                                 learn_rate_decays=self.params_best['learn_rate_decays'],
                                 learn_rate_minimums=self.params_best['learn_rate_minimums'],
                                 momentum=self.params_best['momentum'],
                                 l2_costs=self.params_best['l2_costs'],
                                 dropouts=self.params_best['dropouts'],
                                 epochs=self.params_best['epochs'],
                                 minibatch_size=self.params_best['minibatch_size'],verbose=1)




