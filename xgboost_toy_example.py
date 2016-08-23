# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 23:15:22 2016

@author: Jiahong
"""

import xgboost as xgb
# read in data
dtrain = xgb.DMatrix('D:/git_repository/xgboost/demo/data/agaricus.txt.train')
dtest = xgb.DMatrix('D:/git_repository/xgboost/demo/data/agaricus.txt.test')
# specify parameters via map
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
num_round = 200
bst = xgb.train(param, dtrain, num_round)
# make prediction
preds = bst.predict(dtest)
preds



import xgboost as xgb
# read in data
dtrain = xgb.DMatrix('demo/data/agaricus.txt.train')
dtest = xgb.DMatrix('demo/data/agaricus.txt.test')
# specify parameters via map
param = {'booster': 'dart',
         'max_depth': 5, 'learning_rate': 0.1,
         'objective': 'binary:logistic', 'silent': True,
         'sample_type': 'uniform',
         'normalize_type': 'tree',
         'rate_drop': 0.1,
         'skip_drop': 0.5}
num_round = 500
bst = xgb.train(param, dtrain, num_round)
# make prediction
# ntree_limit must not be 0
preds = bst.predict(dtest, ntree_limit=num_round)
preds