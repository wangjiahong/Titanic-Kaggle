""" Writing my first randomforest code.
Author : AstroDave
Date : 23rd September 2012
Revised: 15 April 2014
please see packages.python.org/milk/randomforests.html for more

""" 
import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier
import os
import xgboost as xgb
from sklearn import cross_validation
os.chdir('D:/git_repository/Kaggle-titanic---Jiahong/')
# Data cleanup
# TRAIN DATA


train_df = pd.read_csv('input/train.csv', header=0)        # Load the train file into a dataframe
train_df
# I need to convert all strings to integer classifiers.
# I need to fill in the missing values of the data and make it complete.

# female = 0, Male = 1
train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Embarked from 'C', 'Q', 'S'
# Note this is not ideal: in translating categories to numbers, Port "2" is not 2 times greater than Port "1", etc.

# All missing Embarked -> just make them embark from most common place
if len(train_df.Embarked[ train_df.Embarked.isnull() ]) > 0:
    train_df.Embarked[ train_df.Embarked.isnull() ] = train_df.Embarked.dropna().mode().values

Ports = list(enumerate(np.unique(train_df['Embarked'])))    # determine all values of Embarked,
Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
train_df.Embarked = train_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

# All the ages with no data -> make the median of all Ages
median_age = train_df['Age'].dropna().median()
if len(train_df.Age[ train_df.Age.isnull() ]) > 0:
    train_df.loc[ (train_df.Age.isnull()), 'Age'] = median_age
train_df['familysize'] = train_df['SibSp'] + train_df['Parch']
# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 
train_df = train_df.drop(['SibSp', 'Parch'], axis=1) 


train_df['familysize']

# TEST DATA
test_df = pd.read_csv('input/test.csv', header=0)        # Load the test file into a dataframe

# I need to do the same with the test data now, so that the columns are the same as the training data
# I need to convert all strings to integer classifiers:
# female = 0, Male = 1
test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Embarked from 'C', 'Q', 'S'
# All missing Embarked -> just make them embark from most common place
if len(test_df.Embarked[ test_df.Embarked.isnull() ]) > 0:
    test_df.Embarked[ test_df.Embarked.isnull() ] = test_df.Embarked.dropna().mode().values
# Again convert all Embarked strings to int
test_df.Embarked = test_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)


# All the ages with no data -> make the median of all Ages
median_age = test_df['Age'].dropna().median()
if len(test_df.Age[ test_df.Age.isnull() ]) > 0:
    test_df.loc[ (test_df.Age.isnull()), 'Age'] = median_age

# All the missing Fares -> assume median of their respective class
if len(test_df.Fare[ test_df.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):                                              # loop 0 to 2
        median_fare[f] = test_df[ test_df.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        test_df.loc[ (test_df.Fare.isnull()) & (test_df.Pclass == f+1 ), 'Fare'] = median_fare[f]

# Collect the test data's PassengerIds before dropping it
ids = test_df['PassengerId'].values
test_df['familysize'] = test_df['SibSp'] + test_df['Parch']
# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 

# drop sibsp and Parch
test_df = test_df.drop(['SibSp', 'Parch'], axis=1) 

#test_df
# The data is now ready to go. So lets fit to the train, then predict to the test!
# Convert back to a numpy array
train_data = train_df.values
test_data = test_df.values

train_data
test_data


#1 Random forest
print 'Training...'
forest = RandomForestClassifier(n_estimators=10000)
forest = forest.fit( train_data[0::,1::], train_data[0::,0] )

print 'Predicting...'
output = forest.predict(test_data).astype(int)
output = clf.predict(test_data).astype(int)
# Random forest get result of 0.74163


X_train_cv, X_test_cv, y_train_cv, y_test_cv = cross_validation.train_test_split(train_data[0::,1::], train_data[0::,0], test_size=0.2)

clf = RandomForestClassifier(n_estimators=10000).fit(X_train_cv, y_train_cv)
clf.score(X_test_cv, y_test_cv) 


clf = xgb.XGBClassifier(max_depth=3, n_estimators=10000, learning_rate=0.05).fit(X_train_cv, y_train_cv)
clf.score(X_test_cv, y_test_cv)   


#gbm = xgb.XGBClassifier(max_depth=3, n_estimators=3000, learning_rate=0.05).fit(train_data[0::,1::], train_data[0::,0])
#predictions = gbm.predict(test_data).astype(int)
# 70% on leader board


from sklearn.grid_search import GridSearchCV
import xgboost as xgb


xgb_grid = xgb.XGBClassifier(n_estimators=1000, subsample = 0.8)

params = {
  #  'subsample': [0.8, 0.7, 0.9, 1],
    'learning_rate': [0.01, 0.05, 0.08], #, 0.1, 0.5
    'colsample_bytree': [0.7, 0.8,  0.9],  #'colsample_bytree': [0.5, 0.8,  1],
    'max_depth': [2, 3, 4],  #  'max_depth': [3, 4, 5],
}

gs = GridSearchCV(xgb_grid, params, cv=5, scoring='accuracy', n_jobs= 1)        #accuracy
gs.fit(train_data[0::,1::], train_data[0::,0])
print gs.best_params_
print gs.best_score_



test_data
predictions = gs.predict(test_data).astype(int)
predictions
output = predictions


predictions_file = open("input/10000 random forest trees with rich features.csv", "wb") 
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'
