

import pandas as pd
import numpy as np
import play_a_random_music
import os
os.chdir("C:\Users\Jiahong\Documents\Titanic-Kaggle")


# Read data
titanic_train = pd.read_csv("input/train.csv", dtype={"Age": np.float64}, )
titanic_test = pd.read_csv("input/test.csv", dtype={"Age": np.float64}, )

train_set = titanic_train.drop("Survived", axis = 1)
df_combo = pd.concat((train_set, titanic_test), axis = 0, ignore_index = True)

def makeFeatureEngineering(df):  
    df = fill_null_embarked(df)

    
    df = add_title(df)
    df = simplify_title(df)
    df = divide_title_into_2_groups(df)
    
    df = add_family_size(df)
    df = divide_family_size_into_3_groups(df)
    
    df = add_deck_code_from_cabin_code(df)

        
    df = delete_not_used_columns(df)
    
    df = fill_null_age(df)
    df = fill_null_fare(df)
    
    return df

def add_title(df):
    for i in xrange(len(df)):
        df.ix[i, "Title"] = df.ix[i, "Name"].split(",")[1].split(".")[0].replace(" ", "")
    return df

    
titleDictionary = {
                        "Capt": "Officer",
                        "Col": "Officer",
                        "Major": "Officer",
                        "Jonkheer": "Sir",
                        "Don": "Sir",
                        "Sir" : "Sir",
                        "Dr": "Dr",
                        "Rev": "Rev",
                        "theCountess": "Lady",
                        "Dona": "Lady",
                        "Mme": "Mrs",
                        "Mlle": "Miss",
                        "Ms": "Mrs",
                        "Mr" : "Mr",
                        "Mrs" : "Mrs",
                        "Miss" : "Miss",
                        "Master" : "Master",
                        "Lady" : "Lady"
                        }
    
def simplify_title(df):
    for i in xrange(len(df.index)):
        df.ix[i, 'Title'] = titleDictionary[df.ix[i, 'Title']]
    return df
    
def divide_title_into_2_groups(df):
    # Add title label
    
    df.ix[df.Title.isin(["Sir","Lady"]), "Title"] = "Royalty"
    df.ix[df.Title.isin(["Dr", "Officer", "Rev"]), "Title"] = "Officer"
    return df
    
    
    
def add_family_size(df):
    ## Add family size
    df["Fam"] = df.Parch + df.SibSp + 1
    return df

def divide_family_size_into_3_groups(df):
    df.ix[df.Fam.isin([2,3,4]), "Fam"] = 2
    df.ix[df.Fam.isin([1,5,6,7]), "Fam"] = 1
    df.ix[df.Fam> 7, "Fam"] = 0
    return df
    

def add_deck_code_from_cabin_code(df):
    df["Cabin"] = df.Cabin.fillna("UNK")
    for i in xrange(len(df.index)):
        df.ix[i, "Deck"] = df.ix[i, 'Cabin'][:1]    # get "U" from "UKN"
    return df
    

              

def delete_not_used_columns(df):
    df.drop(["PassengerId", "Name", "Ticket", "Cabin", "Parch", "SibSp"], axis=1, inplace = True)
    return df

def fill_null_embarked(df):
    # Fill NA 'Embarked' with "C"
    df["Embarked"] = df["Embarked"].fillna("C")
    return df
    
    
def fill_null_age(df):
    T_AgeMedians = df.pivot_table('Age', index=["Title", "Sex", "Pclass"], aggfunc='median')
    #df_combo.pivot_table('Age', index=["Title", "Sex", "Pclass"], aggfunc=len, fill_value=0)

    df['Age'] = df.apply( (lambda x: T_AgeMedians[x.Title, x.Sex, x.Pclass] if pd.isnull(x.Age) else x.Age), axis=1 )
    return df


def fill_null_fare(df):
    dumdum = (df.Embarked == "S") & (df.Pclass == 3)
    df.Fare.fillna(df[dumdum].Fare.median(), inplace = True)
    return df
    
    
########
df = df_combo
df.Fare
df.Title
df_combo.pivot_table('Age', index=["Title", "Sex", "Pclass"], aggfunc=len, fill_value=0)
df_combo.Title
train = titanic_train

train = makeFeatureEngineering(titanic_train)
train.columns
train.groupby(['Title'])['Age'].mean()
train.groupby(['Title'])['Age'].count()

train[train.Age.isnull()]
train[train.Title == 'Master']
train.Age
titanic_train.Age.isnull()
sum(df_combo.Age == 8.05)
#########  

df.pivot_table('Age', index=["Title", "Sex", "Pclass"], aggfunc=len, fill_value=0)
df.pivot_table('Age', index=["Title", "Sex", "Pclass"], aggfunc='median')
  
df[df.Age.isnull()].Title.unique()

T_AgeMedians = df.pivot_table('Age', index=["Title", "Sex", "Pclass"], aggfunc='median')

T_count = df.pivot_table('Age', index=["Title", "Sex", "Pclass"], aggfunc=len, fill_value=0)

###########




df_combo = makeFeatureEngineering(df_combo)



#### OHE encoding nominal categorical features ###
df_combo = pd.get_dummies(df_combo)

n_train_rows = len(titanic_train["Survived"])

df_train = df_combo.loc[0:n_train_rows-1]
df_test = df_combo.loc[n_train_rows: ]
                       
df_target = titanic_train.Survived

X_train, y_train = df_train, df_target
â€?

from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
import sklearn


kbest = SelectKBest(k = 20)
randomforest = RandomForestClassifier(
                             warm_start = True,
                             
                             #max_depth = 6
                            max_features = 'sqrt'
                             )
pipeline = make_pipeline(kbest, randomforest)               

parameters = dict(
              n_estimators=[26, 50, 100], #, 200, 500, 700, 1200
               
            max_depth = [5,6,7,8],
              min_samples_split=[2,3, 4],
                min_samples_leaf = [1,2,3]
                    ) 

clf = sklearn.grid_search.GridSearchCV(randomforest, 
                                       param_grid=parameters, 
                                       cv = 10,
                                      scoring='roc_auc',
                                      verbose=2, 
                                      n_jobs = 4
                                     )
clf.fit(X_train, y_train)
pipeline.fit(X_train, y_train)
clf.best_score_
clf.best_params_
predictions = clf.predict(X_test)
predict_proba = clf.predict_proba(X_train)[:,1]
sklearn.metrics.classification_report( df_target, predictions )
sklearn.metrics.accuracy_score(df_target, predictions)
  
 
 
cv_score = cross_validation.cross_val_score(clf, df_train, df_target, cv= 10)
print("Accuracy : %.4g" % metrics.accuracy_score(df_target.values, predictions))
print("AUC Score (Train): %f" % metrics.roc_auc_score(df_target, predict_proba))
print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" \
      % (np.mean(cv_score), np.std(cv_score), np.min(cv_score),np.max(cv_score)))

 
final_pred = clf.predict(df_test)
submission = pd.DataFrame({"PassengerId": titanic_test["PassengerId"], "Survived": final_pred })


#make test to make sure my change did not affect the result:
orginal_result = pd.read_csv("RandomForest_v1.csv")
print 'The current version has %d difference with the orginal version result:'\
         %(sum(submission.Survived != orginal_result.Survived))

         
submission.to_csv("RandomForest_v4.csv", index=False) 


import play_a_random_music
play_a_random_music.playMusic()
play_a_random_music.stopMusic()


####*****************************
#*******************************************Gridiant boosting classifier
gridiantClf = GradientBoostingClassifier()

parameters = dict(learning_rate=[0.05, 0.1, 0.15],

              #n_estimators=[26, 50 , 100, 200, 400, 700],
                max_depth = [3,4],
              min_samples_split=[2, 3],
            min_samples_leaf  = [1,2]
                    ) 
    


gridiantBoosting_grid_search = sklearn.grid_search.GridSearchCV(gridiantClf, 
                                       param_grid=parameters, 
                                       cv = 10,
                                      scoring='roc_auc',
                                      verbose=3,
                                      n_jobs = 4
                                     )
gridiantBoosting_grid_search.fit(X_train, y_train)
play_a_random_music.playMusic()


gridiantBoosting_grid_search.best_score_
gridiantBoosting_grid_search.best_params_


#######
predictions = gridiantBoosting_grid_search.predict(df_train)
predict_proba = gridiantBoosting_grid_search.predict_proba(df_train)[:,1]


cv_score = cross_validation.cross_val_score(gridiantBoosting_grid_search, df_train, df_target, cv= 10)
print("Accuracy : %.4g" % metrics.accuracy_score(df_target.values, predictions))
print("AUC Score (Train): %f" % metrics.roc_auc_score(df_target, predict_proba))
print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" \
      % (np.mean(cv_score), np.std(cv_score), np.min(cv_score),np.max(cv_score)))

final_pred = gridiantBoosting_grid_search.predict(df_test)
submission = pd.DataFrame({"PassengerId": titanic_test["PassengerId"], "Survived": final_pred })

submission.to_csv("Gridiant boosting_V3_accaucy_learn rate 0.15 max depth 3, min leaf1 min split2.csv", index=False) 

#####


gridiantClf.fit(X_train, y_train)

predictions = gridiantClf.predict(df_train)
predict_proba = gridiantClf.predict_proba(df_train)[:,1]


cv_score = cross_validation.cross_val_score(gridiantClf, df_train, df_target, cv= 10)
print("Accuracy : %.4g" % metrics.accuracy_score(df_target.values, predictions))
print("AUC Score (Train): %f" % metrics.roc_auc_score(df_target, predict_proba))
print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" \
      % (np.mean(cv_score), np.std(cv_score), np.min(cv_score),np.max(cv_score)))

final_pred = pipeline.predict(df_test)
submission = pd.DataFrame({"PassengerId": titanic_test["PassengerId"], "Survived": final_pred })

submission.to_csv("Gridiant boosting_V1.csv", index=False) 

#**********************************************************************
#----------Piplimoon original version classifier------------------------------------------------------------
#**********************************************************************

#### OHE encoding nominal categorical features ###
df_combo = pd.get_dummies(df_combo)

n_train_rows = len(titanic_train["Survived"])

df_train = df_combo.loc[0:n_train_rows-1]
df_test = df_combo.loc[n_train_rows: ]
                       
df_target = titanic_train.Survived

from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import make_pipeline

kbest = SelectKBest(k = 19)
clf = RandomForestClassifier(random_state = 10,
                             warm_start = True, 
                             n_estimators = 26,
                             max_depth = 6, 
                             max_features = 'sqrt'
                             )
pipeline = make_pipeline(kbest, clf)               
 

pipeline.fit(df_train, df_target)
predictions = pipeline.predict(df_train)
predict_proba = pipeline.predict_proba(df_train)[:,1]

 
cv_score = cross_validation.cross_val_score(pipeline, df_train, df_target, cv= 10)
print("Accuracy : %.4g" % metrics.accuracy_score(df_target.values, predictions))
print("AUC Score (Train): %f" % metrics.roc_auc_score(df_target, predict_proba))
print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" \
      % (np.mean(cv_score), np.std(cv_score), np.min(cv_score),np.max(cv_score)))

 


final_pred = pipeline.predict(df_test)
submission = pd.DataFrame({"PassengerId": titanic_test["PassengerId"], "Survived": final_pred })


#make test to make sure my change did not affect the result:
orginal_result = pd.read_csv("RandomForest_v1.csv")
print 'The current version has %d difference with the orginal version result:'\
         %(sum(submission.Survived != orginal_result.Survived))


         
#Accuracy : 0.8676
#AUC Score (Train): 0.918171         
         
submission.to_csv("RandomForest_v2.csv", index=False) 
