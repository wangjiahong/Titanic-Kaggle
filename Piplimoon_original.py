df = df_combo
df.head()
reset

import pandas as pd
import numpy as np
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
    df = add_family_size(df)
    df = add_deck_code_from_cabin_code(df)
    df = divide_family_size_into_3_groups(df)
    df = divide_title_into_2_groups(df)
    return df


    
    
def fill_null_embarked(df):
    # Fill NA 'Embarked' with "C"
    df["Embarked"] = df["Embarked"].fillna("C")
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
    
def add_family_size(df):
    ## Add family size
    df["Fam"] = df.Parch + df.SibSp + 1
    return df
    

def add_deck_code_from_cabin_code(df):
    # Add deck code
    df["Cabin"] = df.Cabin.fillna("UNK")
    for i in xrange(len(df.index)):
        df.ix[i, "Deck"] = df.ix[i, 'Cabin'][:1]
    return df
    
def divide_family_size_into_3_groups(df):
    # Add Family size label

    for i in xrange(len(df.index)):
        df.ix[df.Fam.isin([2,3,4]), "Fam"] = 2
        df.ix[df.Fam.isin([1,5,6,7]), "Fam"] = 1
        df.ix[df.Fam> 7, "Fam"] = 0
    return df

              
def divide_title_into_2_groups(df):
    # Add title label
    
    for i in xrange(len(df.index)):
        df.ix[df.Title.isin(["Sir","Lady"]), "Title"] = "Royalty"
        df.ix[df.Title.isin(["Dr", "Officer", "Rev"]), "Title"] = "Officer"
    return df
    

        

df_combo = makeFeatureEngineering(df_combo)



## DELETE un-used columns
df_combo.drop(["PassengerId", "Name", "Ticket", "Cabin", "Parch", "SibSp"], axis=1, inplace = True)


## Filling missing Age data
T_AgeMedians = df_combo.pivot_table('Age', index=["Title", "Sex", "Pclass"], aggfunc='median')
df_combo['Age'] = df_combo.apply( (lambda x: T_AgeMedians[x.Title, x.Sex, x.Pclass] if pd.isnull(x.Age) else x.Age), axis=1 )


dumdum = (df_combo.Embarked == "S") & (df_combo.Pclass == 3)
df_combo.fillna(df_combo[dumdum].Fare.median(), inplace = True)



#### OHE encoding nominal categorical features ###
df_combo = pd.get_dummies(df_combo)


df_train = df_combo.loc[0:len(titanic_train["Survived"])-1]
df_test = df_combo.loc[len(titanic_train["Survived"]):]
total_number_param = len(df_train.columns)
df_target = titanic_train.Survived

from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import make_pipeline

select = SelectKBest(k = 20)
clf = RandomForestClassifier(random_state = 10, warm_start = True, 
                                  n_estimators = 26,
                                  max_depth = 6, 
                                  max_features = 'sqrt')
pipeline = make_pipeline(select, clf)               
 
#select.fit(df_train, df_target)

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

         
submission.to_csv("RandomForest_v2 without ticket group.csv", index=False) 
