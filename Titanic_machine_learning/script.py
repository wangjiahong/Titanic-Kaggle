from pandas                   import DataFrame, get_dummies, read_csv
from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search      import GridSearchCV
from sklearn.ensemble         import RandomForestClassifier


def clean(df):
    df               = df.join(get_dummies(df['Embarked'], prefix='Embarked'))
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['Fare'].fillna(df['Fare'].mean(), inplace=True)
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Sex'].replace({'male': 0, 'female': 1}, inplace=True)
    df.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

    return df


#if __name__ == '__main__':
data_train       = clean(read_csv('D:/git_repository/Kaggle-titanic---Jiahong/input/train.csv'))
X_train          = data_train.ix[:, 2:]
y_train          = data_train['Survived']

estimator    = RandomForestClassifier(n_estimators = 300).fit(X_train, y_train)

estimator

data_test        = clean(read_csv('D:/git_repository/Kaggle-titanic---Jiahong/input/test.csv'))
X_test           = data_test.ix[:, 1:]
y_test           = estimator.predict(X_test)
y_test
DataFrame(
        {'PassengerId': data_test['PassengerId'], 'Survived': y_test}
    ).to_csv('D:/git_repository/Kaggle-titanic---Jiahong/input/ random forest by kon -submission.csv', index=False)