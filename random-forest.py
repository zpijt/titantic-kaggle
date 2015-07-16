# This code is purely for my own learning benefit, following a tutorial
# by Nike Bernico from https://www.youtube.com/watch?v=0GrciaGYzV0

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score

import pandas as pd

# Good strategy for any data science competition is just to build the most basic
# implementation as quickly as possible to give yourself a benchmark to work from

x = pd.read_csv("train.csv")
y = x.pop("Survived")

# Don't think there will be much to learn from either of these variables
x.drop(['Name','Ticket'], axis=1, inplace=True)

# If you are going to run any kind of scikit learning on the data, need to make
# absolutely sure that there are no missing values which fucks things up
x["Age"].fillna(x.Age.mean(), inplace=True)

numeric_variables = list(x.dtypes[x.dtypes != "object"].index)
x[numeric_variables].head()

# Always sets inital variables to the same for simplicity
#model = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42)

# Training the model purely on the numeric_variables
#model.fit(x[numeric_variables],y)

# oob score gives the R**2 based on the oob predictions
#print(model.oob_score_)

# model.oob_prediction_ gives an array with each person's probability of survival
#y_oob = model.oob_prediction_
#print('c-stat:',roc_auc_score(y,y_oob))

# Get an ituition on the categorical variables
x[x.columns[x.dtypes == 'object']].describe()

# Suspect that the first letter of the cabin variable could be useful
def clean_cabin(x):
    try:
        return x[0]
    except TypeError:
        return "None"

# Invoke the clean cabin function on all values of a series
x["Cabin"] = x.Cabin.apply(clean_cabin)

categorical_variables = ['Sex','Cabin','Embarked']

for variable in categorical_variables:
    # Fill in missing data with the work 'Missing'
    x[variable].fillna("Missing", inplace=True)
    # Create arrat of dummies
    dummies = pd.get_dummies(x[variable], prefix=variable)
    # Update x to include dummies and drop the main variable
    x = pd.concat([x,dummies], axis=1)
    x.drop([variable], axis=1, inplace=True)

model = RandomForestRegressor(100, oob_score=True, n_jobs=-1, random_state=42)

#model.fit(x,y)

#print("C-stat:",roc_auc_score(y,model.oob_prediction_))


# Tweaking the parameters of the RF to optimize the model
# Investigate the grid search functions in scikit learn, this is custom
results1 = []
n_estimators_options = []

for trees in n_estimators_options:
    model = RandomForestRegressor(trees, oob_score=True, n_jobs=-1, random_state=42)
    model.fit(x,y)
    roc = roc_auc_score(y, model.oob_prediction_)
    print(trees,"c-stat:",roc)
    results1.append(roc)

# Rule of thumb for regression tasks, use all your variables at every split point
# Code block missing here about finding out best variables for max_features

model = RandomForestRegressor(n_estimators = 1000,
                              oob_score=True,
                              n_jobs=-1,
                              random_state=42,
                              max_features='auto',
                              min_samples_leaf=5)
model.fit(x,y)

test_df = pd.read_csv('test.csv', header=0)

test_df.drop(['Name','Ticket','Fare'], axis=1, inplace=True)

test_df["Age"].fillna(test_df.Age.mean(), inplace=True)

test_df["Cabin"] = test_df.Cabin.apply(clean_cabin)

categorical_variables = ['Sex','Cabin','Embarked']

for variable in categorical_variables:
    # Fill in missing data with the work 'Missing'
    test_df[variable].fillna("Missing", inplace=True)
    # Create arrat of dummies
    dummies = pd.get_dummies(test_df[variable], prefix=variable)
    # Update test_df to include dummies and drop the main variable
    test_df = pd.concat([test_df,dummies], axis=1)
    test_df.drop([variable], axis=1, inplace=True)

test_data = test_df.values

output = model.predict(test_data).astype(int)
predictions_file = open("myfirstforest.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
