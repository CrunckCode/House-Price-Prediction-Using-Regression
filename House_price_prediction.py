# -*- coding: utf-8 -*-
"""Machine Learning project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XaUTDSfNu9rVfLCxoNiw_8DzJGc8Pcod

## **HOUSING PRICE PREDICTION**
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

housing = pd.read_csv("data.csv")

housing.head()

housing.info()

housing['RM'].fillna(housing['RM'].mean(), inplace=True)

housing.info()

housing['CHAS'].value_counts()

housing.describe()

housing.hist(bins=50, figsize=(20, 15))

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

print(f'Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n')

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
  strat_train_set = housing.loc[train_index]
  strat_test_set = housing.loc[test_index]

strat_test_set['CHAS'].value_counts()

strat_train_set['CHAS'].value_counts()

corr_matrix = housing.corr()

corr_matrix['MEDV'].sort_values(ascending=False)

from pandas.plotting import scatter_matrix
attributes = ["MEDV","RM","ZN","LSTAT"]
scatter_matrix(housing[attributes], figsize =(12, 8))

housing.plot(kind="scatter", x="RM", y="MEDV", alpha=1)

housing["TAXRM"] = housing["TAX"]/housing["RM"]

housing["TAXRM"]

housing.head()

corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)

housing = strat_train_set.drop("MEDV", axis=1)
housing_labels = strat_train_set["MEDV"].copy()

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
                        ('imputer', SimpleImputer(strategy="mean")),
                        ('std_scaler', StandardScaler()),
                        
])

housing_num_tr = my_pipeline.fit_transform(housing)
housing_num_tr

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(housing_num_tr, housing_labels)

some_data = housing.iloc[:5]

some_labels = housing_labels.iloc[:5]

prepared_data = my_pipeline.transform(some_data)

model.predict(prepared_data)

list(some_labels)

from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_mse

lin_rmse = np.sqrt(lin_mse)
lin_rmse

from sklearn.tree import DecisionTreeRegressor
model1 = DecisionTreeRegressor()
model1.fit(housing_num_tr, housing_labels)

from sklearn.metrics import mean_squared_error
housing_predictions = model1.predict(housing_num_tr)
dt_mse = mean_squared_error(housing_labels, housing_predictions)
dt_mse

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model1, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)

rmse_scores

def print_scores(scores):
  print("Scores:", scores)
  print("Mean:", scores.mean())
  print("Standard deviation:", scores.std())

print_scores(rmse_scores)

from sklearn.ensemble import RandomForestRegressor
model2 = RandomForestRegressor()
model2.fit(housing_num_tr, housing_labels)

from sklearn.metrics import mean_squared_error
housing_predictions = model2.predict(housing_num_tr)
rf_mse = mean_squared_error(housing_labels, housing_predictions)
rf_mse

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model2, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)

rmse_scores

def print_scores(scores):
  print("Scores:", scores)
  print("Mean:", scores.mean())
  print("Standard deviation:", scores.std())

print_scores(rmse_scores)

X_test = strat_test_set.drop("MEDV", axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(list(final_predictions))
print("----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
print(list(Y_test))

final_rmse











