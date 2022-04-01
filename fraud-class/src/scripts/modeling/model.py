# ---
# jupyter:
#   jupytext:
#     formats: ipynb,../scripts/modeling//py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3.8.0 ('fraud-class')
#     language: python
#     name: python380jvsc74a57bd03cd04a71416ab130df52c6ad253f6c01cfe1f5da6ec3d93fcc0cee3c0dbab0b4
# ---

# +
import numpy as np 
import pandas as pd 
import datasist as ds
import datasist.project as dp
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, train_test_split
# -

#retrieve data from the processed folder
train = dp.get_data("train_proc.csv", method='csv')
test = dp.get_data("test_proc.csv", method='csv')
labels = dp.get_data("train_labels.csv", method='csv')

# +
X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.3, random_state=2)

rf_model = RandomForestClassifier(n_estimators=100,random_state=232)
lg_model = LogisticRegression(max_iter=100, random_state=2, solver='lbfgs')

# +
lg_model.fit(X_train, y_train)
pred = lg_model.predict(X_test)

#Get report from true and predicted values
ds.model.get_classification_report(y_test, pred)

# +
rf_model.fit(X_train, y_train)
pred = rf_model.predict(X_test)

#Get report from true and predicted values
ds.model.get_classification_report(y_test, pred)
# -

ds.model.train_classifier(train_data=train, target=target, model=rf_model,
                                                     cross_validate=True, cv=3)

feats = train.columns
ds.model.plot_feature_importance(estimator=rf_model, col_names=feats)


