# ---
# jupyter:
#   jupytext:
#     formats: ipynb,../scripts/preparation//py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: 'Python 3.8.10 64-bit (''ztfsummer'': conda)'
#     language: python
#     name: python3810jvsc74a57bd04d2f789b78c66cbf5c93f5f135cccd838a365a4b961bce6d31ed51fa9e334273
# ---

# +
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

#Import the Datasist Library
import datasist as ds
import datasist.project as dp

# %matplotlib inline

# +
#read data from the raw data directory using datasist
train_data = dp.get_data('training.csv', loc='raw', method='csv')
test_data = dp.get_data('test.csv', loc='raw', method='csv')

#train_data = pd.read_csv('training.csv')
#test_data = pd.read_csv('test.csv')
# -

ds.structdata.describe(train_data)

ds.structdata.describe(test_data)

#Drop redundant features
ds.feature_engineering.drop_redundant(data=train_data)
ds.feature_engineering.drop_redundant(data=test_data)

cat_feats = ds.structdata.get_cat_feats(train_data)
cat_feats

ds.structdata.get_unique_counts(train_data)

train_data.drop(['CurrencyCode', 'CountryCode', 'TransactionId', 'BatchId'], axis=1, inplace=True)
test_data.drop(['CurrencyCode', 'CountryCode', 'TransactionId', 'BatchId'], axis=1, inplace=True)

train_data.shape

plt.figure(figsize=(12,7))
ds.visualizations.countplot(train_data)

plt.figure(figsize=(12,7))
sns.countplot(x='ProductId', data = train_data)
plt.xticks(rotation = 45)
plt.show()

plt.figure(figsize=(12,7))
sns.countplot(x='ProviderId', data = train_data)
plt.xticks(rotation = 45)
plt.show()

plt.figure(figsize=(12,7))
sns.countplot(x='ChannelId', data = train_data)
plt.xticks(rotation = 45)
plt.show()

# ?sns.countplot

ds.visualizations.class_count(train_data)

ds.visualizations.catbox(data=train_data, target='FraudResult', fig_size=(12,8))

ds.visualizations.histogram(train_data, fig_size=(9,6), bins=5)

ds.visualizations.boxplot(data=train_data, target='FraudResult', fig_size=(5,5))

date_feats = ds.structdata.get_date_cols(train_data)
date_feats

train_data[date_feats]

# +
num_feats = ds.structdata.get_num_feats(train_data)

ds.timeseries.timeplot(data=train_data,num_cols=num_feats, time_col='TransactionStartTime')
# -

train_data.columns

# +
train_data = ds.timeseries.extract_dates(data=train_data, date_cols=['TransactionStartTime'])
test_data = ds.timeseries.extract_dates(data=test_data, date_cols=['TransactionStartTime'])

train_data.head(2).T

# +
#perform merge 
all_data, ntrain, ntest = ds.structdata.join_train_and_test(train_data, test_data)

#Label Encode Large Categorical features
large_cats = ['AccountId', 'SubscriptionId', 'CustomerId', 'ProductId']

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
for cat in large_cats:
    all_data[cat] = lb.fit_transform(all_data[cat])

# One hot encode small categorical features
all_data = pd.get_dummies(all_data, drop_first=True)

#Get train and test set back
train = all_data[:ntrain]
test = all_data[ntrain:]

#Get target and drop it
target = train['FraudResult']
train.drop('FraudResult', axis=1, inplace=True)
test.drop('FraudResult', axis=1, inplace=True)

print("Shape of training datasets is {}".format(train.shape))
print("Shape of training target is {}".format(test.shape))
print("Shape of target is {}".format(target.shape))
# -

#export the processed data and label to the processed folder
dp.save_data(data, 'train_proc')
dp.save_data(label, 'train_labels')

# +
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, train_test_split

X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.3, random_state=2)

rf_model = RandomForestClassifier(n_estimators=100,random_state=232)
lg_model = LogisticRegression(max_iter=100, random_state=2, solver='lbfgs')

# +
lg_model.fit(X_train, y_train, )
pred = lg_model.predict(X_test)

#Get report from true and predicted values
ds.model.get_classification_report(y_test, pred)

# +
lg_model.fit(X_train, y_train, )
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
