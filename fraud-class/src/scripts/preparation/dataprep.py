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
#     display_name: Python 3.8.0 ('fraud-class')
#     language: python
#     name: python380jvsc74a57bd03cd04a71416ab130df52c6ad253f6c01cfe1f5da6ec3d93fcc0cee3c0dbab0b4
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

# ## Exploratory Data Analysis

ds.structdata.describe(train_data)

train_data['TransactionStartTime'] = pd.to_datetime(train_data['TransactionStartTime'])
test_data['TransactionStartTime'] = pd.to_datetime(test_data['TransactionStartTime'])

ds.structdata.describe(test_data)

cat_feats = ds.structdata.get_cat_feats(train_data)
cat_feats

ds.structdata.get_unique_counts(train_data)

train_data.drop(['CurrencyCode', 'CountryCode', 'TransactionId', 'BatchId'], axis=1, inplace=True)
test_data.drop(['CurrencyCode', 'CountryCode', 'TransactionId', 'BatchId'], axis=1, inplace=True)

ds.visualizations.countplot(train_data, fig_size = (9,6))

num_cols = ["Amount", 'Value', 'PricingStrategy', 'FraudResult']
for col in num_cols:
    plt.style.use("seaborn")
    plt.figure(figsize=(12,4))
    plt.plot(train_data["TransactionStartTime"], train_data[col], label = col)
    plt.legend(loc = 'best')
    plt.xticks(rotation = '75')
    plt.show()

plt.style.use("seaborn")
figure, axes = plt.subplots(4, 1)
train_data.plot(subplots=True, figsize=(12, 15))
plt.xticks(rotation = '75')
plt.show()

cols = ['ProductId', 'ProviderId', 'ProductCategory', 'ChannelId']
for col in cols:
    plt.figure(figsize=(8,6))
    g = sns.countplot(x=col, data = train_data)
    plt.xticks(rotation =90)
    plt.show()

ds.visualizations.class_count(train_data)

# +
cols = ['ProductId', 'ProviderId', 'ProductCategory', 'ChannelId']

for col in cols:
    plt.figure(figsize=(7,7))
    g = sns.catplot(x=col, col='FraudResult', data=train_data,
                kind="count", sharey=False)
    g.set_xticklabels(rotation = '75')
    plt.show()

# +
num_cols = ['Amount', 'Value', 'PricingStrategy']

for col in num_cols:
    plt.figure(figsize=(6,6))
    g = sns.histplot(data = train_data, x = col, 
                    bins = 50, kde=True, palette='ocean')
    #plt.xticks(rotation = '90')
    plt.show

# +
num_cols = ['Amount', 'Value', 'PricingStrategy', 'FraudResult']

for col in num_cols:
    plt.figure(figsize=(7,7))
    sns.boxplot(x = 'FraudResult', y = col, data = train_data, palette='gist_heat')
    plt.show()
# -

sns.pairplot(data = train_data)
plt.plot()

train_data.columns


def extract_dates(data=None, date_cols=None, subset=None, drop=True):
    df = data
    for date_col in date_cols:
        #Convert date feature to Pandas DateTime
        df[date_col ]= pd.to_datetime(df[date_col])

        #specify columns to return
        dict_dates = {  "dow":  df[date_col].dt.weekday,
                        "doy":   df[date_col].dt.dayofyear,
                        "dom": df[date_col].dt.day,
                        "hr": df[date_col].dt.hour,
                        "min":   df[date_col].dt.minute,
                        "is_wkd":  df[date_col].apply(lambda x : 1 if x  in [5,6] else 0 ),
                        "wkoyr": df[date_col].dt.isocalendar().week,
                        "mth": df[date_col].dt.month,
                        "qtr":  df[date_col].dt.quarter,
                        "yr": df[date_col].dt.year
                    } 

        if subset is None:
            #return all features
            subset = ['dow', 'doy', 'dom', 'hr', 'min', 'is_wkd', 'wkoyr', 'mth', 'qtr', 'yr']
            for date_ft in subset:
                df[date_col + '_' + date_ft] = dict_dates[date_ft]
        else:
            #Return only sepcified date features
            for date_ft in subset:
                df[date_col + '_' + date_ft] = dict_dates[date_ft]
                
    #Drops original time columns from the dataset
    if drop:
        df.drop(date_cols, axis=1, inplace=True)

    return df


train_data = extract_dates(data = train_data, date_cols = ['TransactionStartTime'])
test_data = extract_dates(data = test_data, date_cols = ['TransactionStartTime'])

# +
all_data, ntrain, ntest = ds.structdata.join_train_and_test(train_data, test_data)

#Label Encode Large Categorical features
large_cats = ['AccountId', 'SubscriptionId', 'CustomerId', 'ProductId']

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
for cat in large_cats:
    all_data[cat] = lb.fit_transform(all_data[cat])
    

# One hot encode small categorical features
cols = ['ProductId', 'ProviderId', 'ProductCategory', 'ChannelId']
dummies = pd.get_dummies(all_data[cols], drop_first=True)

all_data = pd.concat([all_data.drop(cols,axis=1), dummies],axis=1)

#Get train and test set back
train = all_data[:ntrain]
test = all_data[ntrain:]

#Get target and drop it
target = train['FraudResult']
train.drop('FraudResult', axis=1, inplace=True)
test.drop('FraudResult', axis=1, inplace=True)

print("Shape of training datasets is {}".format(train.shape))
print("Shape of test datasets is {}".format(test.shape))
print("Shape of target is {}".format(target.shape))
# -

dp.save_data(train, 'train_proc')
dp.save_data(test, 'test_proc')
dp.save_data(target, 'train_labels')


