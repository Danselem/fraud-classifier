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


# -

def get_classification_report(y_train=None, prediction=None, show_roc_plot=True, save_plot=False):
    '''
    Generates performance report for a classification problem.

    Parameters:
    ------------------
    y_train: Array, series, list.

        The truth/ground value from the train data set.
    
    prediction: Array, series, list.

        The predicted value by a trained model.

    show_roc_plot: Bool, default True.

        Show the model ROC curve.

    save_plot: Bool, default True.

        Save the plot to the current working directory.

    '''
    acc = accuracy_score(y_train, prediction)
    f1 = f1_score(y_train, prediction)
    precision = precision_score(y_train, prediction)
    recall = recall_score(y_train, prediction)
    confusion_mat = confusion_matrix(y_train, prediction)

    print("Accuracy is ", round(acc * 100))
    print("F1 score is ", round(f1 * 100))
    print("Precision is ", round(precision * 100))
    print("Recall is ", round(recall * 100))
    print("*" * 100)
    print("confusion Matrix")
    print('                 Score positive    Score negative')
    print('Actual positive    %6d' % confusion_mat[0,0] + '             %5d' % confusion_mat[0,1])
    print('Actual negative    %6d' % confusion_mat[1,0] + '             %5d' % confusion_mat[1,1])
    print('')

    if show_roc_plot:        
        plot_auc(y_train, prediction)

        if save_plot:
            plt.savefig("roc_plot.png")


# +
rf_model.fit(X_train, y_train)
pred = rf_model.predict(X_test)

#Get report from true and predicted values
ds.model.get_classification_report(y_test, pred)
# -

ds.model.train_classifier(X_train=train, y_train=labels, estimator=rf_model,
                                                     cross_validate=True, cv=5)

feats = train.columns




def plot_feature_importance(estimator=None, col_names=None):
    '''
    Plots the feature importance from a trained scikit learn estimator
    as a bar chart.

    Parameters:
    -----------
        estimator: scikit  learn estimator.

            Model that has been fit and contains the feature_importance_ attribute.

        col_names: list

            The names of the columns. Must map unto feature importance array.

    Returns:
    --------
        Matplotlib figure showing feature importances
    '''
    if estimator is None:
        raise ValueError("estimator: Expecting an estimator that implements the fit api, got None")
    if col_names is None:
        raise ValueError("col_names: Expecting a list of column names, got 'None'")
    
    if len(col_names) != len(estimator.feature_importances_):
        raise ValueError("col_names: Lenght of col_names must match lenght of feature importances")

    imps = estimator.feature_importances_
    feats_imp = pd.DataFrame({"features": col_names, "importance": imps}).sort_values(by='importance', ascending=False)
    sns.barplot(x='features', y='importance', data=feats_imp)
    plt.xticks(rotation=90)
    plt.title("Feature importance plot")
    plt.show()


plot_feature_importance(estimator=rf_model, col_names=feats)

dp.save_model(rf_model, name='rf_model_n10')


# pickling the model
import pickle
pickle_out = open("../../outputs/models/rf_model.pkl", "wb")
pickle.dump(rf_model, pickle_out)
pickle_out.close()