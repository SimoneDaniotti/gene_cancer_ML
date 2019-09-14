import pandas as pd
import os.path
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import hamming_loss, accuracy_score, confusion_matrix
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict,cross_val_score,GridSearchCV

import seaborn as sns

from time import time


print('Setting..')
################################
# load the dataset
################################
x_train = pd.read_csv("../x_train.csv")
y_train = pd.read_csv("../y_train.csv")
x_test = pd.read_csv("../x_test.csv")
y_test = pd.read_csv("../y_test.csv")

################################
# Cleaning
################################
# drop the first column which only contains strings
x_train = x_train.drop(x_train.columns[x_train.columns.str.contains('unnamed', case=False)], axis=1)
x_test = x_test.drop(x_test.columns[x_test.columns.str.contains('unnamed', case=False)], axis=1)

#drop first column, is only index
y_train = y_train.drop(y_train.columns[0], axis=1)
y_test = y_test.drop(y_test.columns[0], axis=1)

print('ready.')

class_names = y_train['Class'].unique()

##################################################################
#model_3 definition and evaluation: k-fold cross val., confusion matrix, gridsearch
##################################################################

tree3 = DecisionTreeClassifier(random_state=42)

#function for creating better model from grid search
def model_grid_params(model, params):
        new_model = GridSearchCV(estimator=model,
                                 param_grid=params, cv=5, n_jobs=-1,
                                 scoring="recall_macro")
        start = time()
        new_model.fit(x_train,y_train) #when “fitting” it on a dataset 
        #all the possible combinations of parameter values are evaluated and the best combination is retained

        print("GridSearchCV took %.2f seconds." % (time() - start))

        print(new_model, '\n')
        return new_model

tree_params = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [4,5,6,7,8,20,30]
}

tree3 = model_grid_params(tree3, tree_params)


print("Best params GridS for decision tree:", tree3.best_params_, '\n')


score = cross_val_score(tree3, x_train, y_train,
                              cv=5,
                              scoring='accuracy')

print("Cv score for model_3:\n", score)

tree3.fit(x_train,y_train)
y_pred=tree3.predict(x_test)

cm =confusion_matrix(y_test, y_pred)
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax,cmap='Greens') #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix for grid search')
ax.xaxis.set_ticklabels(class_names)
ax.yaxis.set_ticklabels(class_names)
ax.set_xlim(0,5)
ax.set_ylim(5,0)

plt.show()