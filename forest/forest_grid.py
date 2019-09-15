import pandas as pd
import os.path
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import hamming_loss, accuracy_score, confusion_matrix
from sklearn import linear_model,decomposition,preprocessing
from sklearn.model_selection import cross_val_predict,cross_val_score,GridSearchCV,train_test_split

import seaborn as sns

from time import time

###########################################
# load the dataset and drop useless columns
###########################################
print('Dataset loading...')

x = pd.read_csv("../data.csv")
y = pd.read_csv("../labels.csv")

# drop the first column which only contains strings
x = x.drop(x.columns[0], axis=1)
#drop first column, is only index
y = y.drop(y.columns[0], axis=1)

print('Done.')


##########################################
# split data into training and testing set
##########################################

print('Reducing and splitting..')

#PCA on x
pca = decomposition.PCA(n_components=700)
x = pca.fit_transform(x)

# normalization
x = preprocessing.normalize(x)

# label encoding
le = preprocessing.LabelEncoder()
Y1 = y.apply(le.fit_transform)
y = le.fit_transform(Y1) # complete label encoded array

#splitting
x_train, x_val, y_train, y_val \
    = train_test_split(x, y, test_size=0.15, random_state=42 , shuffle=True)
   #
#For example, if variable y is a binary categorical variable with values 0 and 1 and there are 25% of zeros 
#and 75% of ones, stratify=y will make sure that your random split has 25% of 0's and 75% of 1's
# feature selection through PCA

print('ready.')

#class_names = y_train['Class'].unique()

##################################################################
#model_3 definition and evaluation: k-fold cross val., confusion matrix, gridsearch
##################################################################

rf = RandomForestClassifier(random_state=42)

#function for creating better model from grid search

rf_params = {
    'n_estimators': [10, 20, 30, 40, 50,60,70,80,100],
    'max_leaf_nodes': [10,20,50, 100, 150, 200],
    'min_samples_split': [2, 3, 10],
    'min_samples_leaf': [1, 3, 10],
    'bootstrap': [True],
    'criterion': ['gini', 'entropy']
}

start=time()

grid_search = GridSearchCV(estimator=rf, param_grid=rf_params, scoring='accuracy', cv=5, n_jobs=-1)
grid_search.fit(x_train, y_train)
accuracy = grid_search.best_score_
best_params = grid_search.best_params_

print("Grid search took:", time() - start, '\n')

print("Best params GridS for random forest:", grid_search.best_params_, '\n')
print("Best accuracy:", grid_search.best_score_, '\n')



'''score = cross_val_score(rf, x_train, y_train,
                              cv=5,
                              scoring='accuracy')

print("Cv score for model_3:\n", score)

rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)

cm =confusion_matrix(y_test, y_pred)
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax,cmap='Greens') #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix for grid search for rondom forest')
ax.xaxis.set_ticklabels(class_names)
ax.yaxis.set_ticklabels(class_names)
ax.set_xlim(0,5)
ax.set_ylim(5,0)

plt.show()'''