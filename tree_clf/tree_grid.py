import pandas as pd
import os.path
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from sklearn.tree import DecisionTreeClassifier
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

'''#PCA on x
pca = decomposition.PCA(n_components=700)
x = pca.fit_transform(x)'''

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


##################################################################
#model_3 definition and evaluation: k-fold cross val., confusion matrix, gridsearch
##################################################################

tree3 = DecisionTreeClassifier(random_state=42)

tree_params = {
    'criterion': ['gini','entropy'],
    'max_depth': [4,5,6,7,20,50,100],
    'max_leaf_nodes':[5,6,7,9,10,12,15,50,100],
}
start=time()
#The parameters of the estimator used to apply these methods 
#are optimized by cross-validated grid-search over a parameter grid.
grid_search = GridSearchCV(estimator=tree3, param_grid=tree_params, scoring='accuracy', cv=10, n_jobs=-1)
grid_search.fit(x_train, y_train)
accuracy = grid_search.best_score_
best_params = grid_search.best_params_

print("Grid search took:", time() - start, 'seconds \n')

print("Best params GridS for tree:", grid_search.best_params_, '\n')
print("Best accuracy:", grid_search.best_score_, '\n')

#best: 0.9808823529411764 

#{'criterion': 'gini', 'max_depth': 5, 'max_leaf_nodes': 7, 'min_samples_leaf': 1, 'min_samples_split': 2} 
'''score = cross_val_score(tree3, x_train, y_train,
                              cv=5,
                              scoring='accuracy')

print("Cv score for model_3:\n", score)

tree3.fit(x_train,y_train)
y_pred=tree3.predict(x_test)

cm =confusion_matrix(y_test, y_pred)
class_names = ['BRCA', 'COAD', 'KIRC', 'LUAD', 'PRAD']
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

plt.show()'''