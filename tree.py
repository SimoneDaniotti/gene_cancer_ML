import pandas as pd
import matplotlib.pyplot as plt
import os.path
import numpy as np

from sklearn.model_selection import cross_val_predict

print('Setting..')
# load the dataset
x_train = pd.read_csv("x_train.csv")
y_train = pd.read_csv("y_train.csv")
x_test = pd.read_csv("x_test.csv")
y_test = pd.read_csv("y_test.csv")

# drop the first column which only contains strings
x_train = x_train.drop(x_train.columns[x_train.columns.str.contains('unnamed', case=False)], axis=1)
x_test = x_test.drop(x_test.columns[x_test.columns.str.contains('unnamed', case=False)], axis=1)

#drop first column, is only index
y_train = y_train.drop(y_train.columns[0], axis=1)
y_test = y_test.drop(y_test.columns[0], axis=1)

print('ready.')

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import hamming_loss

from sklearn import linear_model

#model_1 definition
tree1 = DecisionTreeClassifier(random_state=42,max_depth=3)
tree1.fit(x_train, y_train)
y_pred = tree1.predict(x_test)

from sklearn.metrics import accuracy_score
tree_val_acc=accuracy_score(y_test, y_pred)

print("Accuracy for Random Forest Model: {}".format(tree_val_acc))

class_names = y_train['Class'].unique()

'''accuracy is generally not the preferred performance measure for classifiers, 
especially when you are dealing with skewed datasets (i.e., when some classes are much more frequent than others).'''

# Define the model. Set random_state to 1
tree2 = DecisionTreeClassifier(random_state=42,max_depth=3)


from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(tree2, x_train, y_train, cv=3)

'''Just like the cross_val_score() function, cross_val_predict() performs K-fold cross-validation, 
but instead of returning the evaluation scores, it returns the predic‐ tions made on each test fold. 
This means that you get a clean prediction for each instance in the training set 
(“clean” meaning that the prediction is made by a model that never saw the data during training).'''


import seaborn as sns
import matplotlib.pyplot as plt     

from sklearn.metrics import confusion_matrix

cm =confusion_matrix(y_train, y_train_pred)

print(cm)

ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax,cmap='Greens') #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(class_names)
ax.yaxis.set_ticklabels(class_names)
ax.set_xlim(0,5)
ax.set_ylim(5,0)

plt.show()


tree4=DecisionTreeClassifier(class_weight=None,
                                              criterion='gini', max_depth=5,
                                              max_leaf_nodes=10,
                                              min_samples_leaf=1,
                                              min_samples_split=10,
                                              random_state=42)

y_train_pred = cross_val_predict(tree4, x_train, y_train, cv=3)
cm =confusion_matrix(y_train, y_train_pred)
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax,cmap='Greens') #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(class_names)
ax.yaxis.set_ticklabels(class_names)
ax.set_xlim(0,5)
ax.set_ylim(5,0)

plt.show()

tree4.fit(x_train,y_train)
y_pred= tree4.predict(x_test)

cm =confusion_matrix(y_test, y_pred)

print(cm)

ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax,cmap='Greens') #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(class_names)
ax.yaxis.set_ticklabels(class_names)
ax.set_xlim(0,5)
ax.set_ylim(5,0)

plt.show()

from sklearn.metrics import accuracy_score
tree_val_acc=accuracy_score(y_test, y_pred)

print("Accuracy for Random Forest Model: {}".format(tree_val_acc))
from sklearn.metrics import hamming_loss
tree_val_hamming=hamming_loss(y_test, y_pred)

print("Hamming Loss for Random Forest Model: {}".format(tree_val_hamming))

tree5=DecisionTreeClassifier(class_weight=None,
                                              criterion='gini', max_depth=5,
                                              max_leaf_nodes=10,
                                              min_samples_leaf=1,
                                              min_samples_split=10,
                                              random_state=42)

from sklearn.model_selection import cross_val_score

# Multiply by -1 since sklearn calculates *negative* MAE
score = cross_val_score(tree5, x_train, y_train,
                              cv=5,
                              scoring='accuracy')

print("jaccard cv score:\n", score)

