import pandas as pd
import os.path
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import hamming_loss, accuracy_score, confusion_matrix
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict,cross_val_score

import seaborn as sns
from time import time



print('Setting..')
start = time()

################################
# load the dataset
################################
x_train = pd.read_csv("x_train.csv")
y_train = pd.read_csv("y_train.csv")
x_test = pd.read_csv("x_test.csv")
y_test = pd.read_csv("y_test.csv")

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
print("Preparing datasets took %.2f seconds." % (time() - start))

#class names for plotting confusion matrix
class_names = y_train['Class'].unique()





###########################
#model_1 definition and evaluation
###########################

tree1 = DecisionTreeClassifier(random_state=42,max_depth=3)

start = time()
tree1.fit(x_train, y_train)
print("Training model_1 took %.2f seconds." % (time() - start))

y_pred = tree1.predict(x_test)

acc1=accuracy_score(y_test, y_pred)

print("Accuracy for a bad tree classifier model_1: {}".format(acc1))

hamming1=hamming_loss(y_test, y_pred)

print("Hamming Loss for model_1: {}".format(hamming1))

'''accuracy is generally not the preferred performance measure for classifiers, 
especially when you are dealing with skewed datasets (i.e., when some classes are much more frequent than others).'''
cm =confusion_matrix(y_test, y_pred)

print(cm)

ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax,cmap='Greens') #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix for model_1')
ax.xaxis.set_ticklabels(class_names)
ax.yaxis.set_ticklabels(class_names)
ax.set_xlim(0,5)
ax.set_ylim(5,0)

plt.show()





##################################################################
#model_2 definition and evaluation: k-fold cross val. and confusion matrix
##################################################################

#parameters found with scripts for hyperparam. evaluation

tree2 = DecisionTreeClassifier(class_weight=None,
                                              criterion='gini', max_depth=5,
                                              max_leaf_nodes=10,
                                              min_samples_leaf=1,
                                              min_samples_split=10,
                                              random_state=42)
######
#Validation scores
######
score = cross_val_score(tree2, x_train, y_train,
                              cv=5,
                              scoring='accuracy')

print("Cv score for model_2:\n", score)

y_train_pred = cross_val_predict(tree2, x_train, y_train, cv=3)

'''Just like the cross_val_score() function, cross_val_predict() performs K-fold cross-validation, 
but instead of returning the evaluation scores, it returns the predictions made on each test fold. 
This means that you get a clean prediction for each instance in the training set 
(“clean” meaning that the prediction is made by a model that never saw the data during training).'''

cm =confusion_matrix(y_train, y_train_pred)

print(cm)

ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax,cmap='Greens') #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Validation Confusion Matrix for model_2')
ax.xaxis.set_ticklabels(class_names)
ax.yaxis.set_ticklabels(class_names)
ax.set_xlim(0,5)
ax.set_ylim(5,0)

plt.show()


######
#Test scores
######
start = time()
tree2.fit(x_train,y_train)
print("Training model_2 took %.2f seconds." % (time() - start))

y_pred= tree2.predict(x_test)

cm =confusion_matrix(y_test, y_pred)

print(cm)

ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax,cmap='Greens') #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Test Confusion Matrix for model_2')
ax.xaxis.set_ticklabels(class_names)
ax.yaxis.set_ticklabels(class_names)
ax.set_xlim(0,5)
ax.set_ylim(5,0)

plt.show()

tree_val_acc=accuracy_score(y_test, y_pred)

print("Accuracy for Random Forest Model: {}".format(tree_val_acc))


tree_val_hamming=hamming_loss(y_test, y_pred)

print("Hamming Loss for Random Forest Model: {}".format(tree_val_hamming))


#########
#Printing tree shape
#########

from sklearn.tree import export_graphviz
export_graphviz(
            tree2,
            out_file="tree2.dot",
            feature_names=x_train.columns.tolist(),
            class_names=y_train.Class.tolist(),
            rounded=True,
            filled=True
        )
# Convert to png
from subprocess import call
call(['dot', '-Tpng', 'tree2.dot', '-o', 'tree2.png', '-Gdpi=600'])

#####################################################################



'''#######################################################################################
#                    Performance Evaluation
########################################################################################
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import hamming_loss

from sklearn import linear_model



figure(num=None, figsize=(6,6), facecolor='w', edgecolor='k')

train_errors = list()
test_errors = list()

tree_clf = DecisionTreeClassifier()

depths = np.linspace(2,50)
depths= depths.astype(int)

for depth in depths:
    tree_clf.set_params(max_depth=depth)
    tree_clf.fit(x_train, y_train)
    y_pred_train = tree_clf.predict(x_train)
    train_errors.append(hamming_loss(y_train, y_pred_train))
    y_pred = tree_clf.predict(x_test)
    test_errors.append(hamming_loss(y_test, y_pred))

plt.xlabel('Max Depth')
plt.ylabel('hamming loss')
plt.title("Max depth vs. hamming loss")

plt.plot(depths, test_errors, 'o-', color='blue',label='test_errors')
plt.plot(depths, train_errors, '^-', color='green',label='train_errors')
#plt.plot(n_features_logit, recall_logit, 's-', color='red')
plt.legend()

plt.axis([0, 50, 0.0, 1])
plt.show()
'''