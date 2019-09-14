import pandas as pd
import matplotlib.pyplot as plt
import os.path
import numpy as np


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

from sklearn.ensemble import RandomForestClassifier

# Define the model. Set random_state to 1
rf_model = RandomForestClassifier(n_estimators=2,random_state=42)

# fit your model
rf_model.fit(x_train,y_train)
y_pred=rf_model.predict(x_test)

from sklearn.metrics import accuracy_score
rf_val_acc=accuracy_score(y_test, y_pred)

print("Accuracy for Random Forest Model: {}".format(rf_val_acc))

from sklearn.metrics import hamming_loss
rf_val_hamming=hamming_loss(y_test, y_pred)

print("Hamming Loss for Random Forest Model: {}".format(rf_val_hamming))

########################################################################################
#                    Performance Evaluation : errors vs # estimators
########################################################################################
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from sklearn.metrics import hamming_loss

from sklearn import linear_model



figure(num=None, figsize=(6,6), facecolor='w', edgecolor='k')

train_errors = list()
test_errors = list()

rf=RandomForestClassifier()

estimators = np.linspace(2,50)
estimators= estimators.astype(int)

for estimator in estimators:
    rf.set_params(n_estimators=estimator)
    rf.fit(x_train, y_train)
    y_pred_train = rf.predict(x_train)
    train_errors.append(hamming_loss(y_train, y_pred_train))
    y_pred = rf.predict(x_test)
    test_errors.append(hamming_loss(y_test, y_pred))

plt.xlabel('# estimators')
plt.ylabel('hamming loss')
plt.title("# estimators vs. hamming loss")

plt.plot(estimators, test_errors, 'o-', color='blue',label='test_errors')
plt.plot(estimators, train_errors, '^-', color='green',label='train_errors')

plt.legend()

plt.axis([0, 50, 0.0, 1])
plt.show()

########################################################################################
#                    Performance Evaluation: accuracy
########################################################################################
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


figure(num=None, figsize=(6,6), facecolor='w', edgecolor='k')

train_errors = list()
test_errors = list()
rf= RandomForestClassifier()
n_features_rf = np.linspace(2,50)
n_features_rf= n_features_rf.astype(int)

for feature in n_features_rf:
    rf.set_params(max_leaf_nodes=feature)
    rf.fit(x_train, y_train)
    train_errors.append(rf.score(x_train, y_train))
    test_errors.append(rf.score(x_test, y_test))

plt.xlabel('Number of Features')
plt.ylabel('Accuracy')
plt.title("Number of Features vs. Accuracy")

plt.plot(n_features_rf, test_errors, 'o-', color='blue',label='test_errors')
plt.plot(n_features_rf, train_errors, '^-', color='green',label='train_errors')
plt.legend()

plt.axis([0, 50, 0.5, 1])
plt.show()

########################################################################################
#                    Confusion Matrix
########################################################################################

class_names = y_train['Class'].unique()

'''accuracy is generally not the preferred performance measure for classifiers, 
especially when you are dealing with skewed datasets (i.e., when some classes are much more frequent than others).'''

# Define the model. Set random_state to 1
rf = RandomForestClassifier(random_state=42,n_estimators=2)


from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(rf, x_train, y_train, cv=3)

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

rf.fit(x_train,y_train)
y_pred= rf.predict(x_test)

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

########################################################################################
#                   Model with hyperp. found by gridsearch
########################################################################################

rf2=RandomForestClassifier(criterion='gini', 
                           max_depth=5,
                        max_leaf_nodes=5,
                            bootstrap=True,
                        random_state=42)

y_train_pred = cross_val_predict(rf2, x_train, y_train, cv=3)

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

rf2.fit(x_train,y_train)
y_pred= rf2.predict(x_test)

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
rf_val_acc=accuracy_score(y_test, y_pred)

print("Accuracy for Random Forest Model: {}".format(rf_val_acc))

from sklearn.metrics import hamming_loss
rf_val_hamming=hamming_loss(y_test, y_pred)

print("Hamming Loss for Random Forest Model: {}".format(rf_val_hamming))
