import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.pyplot import figure
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import hamming_loss
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

# load the dataset and drop useless columns
print('Dataset loading...')
x = pd.read_csv("data.csv")
y = pd.read_csv("labels.csv")
# drop the first column which only contains strings
x = x.drop(x.columns[0], axis=1)
# drop first column (contains only indexes)
y = y.drop(y.columns[0], axis=1)
print('Done.')

# split data into training and testing set
print('Reducing and splitting..')

# normalization
x = preprocessing.normalize(x)

# label encoding
le = preprocessing.LabelEncoder()
Y1 = y.apply(le.fit_transform)
y = le.fit_transform(Y1)  # complete label encoded array

# splitting
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42, shuffle=True)

print('Ready.')

''' 
Defining first (not tuned) Decision Tree, called model_1.
'''

tree1 = DecisionTreeClassifier(random_state=42)
tree1.fit(x_train, y_train)

# Accuracy and Hamming Loss for the first tree
y_pred = tree1.predict(x_test)
acc1 = accuracy_score(y_test, y_pred)
print("Accuracy for a bad tree classifier model_1: {}".format(acc1))
hamming1 = hamming_loss(y_test, y_pred)
print("Hamming Loss for model_1: {}".format(hamming1))

# Confusion matrix for the first tree
cm = confusion_matrix(y_test, y_pred)
print(cm)
class_names = ['BRCA', 'COAD', 'KIRC', 'LUAD', 'PRAD']
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax=ax, cmap='Greens')
# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Not Tuned Tree Confusion Matrix')
ax.xaxis.set_ticklabels(class_names)
ax.yaxis.set_ticklabels(class_names)
ax.set_xlim(0, 5)
ax.set_ylim(5, 0)

plt.show()

''' 
Defining second tree, called model_2. Tree parameters have been tuned using Grid Search.
'''

tree2 = DecisionTreeClassifier(criterion='gini', max_depth=5, max_leaf_nodes=7, random_state=42)

# Cross-validation scores for model_2
score = cross_val_score(tree2, x_train, y_train, cv=10, scoring='accuracy')
print("Cv score for model_2:\n", score)

# Test scores for model_2

tree2.fit(x_train, y_train)
y_pred = tree2.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
class_names = ['BRCA', 'COAD', 'KIRC', 'LUAD', 'PRAD']
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax=ax, cmap='Greens')
# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Tuned Decision Tree Confusion Matrix')
ax.xaxis.set_ticklabels(class_names)
ax.yaxis.set_ticklabels(class_names)
ax.set_xlim(0, 5)
ax.set_ylim(5, 0)

plt.show()

# Accuracy for model_2 for the test set
tree_val_acc = accuracy_score(y_test, y_pred)
print("Accuracy for best Tree Classifier Model: {}".format(tree_val_acc))
# Hamming Loss for model_2 for the test set
tree_val_hamming = hamming_loss(y_test, y_pred)
print("Hamming Loss for best Tree Classifier Model: {}".format(tree_val_hamming))

# Performance Evaluation: Hamming Loss for each possible tree depth

figure(num=None, figsize=(6, 6), facecolor='w', edgecolor='k')

train_errors = list()
test_errors = list()
# new classifier
tree_clf = DecisionTreeClassifier()
# array that store the possible depths
depths = np.linspace(2, 50)
depths = depths.astype(int)
# train one tree for each possible depth
for depth in depths:
    tree_clf.set_params(max_depth=depth)
    tree_clf.fit(x_train, y_train)
    y_pred_train = tree_clf.predict(x_train)
    train_errors.append(hamming_loss(y_train, y_pred_train))
    y_pred = tree_clf.predict(x_test)
    test_errors.append(hamming_loss(y_test, y_pred))

plt.xlabel('Max Depth')
plt.ylabel('Hamming loss')
plt.title("Max depth vs. Hamming loss")

plt.plot(depths, test_errors, 'o-', color='blue', label='test_errors')
plt.plot(depths, train_errors, '^-', color='green', label='train_errors')
plt.legend()

plt.axis([0, 50, 0.0, 1])
plt.show()
