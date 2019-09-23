import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn import preprocessing, decomposition, model_selection, metrics, svm, utils

# loading
X = pd.read_csv("../data.csv")
X = X.drop(X.columns[0], axis=1)
x = pd.DataFrame(X).to_numpy() # complete examples array
Y = pd.read_csv("../labels.csv")
Y = Y.drop(Y.columns[0], axis=1)

# one hot encoding
le = preprocessing.LabelEncoder()
Y1 = Y.apply(le.fit_transform)
y = le.fit_transform(Y1) # complete label encoded array

# feature selection through PCA
pca = decomposition.PCA(n_components=700)
x = pca.fit_transform(x)

# normalization. Scale input vectors individually to unit norm (vector length)
x = preprocessing.normalize(x)

# splitting data in training and test
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.15, random_state=42)

# class weight calculation for label encoding
# with balanced weights will be calculated by n_samples / (n_classes * np.bincount(y))
class_weights = utils.compute_class_weight('balanced', np.unique(y), y)
weights = dict(enumerate(class_weights))

''' 
# parameters for grid search
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf']},
              {'C': [1, 10, 100, 1000], 'kernel': ['poly']},
              {'C': [1, 10, 100, 1000], 'kernel': ['sigmoid']}]

clf = svm.SVC()
clf.fit(x_train, y_train)
grid_search = model_selection.GridSearchCV(estimator=clf, param_grid=parameters, scoring='accuracy', cv=5, n_jobs=-1)
grid_search.fit(x_train, y_train)
accuracy = grid_search.best_score_
best_params = grid_search.best_params_
print(accuracy)
print(best_params)

 
An accuracy of 0.9970588235294118 was given by the parameters {'C': 10, 'kernel': 'linear'} through grid search.
'''

# training step
best_clf = svm.SVC(C=10, kernel='linear', random_state=42, class_weight=weights, decision_function_shape='ovo')
best_clf.fit(x_train, y_train)

score = best_clf.score(x_test, y_test) # mean accuracy

# confusion matrix of test set
predict = best_clf.predict(x_test)
cm = metrics.confusion_matrix(y_test, predict)

# plot the results predictions on test set (confusion matrix)
class_names = ['BRCA', 'COAD', 'KIRC', 'LUAD', 'PRAD']
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax=ax, cmap='Greens')
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(class_names)
ax.yaxis.set_ticklabels(class_names)
ax.set_xlim(0, 5)
ax.set_ylim(5, 0)

plt.show()

