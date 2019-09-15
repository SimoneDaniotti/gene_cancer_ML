import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import preprocessing, decomposition, model_selection, metrics, svm

# loading
data = r'dataset.csv'
labels = r'lab.csv'
X = pd.read_csv(data)
X = X.drop(X.columns[0], axis=1)
x = pd.DataFrame(X).to_numpy() # complete examples array
Y = pd.read_csv(labels)
Y = Y.drop(Y.columns[0], axis=1)

# one hot encoding
le = preprocessing.LabelEncoder()
Y1 = Y.apply(le.fit_transform)
y = le.fit_transform(Y1) # complete label encoded array

# feature selection through PCA
pca = decomposition.PCA(n_components=700)
x = pca.fit_transform(x)

# normalization
x = preprocessing.normalize(x)

# splitting data in training and test
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.15, random_state=42)


# parameters for grid search
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
              {'C': [1, 10, 100, 1000], 'kernel': ['poly'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
              {'C': [1, 10, 100, 1000], 'kernel': ['sigmoid'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]

clf = svm.SVC()
clf.fit(x_train, y_train)
grid_search = model_selection.GridSearchCV(estimator=clf, param_grid=parameters, scoring='accuracy', cv=5, n_jobs=-1)
grid_search.fit(x_train, y_train)
accuracy = grid_search.best_score_
best_params = grid_search.best_params_


''' 
A best accuracy of 0.9970588235294118
Was given by these parameters {'C': 10, 'gamma': 0.1, 'kernel': 'linear'}
'''

# training step
best_clf = svm.SVC(C=10, gamma=0.1, kernel='linear', random_state=42)
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
