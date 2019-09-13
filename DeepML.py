import keras
import pandas as pd
from sklearn import preprocessing, model_selection, decomposition, utils, metrics
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# loading
data = r'dataset.csv'
labels = r'lab.csv'
X = pd.read_csv(data)
X = X.drop(X.columns[0], axis=1)
x = pd.DataFrame(X).to_numpy() # complete array of examples
Y = pd.read_csv(labels)
Y = Y.drop(Y.columns[0], axis=1)

# one hot encoding
le = preprocessing.LabelEncoder()
Y2 = Y.apply(le.fit_transform)
enc = preprocessing.OneHotEncoder()
enc.fit(Y2)
y = enc.transform(Y2).toarray() # complete array of labels

# feature selection through PCA
pca = decomposition.PCA(n_components=700)
x = pca.fit_transform(x)

# normalization
x = keras.utils.normalize(x, axis=1)

# dataset division
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.15, random_state=42)
y_integers = np.argmax(y, axis=1)
class_weights = utils.compute_class_weight('balanced', np.unique(y_integers), y_integers)
d_class_weights = dict(enumerate(class_weights))

'''
# K-fold
k_fold = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cvscores = []
'''

# model
model = keras.models.Sequential()
model.add(keras.layers.Dense(50, activation=keras.activations.relu))
model.add(keras.layers.Dropout(rate=0.2, seed=42))
model.add(keras.layers.Dense(25, activation=keras.activations.relu))
model.add(keras.layers.Dropout(rate=0.2, seed=42))
model.add(keras.layers.Dense(10, activation=keras.activations.relu))
model.add(keras.layers.Dropout(rate=0.2, seed=42))
model.add(keras.layers.Dense(5, activation=keras.activations.softmax))
model.compile(optimizer=keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
              loss="categorical_crossentropy",
              metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=16, validation_split=0.2, shuffle=True,
          class_weight=d_class_weights, epochs=100, verbose=2)
# validation loss and validation accuracy
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print("test loss is", test_loss)
print("test accuracy is", test_acc)

# confusion matrix of test set
predict = model.predict(x_test)
p = enc.inverse_transform(predict)
r = enc.inverse_transform(y_test)
cm = metrics.confusion_matrix(r, p)

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


#  - 0s - loss: 0.0505 - acc: 0.9835 - val_loss: 0.0067 - val_acc: 1.0000       SGD (lr=0.01, decay=1e-6)
#  - 0s - loss: 0.0471 - acc: 0.9816 - val_loss: 1.5934e-04 - val_acc: 1.0000   SGD (lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#  - 0s - loss: 1.5334 - acc: 0.4449 - val_loss: 1.5320 - val_acc: 0.6471       SGD(lr=0.001, decay=1e-6)
#  - 0s - loss: 0.1722 - acc: 0.9412 - val_loss: 0.0295 - val_acc: 1.0000       SGD (lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
#  - 0s - loss: 0.0223 - acc: 0.9908 - val_loss: 5.7778e-05 - val_acc: 1.0000   Adam(lr=0.001)
#  - 0s - loss: 0.0277 - acc: 0.9908 - val_loss: 3.1556e-05 - val_acc: 1.0000   Adam(lr=0.01)
#  - 0s - loss: 0.0300 - acc: 0.9908 - val_loss: 2.1453e-05 - val_acc: 1.0000   Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
