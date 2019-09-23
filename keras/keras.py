import keras
import pandas as pd
from sklearn import preprocessing, model_selection, decomposition, utils, metrics
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# loading dataset and put it into variables (labels One Hot Encoded)
data = r'dataset.csv'
labels = r'lab.csv'
X = pd.read_csv(data)
X = X.drop(X.columns[0], axis=1)
x = pd.DataFrame(X).to_numpy()  # complete array of examples
Y = pd.read_csv(labels)
Y = Y.drop(Y.columns[0], axis=1)
# one hot encoding
le = preprocessing.LabelEncoder()
Y1 = Y.apply(le.fit_transform)
enc = preprocessing.OneHotEncoder()
y = enc.fit_transform(Y1).toarray()  # from sparse matrix to array

# feature selection through PCA
pca = decomposition.PCA(n_components=700)
x = pca.fit_transform(x)

# normalization
x = preprocessing.normalize(x)

# dataset division
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.15, random_state=42)

# class weight calculation for OHE
y_integers = np.argmax(y, axis=1) # temporary label encoded array to compute class weights
# with balanced weights will be calculated by n_samples / (n_classes * np.bincount(y))
class_weights = utils.compute_class_weight('balanced', np.unique(y_integers), y_integers)
weights = dict(enumerate(class_weights))

# model
model = keras.models.Sequential()
model.add(keras.layers.Dense(50, activation=keras.activations.relu))
model.add(keras.layers.Dropout(rate=0.2, seed=42))
model.add(keras.layers.Dense(25, activation=keras.activations.relu))
model.add(keras.layers.Dropout(rate=0.2, seed=42))
model.add(keras.layers.Dense(10, activation=keras.activations.relu))
model.add(keras.layers.Dropout(rate=0.2, seed=42))
model.add(keras.layers.Dense(5, activation=keras.activations.softmax))
# optimizer=keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
# keras.optimizers.SGD(lr=0.01, decay=0, momentum=0.9, nesterov=True
model.compile(keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
              loss='categorical_crossentropy', # loss used when targets are one hot encoded
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, epochs=300, validation_split=0.15, shuffle=True,
          class_weight=weights, verbose=2)

# validation loss and validation accuracy
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('Test loss is', test_loss)
print('Test accuracy is', test_acc)

print('The summary of the model is the following:')
print(model.summary())

# confusion matrix of test set
predict = model.predict(x_test)
p = enc.inverse_transform(predict)
r = enc.inverse_transform(y_test)
cm = metrics.confusion_matrix(r, p)

# plot the results predictions on test set (confusion matrix)
class_names = ['BRCA', 'COAD', 'KIRC', 'LUAD', 'PRAD']
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax=ax, cmap='Reds')
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Keras Confusion Matrix')
ax.xaxis.set_ticklabels(class_names)
ax.yaxis.set_ticklabels(class_names)
ax.set_xlim(0, 5)
ax.set_ylim(5, 0)

plt.show()

