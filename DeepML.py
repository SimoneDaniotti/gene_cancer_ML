import keras
import pandas as pd
from sklearn import preprocessing, model_selection, decomposition, utils
import numpy as np
import matplotlib as mpl

# loading
data = r'dataset.csv'
labels = r'lab.csv'
X = pd.read_csv(data)
X = X.drop(X.columns[0], axis=1)
x = pd.DataFrame(X).to_numpy()
Y = pd.read_csv(labels)
Y = Y.drop(Y.columns[0], axis=1)

# one hot encoding
le = preprocessing.LabelEncoder()
Y2 = Y.apply(le.fit_transform)
enc = preprocessing.OneHotEncoder()
enc.fit(Y2)
y = enc.transform(Y2).toarray()

# feature selection
pca = decomposition.PCA(n_components=700)
x = pca.fit_transform(x)

# normalization
x = keras.utils.normalize(x, axis=1)
# dataset division
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.15, random_state=42)
y_integers = np.argmax(y, axis=1)
class_weights = utils.compute_class_weight('balanced', np.unique(y_integers), y_integers)
d_class_weights = dict(enumerate(class_weights))


# model
model = keras.models.Sequential()
model.add(keras.layers.Dense(50, activation=keras.activations.relu))
model.add(keras.layers.Dropout(rate=0.2, seed=42))
model.add(keras.layers.Dense(25, activation=keras.activations.relu))
model.add(keras.layers.Dropout(rate=0.2, seed=42))
model.add(keras.layers.Dense(10, activation=keras.activations.relu))
model.add(keras.layers.Dropout(rate=0.2, seed=42))
model.add(keras.layers.Dense(5, activation=keras.activations.softmax))
model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=16, validation_split=0.2, shuffle=True,
          class_weight=d_class_weights, epochs=100, verbose=2)
# validation loss and validation accuracy
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print("test loss is", test_loss)
print("test accuracy is", test_acc)
