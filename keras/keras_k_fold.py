import keras
import pandas as pd
from sklearn import preprocessing, model_selection, decomposition, utils, metrics
import numpy as np


# loading
data = r'data.csv'
labels = r'labels.csv'
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
x = keras.utils.normalize(x, axis=1)

# class weight calculation for label encoding
# with balanced weights will be calculated by n_samples / (n_classes * np.bincount(y))
class_weights = utils.compute_class_weight('balanced', np.unique(y), y)
weights = dict(enumerate(class_weights))

# Stratified K-fold, the folds are made by preserving the percentage of samples for each class.
k_fold = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cvscores = []

for train, test in k_fold.split(x, y): # split indices
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
            loss='categorical_crossentropy',
            metrics=['accuracy'])
    model.fit(x[train], keras.utils.to_categorical(y[train]), batch_size=16, epochs=100,
              validation_split=0.15, shuffle=True, class_weight=weights, verbose=2)
    scores = model.evaluate(x[test], keras.utils.to_categorical(y[test]), verbose=2)
    cvscores.append(scores[1]*100)

mean = np.mean(cvscores)
std = np.std(cvscores)
print("mean", mean, "std", std)
print(cvscores)


