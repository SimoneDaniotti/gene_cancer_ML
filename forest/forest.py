import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, hamming_loss, confusion_matrix
from sklearn.model_selection import train_test_split

# load the dataset and drop useless columns
print('Dataset loading...')
x = pd.read_csv("../data.csv")
y = pd.read_csv("../labels.csv")
# drop the first column which only contains strings
x = x.drop(x.columns[0], axis=1)
# drop first column, is only index
y = y.drop(y.columns[0], axis=1)
print('Done.')

# reducing, split data into training and testing set
print('Reducing and splitting..')
# label encoding
le = preprocessing.LabelEncoder()
Y1 = y.apply(le.fit_transform)
y = le.fit_transform(Y1)  # complete label encoded array
# normalization
x = preprocessing.normalize(x)
# splitting
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42, shuffle=True)

print('Ready.')

''' 
Random Forest model with optimal parameters found by Grid Search.
'''

rf2 = RandomForestClassifier(criterion='entropy', n_estimators=60, min_samples_split=2, min_samples_leaf=1,
                             max_leaf_nodes=50, bootstrap=True, random_state=42)

# Test scores for the optimal Random Forest

rf2.fit(x_train, y_train)
y_pred = rf2.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
class_names = ['BRCA', 'COAD', 'KIRC', 'LUAD', 'PRAD']
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax=ax, cmap='Greens')  # annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Random Forest Confusion Matrix')
ax.xaxis.set_ticklabels(class_names)
ax.yaxis.set_ticklabels(class_names)
ax.set_xlim(0, 5)
ax.set_ylim(5, 0)
plt.show()

# Accuracy for the model
rf_val_acc = accuracy_score(y_test, y_pred)
print("Accuracy for Random Forest Model: {}".format(rf_val_acc))
# Hamming Loss for the model
rf_val_hamming = hamming_loss(y_test, y_pred)
print("Hamming Loss for Random Forest Model: {}".format(rf_val_hamming))
