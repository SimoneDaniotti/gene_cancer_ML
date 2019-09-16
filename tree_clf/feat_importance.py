import pandas as pd
import os.path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import hamming_loss, accuracy_score, confusion_matrix
from sklearn import linear_model,decomposition,preprocessing
from sklearn.model_selection import cross_val_predict,cross_val_score,train_test_split

import seaborn as sns
from time import time
from sklearn.ensemble import RandomForestClassifier





start = time()

###########################################
# load the dataset and drop useless columns
###########################################
print('Dataset loading...')

x = pd.read_csv("../data.csv")
y = pd.read_csv("../labels.csv")

# drop the first column which only contains strings
x = x.drop(x.columns[0], axis=1)
#drop first column, is only index
y = y.drop(y.columns[0], axis=1)

xx=pd.DataFrame()
xx['gene_18746'] = x['gene_18746']
xx['gene_12983'] = x['gene_12983']

xx['gene_6875'] = x['gene_6875']

xx['gene_3523'] = x['gene_3523']
xx['gene_8594'] = x['gene_8594']
xx['gene_14460'] = x['gene_14460']

x=xx

#print(x)
##########################################
# split data into training and testing set
##########################################

print('Reducing and splitting..')




# label encoding
le = preprocessing.LabelEncoder()
Y1 = y.apply(le.fit_transform)
y = le.fit_transform(Y1) # complete label encoded array

#splitting
x_train, x_test, y_train, y_test \
    = train_test_split(x, y, test_size=0.15, random_state=42 , shuffle=True)
   #


#For example, if variable y is a binary categorical variable with values 0 and 1 and there are 25% of zeros 
#and 75% of ones, stratify=y will make sure that your random split has 25% of 0's and 75% of 1's
# feature selection through PCA

tree2 = DecisionTreeClassifier(
                                              criterion='gini', max_depth=5,
                                              max_leaf_nodes=7,
                                              random_state=42)
tree2.fit(x_train,y_train)
import shap
explainer = shap.KernelExplainer(tree2.predict_proba,x_train)
shap_values= explainer.shap_values(x_test.iloc[0,:])
shap.initjs()
shap.force_plot(explainer.expected_value[0], shap_values[0], x_test.iloc[0,:])
