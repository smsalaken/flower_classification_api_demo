import pandas as pd
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split

url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

dt = pd.read_table(url, sep = ",", header= None)
dt.columns = ['sl', 'sw', 'pl', 'pw', 'class']



iris_X = dt[['sl','sw','pl','pw']]
iris_y = dt['class']

# Split iris data in train and test data
# A random permutation, to split the data randomly

iris_X_train, iris_X_test, iris_y_train, iris_y_test = train_test_split(
    iris_X, iris_y, test_size=0.33, random_state=42)


# Create and fit a nearest-neighbor classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_y_train) 



knn.predict(iris_X_test)

import joblib
joblib.dump(knn, 'models/knn_classifier.pkl')


iris_y_test

sl = 3
sw = 2
pl = 1
pw = 12
inp = pd.DataFrame([[sl, sw, pl, pw]], columns=['sl','sw','pl','pw'])

knn.predict(inp)[0]

