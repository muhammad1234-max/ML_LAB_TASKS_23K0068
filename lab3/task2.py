#knn from scratch on Iris dataset

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris #random dataset 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

#train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#chi-Squared Distance given in the lab manual that miss gave
def chi2_distance(x1, x2):
    return np.sum((x1 - x2) ** 2 / (x1 + x2 + 1e-10))

#knn from scratch
class KNN_Scratch:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        preds = [self._predict(x) for x in X]
        return np.array(preds)
    
    def _predict(self, x):
        distances = [chi2_distance(x, x_train) for x_train in self.X_train]
        k_idx = np.argsort(distances)[:self.k]
        k_neighbor_labels = [self.y_train[i] for i in k_idx]
        return max(set(k_neighbor_labels), key=k_neighbor_labels.count)

#train & predict
knn = KNN_Scratch(k=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

#accuracy & confusion matrix
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
