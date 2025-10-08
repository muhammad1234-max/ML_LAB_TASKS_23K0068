import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.model_selection import GridSearchCV








X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y

    
# Define kernel variations
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
svm_results = []

for k in kernels:
    model = SVC(kernel=k, random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    svm_results.append((k, acc))

svm_df = pd.DataFrame(svm_results, columns=['Kernel', 'Accuracy'])
print("\nSVM Accuracy per Kernel:")
print(svm_df)


param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(random_state=42), param_grid, refit=True, verbose=1, cv=5)
grid.fit(X_train, y_train)

print("\nBest Parameters:", grid.best_params_)
print("Best Accuracy:", grid.best_score_)

# Compare default vs tuned SVM
svm_default = SVC(kernel='rbf', random_state=42)
svm_default.fit(X_train, y_train)
default_acc = svm_default.score(X_test, y_test)
tuned_acc = grid.best_estimator_.score(X_test, y_test)

print(f"\nDefault RBF SVM Accuracy: {default_acc:.4f}")
print(f"Tuned RBF SVM Accuracy: {tuned_acc:.4f}")


plt.figure(figsize=(7, 5))
plt.bar(['Default', 'Tuned'], [default_acc, tuned_acc], color=['skyblue', 'orange'])
plt.title("SVM Accuracy Comparison — Default vs Tuned")
plt.ylabel("Accuracy")
plt.show()
