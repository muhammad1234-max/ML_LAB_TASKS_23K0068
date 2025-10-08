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

#i loaded the dataset directly from sklearn
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print("Shape:", X.shape)
print("Missing Values:\n", X.isnull().sum().sum())

sns.countplot(x=y)
plt.title("Class Balance (0 = Malignant, 1 = Benign)")
plt.show()

corr = X.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr.iloc[:10, :10], cmap="coolwarm")
plt.title("Feature Correlation (sample)")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


knn = KNeighborsClassifier(n_neighbors=5)
svm = SVC(kernel='rbf', probability=True, random_state=42)

knn.fit(X_train, y_train)
svm.fit(X_train, y_train)

#predictions
y_pred_knn = knn.predict(X_test)
y_pred_svm = svm.predict(X_test)


print("\nKNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))

print("\nKNN Classification Report:\n", classification_report(y_test, y_pred_knn))
print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))

#confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt='d', ax=axes[0], cmap="Blues")
axes[0].set_title("KNN Confusion Matrix")
sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt='d', ax=axes[1], cmap="Greens")
axes[1].set_title("SVM Confusion Matrix")
plt.show()


y_prob_knn = knn.predict_proba(X_test)[:, 1]
y_prob_svm = svm.predict_proba(X_test)[:, 1]

fpr_knn, tpr_knn, _ = roc_curve(y_test, y_prob_knn)
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)

plt.figure(figsize=(7, 5))
plt.plot(fpr_knn, tpr_knn, label=f"KNN (AUC={auc(fpr_knn, tpr_knn):.3f})")
plt.plot(fpr_svm, tpr_svm, label=f"SVM (AUC={auc(fpr_svm, tpr_svm):.3f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — KNN vs SVM")
plt.legend()
plt.show()
