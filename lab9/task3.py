import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

#load iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#perceptron
per = Perceptron(max_iter=1000, random_state=0)
per.fit(X_train_scaled, y_train)
y_pred_per = per.predict(X_test_scaled)

# Metrics
per_accuracy = accuracy_score(y_test, y_pred_per)
per_precision = precision_score(y_test, y_pred_per, average='macro')
per_recall = recall_score(y_test, y_pred_per, average='macro')
per_f1 = f1_score(y_test, y_pred_per, average='macro')

#logistic regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)

# Metrics
lr_accuracy = accuracy_score(y_test, y_pred_lr)
lr_precision = precision_score(y_test, y_pred_lr, average='macro')
lr_recall = recall_score(y_test, y_pred_lr, average='macro')
lr_f1 = f1_score(y_test, y_pred_lr, average='macro')

#comparison

print(">>> Perceptron")
print("Accuracy:  ", per_accuracy)
print("Precision: ", per_precision)
print("Recall:    ", per_recall)
print("F1 Score:  ", per_f1)
print("\nClassification Report:\n", classification_report(y_test, y_pred_per))


print(">>> Logistic Regression")
print("Accuracy:  ", lr_accuracy)
print("Precision: ", lr_precision)
print("Recall:    ", lr_recall)
print("F1 Score:  ", lr_f1)
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr))
