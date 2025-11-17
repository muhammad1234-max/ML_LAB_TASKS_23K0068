import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("heart.csv")

#features and target
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#logistic Regression with L1 penalty
log_l1 = LogisticRegression(
    penalty='l1',
    solver='saga',
    max_iter=5000
)
log_l1.fit(X_train_scaled, y_train)
pred_l1 = log_l1.predict(X_test_scaled)
acc_l1 = accuracy_score(y_test, pred_l1)

#logistic Regression with L2 penalty
log_l2 = LogisticRegression(
    penalty='l2',
    solver='lbfgs',
    max_iter=5000
)
log_l2.fit(X_train_scaled, y_train)
pred_l2 = log_l2.predict(X_test_scaled)
acc_l2 = accuracy_score(y_test, pred_l2)


# Logistic Regression with ElasticNet penalty
log_en = LogisticRegression(
    penalty='elasticnet',
    solver='saga',
    l1_ratio=0.5,      # 50% L1 + 50% L2
    max_iter=5000
)
log_en.fit(X_train_scaled, y_train)
pred_en = log_en.predict(X_test_scaled)
acc_en = accuracy_score(y_test, pred_en)

#showing results
print("Accuracy (L1 Penalty):       ", acc_l1)
print("Accuracy (L2 Penalty):       ", acc_l2)
print("Accuracy (Elastic Net):      ", acc_en)
