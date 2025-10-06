import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    from sklearn.ensemble import GradientBoostingClassifier
    XGB_AVAILABLE = False

df = pd.read_csv("heart.csv")


X = df[["restecg", "oldpeak"]]
y = df["target"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=0, stratify=y
)


dt = DecisionTreeClassifier(random_state=0)
knn = KNeighborsClassifier(n_neighbors=5)
rf = RandomForestClassifier(n_estimators=100, random_state=0)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=0) if XGB_AVAILABLE else GradientBoostingClassifier(random_state=0)


estimators = [('DecisionTree', dt), ('KNN', knn), ('RandomForest', rf), ('XGBoost', xgb)]


voting_hard = VotingClassifier(estimators=estimators, voting='hard')

voting_soft = VotingClassifier(estimators=estimators, voting='soft')


voting_hard.fit(X_train, y_train)
voting_soft.fit(X_train, y_train)

y_pred_hard = voting_hard.predict(X_test)
y_pred_soft = voting_soft.predict(X_test)

acc_hard = accuracy_score(y_test, y_pred_hard)
acc_soft = accuracy_score(y_test, y_pred_soft)

print(f"Hard Voting Accuracy: {acc_hard:.4f}")
print(f"Soft Voting Accuracy: {acc_soft:.4f}")

best_voting_type = "Soft" if acc_soft > acc_hard else "Hard"
print(f"\n✅ Best Voting Type: {best_voting_type}")

weight_options = [
    (1, 1, 1, 1),
    (2, 1, 1, 1),
    (1, 2, 1, 1),
    (1, 1, 2, 1),
    (1, 1, 1, 2),
    (2, 2, 1, 1),
    (1, 2, 2, 1),
    (1, 1, 2, 2)
]

results = []

for w in weight_options:
    vc = VotingClassifier(estimators=estimators, voting='soft', weights=w)
    vc.fit(X_train, y_train)
    y_pred = vc.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append((w, acc))

results_df = pd.DataFrame(results, columns=["Weights (DT, KNN, RF, XGB)", "Accuracy"])
results_df = results_df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
print("\nAccuracy for different weight combinations:")
print(results_df)

best_weights = results_df.iloc[0]["Weights (DT, KNN, RF, XGB)"]
print(f"\n🎯 Best Weights for Soft Voting: {best_weights}")


rf_estimators = [10, 50, 100, 200]
train_acc, test_acc = [], []

for n in rf_estimators:
    rf_temp = RandomForestClassifier(n_estimators=n, random_state=0)
    xgb_temp = XGBClassifier(n_estimators=n, eval_metric='logloss', random_state=0) if XGB_AVAILABLE else GradientBoostingClassifier(n_estimators=n, random_state=0)
    vc_temp = VotingClassifier(
        estimators=[
            ('DecisionTree', dt),
            ('KNN', knn),
            ('RandomForest', rf_temp),
            ('XGBoost', xgb_temp)
        ],
        voting='soft',
        weights=best_weights
    )
    vc_temp.fit(X_train, y_train)
    train_acc.append(vc_temp.score(X_train, y_train))
    test_acc.append(vc_temp.score(X_test, y_test))

plt.figure(figsize=(8, 5))
plt.plot(rf_estimators, train_acc, marker='o', label='Training Accuracy')
plt.plot(rf_estimators, test_acc, marker='o', label='Testing Accuracy')
plt.title("Bias–Variance Tradeoff (Voting Classifier)")
plt.xlabel("Number of Trees (Random Forest/XGBoost)")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()
