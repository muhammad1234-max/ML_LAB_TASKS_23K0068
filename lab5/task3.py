import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("heart.csv")


X = df[["restecg", "chol"]]   # Independent variables
y = df["target"]               # Dependent variable


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=0, stratify=y
)


rf = RandomForestClassifier(n_estimators=100, random_state=0)
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=3),
    n_estimators=100,
    random_state=0
)


rf.fit(X_train, y_train)
ada.fit(X_train, y_train)

# Predictions
y_pred_rf = rf.predict(X_test)
y_pred_ada = ada.predict(X_test)

# Accuracy scores
rf_train_acc = rf.score(X_train, y_train)
rf_test_acc = rf.score(X_test, y_test)
ada_train_acc = ada.score(X_train, y_train)
ada_test_acc = ada.score(X_test, y_test)

print("Random Forest -> Train Accuracy:", round(rf_train_acc, 4), "| Test Accuracy:", round(rf_test_acc, 4))
print("AdaBoost      -> Train Accuracy:", round(ada_train_acc, 4), "| Test Accuracy:", round(ada_test_acc, 4))


voting = VotingClassifier(
    estimators=[('RandomForest', rf), ('AdaBoost', ada)],
    voting='soft'
)

voting.fit(X_train, y_train)
y_pred_vote = voting.predict(X_test)

vote_train_acc = voting.score(X_train, y_train)
vote_test_acc = voting.score(X_test, y_test)

print("\nVoting Ensemble -> Train Accuracy:", round(vote_train_acc, 4), "| Test Accuracy:", round(vote_test_acc, 4))


#prepare data for plotting
models = ['Random Forest', 'AdaBoost', 'Voting Ensemble']
train_accs = [rf_train_acc, ada_train_acc, vote_train_acc]
test_accs  = [rf_test_acc,  ada_test_acc,  vote_test_acc]

#bar plot
plt.figure(figsize=(8, 5))
x = np.arange(len(models))
bar_width = 0.35

plt.bar(x - bar_width/2, train_accs, width=bar_width, label='Training Accuracy', color='skyblue')
plt.bar(x + bar_width/2, test_accs, width=bar_width, label='Testing Accuracy', color='lightgreen')

for i in range(len(models)):
    plt.text(x[i]-0.25, train_accs[i]+0.01, f"{train_accs[i]:.2f}")
    plt.text(x[i]+0.05, test_accs[i]+0.01, f"{test_accs[i]:.2f}")

plt.xticks(x, models)
plt.ylabel('Accuracy')
plt.title('Training & Testing Accuracy Comparison\n(RandomForest, AdaBoost, and Voting Ensemble)')
plt.legend()
plt.tight_layout()
plt.show()


n_estimators_range = [10, 30, 50, 70, 100, 150, 200]
ensemble_train_acc, ensemble_test_acc = [], []

for n in n_estimators_range:
    ada_temp = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3),
        n_estimators=n,
        random_state=0
    )
    rf_temp = RandomForestClassifier(n_estimators=100, random_state=0)
    voting_temp = VotingClassifier(estimators=[('RF', rf_temp), ('ADA', ada_temp)], voting='soft')
    voting_temp.fit(X_train, y_train)
    ensemble_train_acc.append(voting_temp.score(X_train, y_train))
    ensemble_test_acc.append(voting_temp.score(X_test, y_test))

#plot the trend
plt.figure(figsize=(8, 5))
plt.plot(n_estimators_range, ensemble_train_acc, marker='o', label='Training Accuracy')
plt.plot(n_estimators_range, ensemble_test_acc, marker='o', label='Testing Accuracy')
plt.xlabel('Number of AdaBoost Estimators')
plt.ylabel('Accuracy')
plt.title('Voting Ensemble Accuracy vs AdaBoost Complexity (Bias–Variance Trend)')
plt.legend()
plt.grid(True)
plt.show()
