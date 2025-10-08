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
from sklearn.feature_selection import SelectKBest, f_classif

#ANOVA Feature Selection
selector = SelectKBest(f_classif, k=10)
X_new = selector.fit_transform(X, y)

selected_features = X.columns[selector.get_support()]
print("\nTop 10 Selected Features (Table 3 Equivalent):")
print(selected_features.tolist())

#split dataset subsets
splits = [(0.6, 0.4), (0.7, 0.3), (0.8, 0.2)]
results = []

for train_ratio, test_ratio in splits:
    X_train, X_test, y_train, y_test = train_test_split(
        X_new, y, test_size=test_ratio, random_state=42, stratify=y
    )
    clf = SVC(kernel='linear', random_state=42)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    results.append({
        'Train%': int(train_ratio*100),
        'Test%': int(test_ratio*100),
        'Accuracy': round(acc, 4)
    })

table4 = pd.DataFrame(results)
print("\nTable 4 — Accuracy with Various Splits:")
print(table4)

feature_ranks = pd.Series(selector.scores_, index=X.columns).sort_values(ascending=False)
top_features = feature_ranks.head(10)
print("\nFeature Importance Scores:\n", top_features)
sns.barplot(x=top_features.values, y=top_features.index)
plt.title("Top Features by ANOVA F-score")
plt.show()
