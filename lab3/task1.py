import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

train = pd.read_csv("occupancy_train.txt")
test = pd.read_csv("occupancy_test.txt")

#Features and target
X_train = train[["Humidity", "Light", "HumidityRatio"]]
y_train = train["Occupancy"]

X_test = test[["Humidity", "Light", "HumidityRatio"]]
y_test = test["Occupancy"]

accuracies = {}

#train with K = 1 to 10
for k in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies[k] = acc
    print(f"K={k} → Accuracy={acc:.4f}")

#best K
best_k = max(accuracies, key=accuracies.get)
print(f"\nHighest Accuracy: {accuracies[best_k]:.4f} at K={best_k}")
