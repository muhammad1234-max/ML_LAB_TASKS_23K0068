import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("cancer patient data sets.csv")  

#EDA
print("Shape:", df.shape)
print(df.info())
print(df["Level"].value_counts())

#Missing Values Check
print("\nMissing values per column:\n", df.isnull().sum())

#Duplicates
print("\nDuplicate Rows:", df.duplicated().sum())

#correlation (numerical only)
plt.figure(figsize=(12, 6))
sns.heatmap(df.corr(numeric_only=True), annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

#encode target variable (Label/Level)
df["Level"] = df["Level"].map({"Low": 0, "Medium": 1, "High": 2})

#feature Scaling
X = df.drop(columns=["Patient Id", "Level"])
y = df["Level"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Train / Test Split (80/20) + Validation Split (70/30 of train)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)
X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=0)

print("Train size:", X_train_sub.shape, "Validation size:", X_val.shape, "Test size:", X_test.shape)

#applying KNN with different metrics
metrics = ["euclidean", "manhattan", "minkowski"]
results = []

for metric in metrics:
    knn = KNeighborsClassifier(n_neighbors=5, metric=metric)
    knn.fit(X_train_sub, y_train_sub)
    
    train_acc = accuracy_score(y_train_sub, knn.predict(X_train_sub))
    val_acc = accuracy_score(y_val, knn.predict(X_val))
    test_acc = accuracy_score(y_test, knn.predict(X_test))
    
    results.append({"Metric": metric, "Train Acc": train_acc, "Val Acc": val_acc, "Test Acc": test_acc})

results_df = pd.DataFrame(results)
results_df
