# Scikit-learn provides ML algorithms including classification, regression, clustering, and preprocessing
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# Example showing labeled (supervised) vs unlabeled (unsupervised) data
supervised_data = pd.DataFrame({
    "Feature1": [1.1, 2.3, 3.2, 4.1],
    "Feature2": [2.0, 3.5, 1.8, 4.0],
    "Label": ["A", "A", "B", "B"]
})
print("Supervised data (Label provided):\n", supervised_data)

unsupervised_data = supervised_data.drop("Label", axis=1)
print("\nUnsupervised data (Label not provided):\n", unsupervised_data)


import numpy as np
from scipy.spatial import distance

# Two sample points
x = np.array([2, 3])
y = np.array([5, 7])

# 1️⃣ Euclidean Distance
euclidean = np.linalg.norm(x - y)
print("Euclidean Distance:", euclidean)

# 2️⃣ Manhattan Distance
manhattan = distance.cityblock(x, y)
print("Manhattan Distance:", manhattan)

# 3️⃣ Minkowski Distance
minkowski = distance.minkowski(x, y, p=3)
print("Minkowski Distance (p=3):", minkowski)

# 4️⃣ Hamming Distance (for categorical/binary)
x_bin = np.array([1, 0, 1, 1])
y_bin = np.array([1, 1, 0, 1])
hamming = distance.hamming(x_bin, y_bin)
print("Hamming Distance:", hamming)

# 5️⃣ Cosine Distance (for similarity-based problems)
cosine = distance.cosine(x, y)
print("Cosine Distance:", cosine)


# Assume Occupancy_train.txt and Occupancy_test.txt exist
train_data = pd.read_csv("Occupancy_train.txt")
test_data = pd.read_csv("Occupancy_test.txt")

# Selecting features and target
X_train = train_data[["Humidity", "Light", "HumidityRatio"]]
y_train = train_data["Occupancy"]
X_test = test_data[["Humidity", "Light", "HumidityRatio"]]
y_test = test_data["Occupancy"]

# Running KNN for K = 1 to 10
accuracies = []

for k in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"K = {k} --> Accuracy = {acc:.4f}")

best_k = np.argmax(accuracies) + 1
print(f"\n✅ Highest Accuracy = {max(accuracies):.4f} at K = {best_k}")


# Visualizing Training vs Testing Accuracy for different K values
train_acc, test_acc = [], []

for k in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_acc.append(knn.score(X_train, y_train))
    test_acc.append(knn.score(X_test, y_test))

plt.plot(range(1, 20), train_acc, label='Training Accuracy')
plt.plot(range(1, 20), test_acc, label='Testing Accuracy')
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("Bias-Variance Tradeoff in KNN")
plt.legend()
plt.show()


# Square root method
import math

k_sqrt = int(math.sqrt(len(X_train)))
if k_sqrt % 2 == 0:
    k_sqrt += 1
print(f"Recommended K (square root rule): {k_sqrt}")

# Experimentation: plot accuracy vs K
plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), accuracies, marker='o')
plt.title("K vs Accuracy")
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.show()


#knn from scratch (using chi sqaured method)
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, accuracy_score

# Chi-Squared distance function
def chi_squared_distance(x1, x2):
    return np.sum(((x1 - x2)**2) / (x1 + x2 + 1e-10))

# Custom KNN implementation
class CustomKNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict_one(self, x):
        distances = [chi_squared_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)[0][0]
        return most_common

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

# Testing on Iris Dataset
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = CustomKNN(k=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


#eda and feature scaling before KNN
# Load dataset
data = pd.read_csv("your_dataset.csv")

# Check missing values, duplicates, and categorical columns
print("Missing values:\n", data.isnull().sum())
print("Duplicate records:", data.duplicated().sum())
print("Categorical columns:", data.select_dtypes(include='object').columns.tolist())

# Handle missing & duplicate records
data.drop_duplicates(inplace=True)
data.fillna(data.mean(numeric_only=True), inplace=True)

# Check balance
sns.countplot(x="Label", data=data)
plt.title("Class Distribution")
plt.show()

# Correlation and Feature Selection
corr = data.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(data.drop("Label", axis=1))
y = data["Label"]

# Train/Test/Validation Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train_, X_val, y_train_, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=0)

print("Training samples:", len(X_train_))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))


#compare metrics
metrics = ['euclidean', 'manhattan', 'minkowski']
results = []

for metric in metrics:
    knn = KNeighborsClassifier(n_neighbors=5, metric=metric)
    knn.fit(X_train_, y_train_)
    train_acc = knn.score(X_train_, y_train_)
    test_acc = knn.score(X_test, y_test)
    results.append((metric, train_acc, test_acc))

# Display results
df_results = pd.DataFrame(results, columns=["Metric", "Train Accuracy", "Test Accuracy"])
print(df_results)


#summary visualization of metric comparison
sns.barplot(x="Metric", y="Test Accuracy", data=df_results)
plt.title("Comparison of KNN Distance Metrics")
plt.show()


# TASK 1:
# Occupancy dataset contains four attributes i-e "Humidity, Light, CO2 and Humidity ratio".
#  Apply KNN to find if occupancy is possible or not (0 or 1) based on "Humidity,
# Light and Humidity Ratio" only. Train on "Occupancy_train.txt" and Test on
# "Occupancy_test.txt". You need to do the following then :
#  Run this KNN Algorithm for n_neighbors (K) from 1 to 10. You will get 10
# different accuracies. Print all the accuracies. Then print the highest accuracy and
# also the value of K at which you got the highest accuracy.
# 7

# Machine Learning Lab Lab Manual – 03

# TASK 2 :
# Now instead of using built-in library, write your own code for kNN classifier from
# scratch. Run on iris dataset. Use 80/20 split. Print accuracy and confusion matrix at
# the end. You must use the following chi squared distance function :

# TASK 3 :
# Download the dataset
#  Perform EDA
#  Check the dataset is balance or not (using target variable “Label”)
#  Check whether there is any empty records, categorical feature, duplicate records, if yes
# then handle this and give a brief explanation why you have chosen this technique in a
# text cell or “jupyter/colab”
#  Check the correlation of your dataset and perform feature selection using Pearson
# Correlation
#  Analyze your dataset and think if feature scaling is required or not? If yes then apply
# any scaling technique based on your distribution.
#  Split your dataset in training , testing and validation. Train split will be 80% and test will
# be 20% . In validation split your training samples will be 70% and validation set will be
# 30%. Briefly describe why we use validation set in a text cell. Declare Random_state=0
#  Apply KNN and check model training and testing accuracy.
#  Compare the accuracies by trying different metrics, combine all the training and testing
# accuracies of Euclidean, Manhattan etc. to compare their performance. Make a critical
# analysis what you have observed and where we have used different metrics?
#  Apply KNN and check model training and testing accuracy.
