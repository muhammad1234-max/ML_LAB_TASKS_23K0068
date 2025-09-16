import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv("cancer patient data sets.csv")

print("Shape:", df.shape)
print("\nInfo:")
print(df.info())
print("\nSummary:")
print(df.describe())
print("\nTarget variable distribution:")
print(df["Level"].value_counts())

sns.countplot(x="Level", data=df)
plt.title("Target Class Distribution")
plt.show()


print("\nMissing values per column:")
print(df.isnull().sum())

for col in df.columns:
    if df[col].dtype == "object":
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].mean(), inplace=True)


print("\nDuplicate Rows:", df.duplicated().sum())
df.drop_duplicates(inplace=True)

df["Level"] = df["Level"].map({"Low": 0, "Medium": 1, "High": 2})


plt.figure(figsize=(12,6))
sns.heatmap(df.corr(numeric_only=True), annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

X = df.drop(columns=["Patient Id", "Level"])
y = df["Level"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# First split:Train 80%, Test 20%
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

# Second split:Train (70%), Validation (30%) from training set
X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=0)

print("Train size:", X_train_sub.shape, "Validation size:", X_val.shape, "Test size:", X_test.shape)

dt = DecisionTreeClassifier(random_state=0)
dt.fit(X_train_sub, y_train_sub)

# Training Accuracy
train_acc = accuracy_score(y_train_sub, dt.predict(X_train_sub))
# Validation Accuracy
val_acc = accuracy_score(y_val, dt.predict(X_val))
# Testing Accuracy
test_acc = accuracy_score(y_test, dt.predict(X_test))
print("\nDecision Tree Performance:")
print("Training Accuracy:", train_acc)
print("Validation Accuracy:", val_acc)
print("Testing Accuracy:", test_acc)
print("\nClassification Report (Test Data):")
print(classification_report(y_test, dt.predict(X_test)))
