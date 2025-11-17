import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
    mean_absolute_error, mean_squared_error
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping


print("\n================ DATASET 1: WINE QUALITY (CLASSIFICATION) ================\n")
df1 = pd.read_csv("WineQT.csv")

df1.drop(columns=["Id"], inplace=True)

y1 = df1["quality"]
X1 = df1.drop(columns=["quality"])

X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y1, test_size=0.25, random_state=42, stratify=y1
)

scaler1 = StandardScaler()
X1_train_s = scaler1.fit_transform(X1_train)
X1_test_s = scaler1.transform(X1_test)

model1 = Sequential([
    Input(shape=(X1_train_s.shape[1],)),
    Dense(128, activation="relu"),
    Dense(64, activation="tanh"),
    Dense(32, activation="relu"),
    Dense(1, activation="linear")   # regression for quality score
])

model1.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

es = EarlyStopping(patience=5, restore_best_weights=True)

model1.fit(
    X1_train_s, y1_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[es],
    verbose=2
)

y1_pred_reg = model1.predict(X1_test_s).ravel()

y1_pred = np.round(y1_pred_reg).astype(int)

#metrics
acc1 = accuracy_score(y1_test, y1_pred)
prec1 = precision_score(y1_test, y1_pred, average='macro', zero_division=0)
rec1 = recall_score(y1_test, y1_pred, average='macro', zero_division=0)
f1_1 = f1_score(y1_test, y1_pred, average='macro', zero_division=0)

print("Accuracy :", acc1)
print("Precision:", prec1)
print("Recall   :", rec1)
print("F1 Score :", f1_1)


cm = confusion_matrix(y1_test, y1_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Wine Quality")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



print("\n================ DATASET 2: HOUSE PRICE (REGRESSION) ================\n")

df2 = pd.read_csv("dataset2.csv")

y2 = df2["price"]
X2 = df2.drop(columns=["price"])

categorical_cols = ["date", "street", "city", "statezip", "country"]
X2 = pd.get_dummies(X2, columns=categorical_cols, drop_first=True)

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.25, random_state=42
)

scaler2 = StandardScaler()
X2_train_s = scaler2.fit_transform(X2_train)
X2_test_s = scaler2.transform(X2_test)

model2 = Sequential([
    Input(shape=(X2_train_s.shape[1],)),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(1, activation="linear")
])

model2.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

es2 = EarlyStopping(patience=5, restore_best_weights=True)

model2.fit(
    X2_train_s, y2_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[es2],
    verbose=2
)

y2_pred = model2.predict(X2_test_s).ravel()

#metric
mae2 = mean_absolute_error(y2_test, y2_pred)
rmse2 = np.sqrt(mean_squared_error(y2_test, y2_pred))

print("MAE :", mae2)
print("RMSE:", rmse2)

#sample predictions
print("\nSample Predictions (Actual vs Predicted):")
for actual, pred in list(zip(y2_test[:10], y2_pred[:10])):
    print(f"Actual: {actual}, Predicted: {pred}")
