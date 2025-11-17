import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np


df = pd.read_csv("fraud.csv")   

# Identify target column
if "Class" in df.columns:
    target_col = "Class"
elif "fraud" in df.columns:
    target_col = "fraud"
elif "isFraud" in df.columns:
    target_col = "isFraud"
else:
    target_col = df.columns[-1]   # fallback

X = df.drop(columns=[target_col])
y = df[target_col]

# One-hot encode categorical columns (if any)
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

#build MLP model
model = Sequential([
    Input(shape=(X_train_s.shape[1],)),
    Dense(128, activation="relu"),
    Dense(64, activation="tanh"),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")    # binary classification
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

#to avod overfitting we use earl ystopping
es = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

#train the model
history = model.fit(
    X_train_s, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[es],
    verbose=2
)


y_pred_prob = model.predict(X_test_s).ravel()
y_pred = (y_pred_prob >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n===== TASK 4 RESULTS =====")
print("Accuracy:  ", acc)
print("Precision: ", prec)
print("Recall:    ", rec)
print("F1 Score:  ", f1)
