import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#load dataset
df = pd.read_csv("CarPrice_Assignment.csv")

#EDA
# Step 2: EDA
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

numeric_df = df.select_dtypes(include=[np.number])

plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap (Numeric Only)")
plt.show()


# Fill missing values if any
df = df.dropna()

# Convert categorical data (brand, model, etc.)
df = pd.get_dummies(df, drop_first=True)

#split dataset
X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

#predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

#evaluation Metrics
print("Training RMSE:", np.sqrt(mean_squared_error(y_train, y_train_pred)))
print("Testing RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))
print("R² Score:", r2_score(y_test, y_test_pred))

#plot Results
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_test_pred, color='blue')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Car Prices")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red')
plt.show()
