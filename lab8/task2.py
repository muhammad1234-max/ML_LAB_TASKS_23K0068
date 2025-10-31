import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

#load dataset
df = pd.read_csv("electricity_bill_dataset.csv")   # <-- replace with your actual filename
print(" Dataset Loaded Successfully\n")
print(df.head())

#identify categorical columns
categorical_cols = ['Month', 'City', 'Company']

#convert categorical columns into dummy variables
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

#ensure all values are numeric and handle missing values
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna()

#define X (independent variables) and y (dependent variable)
X = df.drop('ElectricityBill', axis=1)
y = df['ElectricityBill']

#add constant for intercept term
X_const = sm.add_constant(X)

#convert to float explicitly (prevents dtype object error)
X_const = X_const.astype(float)
y = y.astype(float)

#fit OLS Linear Regression model
model = sm.OLS(y, X_const).fit()
print("\n📊 MODEL SUMMARY:\n")
print(model.summary())


#linearity Check
numeric_features = ['Fan', 'Refrigerator', 'AirConditioner', 
                    'Television', 'Monitor', 'MotorPump', 
                    'MonthlyHours', 'TariffRate']
sns.pairplot(df, x_vars=numeric_features, y_vars='ElectricityBill',
             kind='reg', plot_kws={'line_kws': {'color': 'red'}})
plt.suptitle("Linearity Check", y=1.02)
plt.show()

#homoscedasticity (Constant Variance)
plt.scatter(model.fittedvalues, model.resid, color='purple')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Homoscedasticity Check")
plt.axhline(y=0, color='red', linestyle='--')
plt.show()

#normality of Residuals
sns.histplot(model.resid, kde=True, color='green')
plt.title("Residual Distribution (Normality Check)")
plt.show()

sm.qqplot(model.resid, line='45', fit=True)
plt.title("Q-Q Plot of Residuals")
plt.show()

#multicollinearity
plt.figure(figsize=(10, 6))
sns.heatmap(X.corr(), cmap='coolwarm', annot=False)
plt.title("Correlation Matrix for Multicollinearity Check")
plt.show()

#independence (Durbin-Watson Test)
dw = model.durbin_watson
print(f"Durbin-Watson Statistic: {dw:.2f}")

print("\n Interpretation:")
print("""
 If:
- Pairplots show roughly linear trends between predictors and ElectricityBill
- Residuals are evenly scattered around zero (no funnel shape)
- Residuals roughly follow a normal distribution
- Correlations between features < 0.8
- Durbin-Watson ≈ 2 (no autocorrelation)

→ Linear Regression is suitable for this dataset.

 Otherwise, consider transformations or a non-linear model.
""")
