import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

#Try to import XGBoost; fallback to sklearn's GradientBoosting if not available
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    from sklearn.ensemble import GradientBoostingClassifier
    XGB_AVAILABLE = False


df = pd.read_csv("heart.csv")    
print("Initial shape:", df.shape)
print(df.head())


# 2) Quick EDA - structure & basic summaries
print("\nColumns and dtypes:\n", df.dtypes)
print("\nBasic stats:\n", df.describe(include='all').T)


if 'target' not in df.columns:
    raise ValueError("No 'target' column found. Rename your label column to 'target' or edit the script.")


print("\nClass counts (target):")
print(df['target'].value_counts())
plt.figure(figsize=(5,3))
sns.countplot(x='target', data=df)
plt.title("Target class distribution")
plt.show()


# If classes are very imbalanced we would consider resampling methods (oversample/SMOTE/undersample).
# For moderately balanced datasets standard classifiers often still work but monitor metrics other than accuracy.


print("\nMissing values per column:")
print(df.isnull().sum())


dupes = df.duplicated().sum()
print("\nNumber of duplicate rows:", dupes)


categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']  
print("\nAssumed categorical columns:", categorical_cols)

# If any of these are not in df, remove them from the list
categorical_cols = [c for c in categorical_cols if c in df.columns]


if dupes > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print("Dropped duplicates. New shape:", df.shape)

# 4b: Missing values handling
# - For numeric columns: use median imputation (robust to outliers)
# - For categorical columns: use mode (most frequent)
# These choices are simple, unbiased, reproducible and often effective for baseline models.

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c != 'target']

cat_cols = categorical_cols

num_imputer = SimpleImputer(strategy='median')
df[num_cols] = num_imputer.fit_transform(df[num_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
if len(cat_cols) > 0:
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

print("\nMissing values after imputation:\n", df.isnull().sum())


df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=True)  # drop_first avoids multicollinearity
print("\nShape after encoding:", df_enc.shape)


continuous_cols = [c for c in df_enc.columns if c not in ['target'] and (df_enc[c].dtype.kind in 'biufc')]
original_numeric = ['age','trestbps','chol','thalac','oldpeak','cs']  # 'cs' maybe numeric; include if exists
original_numeric = [c for c in original_numeric if c in df_enc.columns]

print("\nNumeric columns considered for scaling:", original_numeric)

# Compute skewness
skewness = df_enc[original_numeric].skew().round(3)
print("\nSkewness of numeric columns:\n", skewness)


skew_threshold = 1.0
cols_high_skew = skewness[abs(skewness) > skew_threshold].index.tolist()
print("\nColumns with high skewness (>|1|):", cols_high_skew)


sc_df = df_enc.copy()

if len(cols_high_skew) > 0:
    rs = RobustScaler()
    sc_df[cols_high_skew] = rs.fit_transform(sc_df[cols_high_skew])

cols_standard = [c for c in original_numeric if c not in cols_high_skew]
if len(cols_standard) > 0:
    ss = StandardScaler()
    sc_df[cols_standard] = ss.fit_transform(sc_df[cols_standard])

print("\nSample after scaling (first 5 rows):")
print(sc_df[original_numeric].head())


X = sc_df.drop('target', axis=1)
y = sc_df['target']

# 80/20 split
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)


X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.3, random_state=0, stratify=y_train_full
)

print("\nFinal sample counts:")
print("Train:", X_train.shape[0])
print("Validation:", X_val.shape[0])
print("Test:", X_test.shape[0])

# Explanation 
# - Validation set is used to tune hyperparameters (choose model, K, depth, etc.) without touching the final test set.
# - Test set is only used once to estimate final generalization performance.


models = {}

# Random Forest
models['RandomForest'] = RandomForestClassifier(n_estimators=100, random_state=0)

# XGBoost or fallback
if XGB_AVAILABLE:
    models['XGBoost'] = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=0, verbosity=0)
else:
    models['GradientBoosting'] = GradientBoostingClassifier(random_state=0)

# AdaBoost with DecisionTree base estimator
base_dt = DecisionTreeClassifier(max_depth=3, random_state=0)
models['AdaBoost'] = AdaBoostClassifier(base_estimator=base_dt, n_estimators=100, random_state=0)

# Fit each model on X_train and evaluate on train and test
results = []
for name, model in models.items():
    print(f"\nTraining {name} ...")
    model.fit(X_train, y_train)
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"{name} -> Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")
    print("Classification report (Test):")
    print(classification_report(y_test, y_test_pred))

    # optional confusion matrix plot
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} Confusion Matrix (Test)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    results.append({
        'Model': name,
        'Train_Acc': train_acc,
        'Val_Acc': val_acc,
        'Test_Acc': test_acc
    })


results_df = pd.DataFrame(results).sort_values(by='Test_Acc', ascending=False).reset_index(drop=True)
print("\nModel comparison (sorted by Test Acc):")
print(results_df)

# Short analysis guidance 
# - Compare Train vs Test accuracy to check overfitting:
#    * If Train >> Test -> overfitting.
#    * If both low -> underfitting.
# - Use Validation accuracy to decide hyperparameters (e.g., n_estimators, max_depth).
# - Prefer models with higher Test accuracy and stable Val/Test performance.
