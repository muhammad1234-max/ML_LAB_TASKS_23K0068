# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    VotingClassifier, 
    BaggingClassifier, 
    RandomForestClassifier, 
    AdaBoostClassifier
)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# DATA PREPARATION AND EDA (TASK 1)
# =============================================================================

def create_sample_heart_disease_dataset():
    """
    Create a sample heart disease dataset for demonstration
    Since the actual dataset isn't specified, we'll create a realistic synthetic one
    """
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic features similar to heart disease data
    age = np.random.normal(55, 10, n_samples).astype(int)
    chol = np.random.normal(240, 50, n_samples).astype(int)  # cholesterol
    restecg = np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.4, 0.1])  # resting ECG
    oldpeak = np.random.exponential(1, n_samples)  # ST depression
    thalch = np.random.normal(150, 25, n_samples).astype(int)  # max heart rate
    cp = np.random.choice([0, 1, 2, 3], n_samples)  # chest pain type
    exang = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])  # exercise induced angina
    
    # Create target variable (Label) with some relationship to features
    # Higher age, chol, oldpeak increase probability of heart disease
    disease_prob = (
        0.1 + 
        0.002 * (age - 55) + 
        0.001 * (chol - 240) + 
        0.1 * oldpeak + 
        0.05 * (thalch - 150) +
        0.1 * cp +
        0.2 * exang
    )
    disease_prob = np.clip(disease_prob, 0, 1)
    label = np.random.binomial(1, disease_prob)
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'chol': chol,
        'restecg': restecg,
        'oldpeak': oldpeak,
        'thalch': thalch,
        'cp': cp,
        'exang': exang,
        'Label': label
    })
    
    # Add some missing values and duplicates for demonstration
    df.iloc[10:15, 2] = np.nan  # restecg missing
    df.iloc[20:25, 3] = np.nan  # oldpeak missing
    
    # Add duplicate rows
    duplicates = df.iloc[50:55].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    
    return df

print("=" * 70)
print("TASK 1: COMPREHENSIVE DATA ANALYSIS AND ENSEMBLE MODELING")
print("=" * 70)

# Create and explore the dataset
df = create_sample_heart_disease_dataset()
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# 1. Perform EDA
print("\n1. EXPLORATORY DATA ANALYSIS (EDA)")
print("-" * 50)

# Basic information
print("Dataset Info:")
print(df.info())

print("\nDescriptive Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Handle missing values
print("\nHandling missing values...")
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in ['float64']:
            # Numerical columns: fill with median (robust to outliers)
            df[col].fillna(df[col].median(), inplace=True)
            print(f"Filled missing values in {col} with median: {df[col].median():.2f}")
        else:
            # Categorical columns: fill with mode
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"Filled missing values in {col} with mode: {mode_val}")

# Remove duplicates
duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate records: {duplicates}")
if duplicates > 0:
    df = df.drop_duplicates()
    print(f"Removed {duplicates} duplicate records")

print(f"Dataset shape after preprocessing: {df.shape}")

# 2. Check if dataset is balanced
print("\n2. DATASET BALANCE CHECK")
print("-" * 50)
label_counts = df['Label'].value_counts()
print("Label distribution:")
print(label_counts)
balance_ratio = label_counts[0] / label_counts[1]
print(f"Balance ratio (Class 0/Class 1): {balance_ratio:.2f}")
print(f"Dataset is {'balanced' if 0.8 <= balance_ratio <= 1.2 else 'imbalanced'}")

# 3. Check if feature scaling is required
print("\n3. FEATURE SCALING ANALYSIS")
print("-" * 50)

# Check feature distributions and ranges
numerical_features = ['age', 'chol', 'oldpeak', 'thalch']
print("Feature ranges before scaling:")
print(df[numerical_features].describe().loc[['min', 'max', 'std']])

# Apply StandardScaler (features have different scales)
print("\nApplying StandardScaler...")
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[numerical_features] = scaler.fit_transform(df[numerical_features])

print("Feature ranges after scaling:")
print(df_scaled[numerical_features].describe().loc[['min', 'max', 'std']])

# 4. Split dataset into train, test, and validation sets
print("\n4. DATA SPLITTING")
print("-" * 50)

# Prepare features and target
X = df_scaled.drop('Label', axis=1)
y = df_scaled['Label']

# First split: train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

# Second split: split train into train (70%) and validation (30%)
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, y_train, test_size=0.3, random_state=0, stratify=y_train
)

print(f"Training set shape: {X_train_final.shape}")
print(f"Validation set shape: {X_val.shape}")
print(f"Test set shape: {X_test.shape}")

print("\nWhy we use validation set:")
print("• Hyperparameter tuning without overfitting to test set")
print("• Model selection and early stopping")
print("• Monitoring model performance during training")
print("• Preventing data leakage from test set")

# =============================================================================
# ENSEMBLE METHODS IMPLEMENTATION
# =============================================================================

print("\n" + "=" * 70)
print("ENSEMBLE METHODS IMPLEMENTATION")
print("=" * 70)

# =============================================================================
# SIMPLE ENSEMBLE TECHNIQUES
# =============================================================================

print("\n1. SIMPLE ENSEMBLE TECHNIQUES")
print("-" * 50)

# Create individual models
dt_model = DecisionTreeClassifier(random_state=42)
knn_model = KNeighborsClassifier()
lr_model = LogisticRegression(random_state=42)

# Train individual models
print("Training individual models...")
dt_model.fit(X_train_final, y_train_final)
knn_model.fit(X_train_final, y_train_final)
lr_model.fit(X_train_final, y_train_final)

# Get predictions from each model
dt_pred = dt_model.predict(X_test)
knn_pred = knn_model.predict(X_test)
lr_pred = lr_model.predict(X_test)

# MAX VOTING (Manual Implementation)
print("\nMAX VOTING (Manual Implementation):")
max_voting_pred = []
for i in range(len(X_test)):
    votes = [dt_pred[i], knn_pred[i], lr_pred[i]]
    final_vote = max(set(votes), key=votes.count)  # Mode of predictions
    max_voting_pred.append(final_vote)

max_voting_accuracy = accuracy_score(y_test, max_voting_pred)
print(f"Max Voting Accuracy: {max_voting_accuracy * 100:.2f}%")

# AVERAGING (for probabilities)
print("\nAVERAGING (Probability-based):")
dt_proba = dt_model.predict_proba(X_test)
knn_proba = knn_model.predict_proba(X_test)
lr_proba = lr_model.predict_proba(X_test)

avg_proba = (dt_proba + knn_proba + lr_proba) / 3
avg_pred = np.argmax(avg_proba, axis=1)
avg_accuracy = accuracy_score(y_test, avg_pred)
print(f"Averaging Accuracy: {avg_accuracy * 100:.2f}%")

# =============================================================================
# VOTING CLASSIFIER (SKLEARN)
# =============================================================================

print("\n2. VOTING CLASSIFIER (SKLEARN IMPLEMENTATION)")
print("-" * 50)

# Define models for voting classifier
model1 = LogisticRegression(random_state=42)
model2 = DecisionTreeClassifier(random_state=42)
model3 = KNeighborsClassifier()

# HARD VOTING
print("Hard Voting Classifier:")
hard_voting = VotingClassifier(
    estimators=[('lr', model1), ('dt', model2), ('knn', model3)], 
    voting='hard'
)
hard_voting.fit(X_train_final, y_train_final)
hard_voting_score = hard_voting.score(X_test, y_test)
print(f"Hard Voting Accuracy: {hard_voting_score * 100:.2f}%")

# SOFT VOTING
print("\nSoft Voting Classifier:")
soft_voting = VotingClassifier(
    estimators=[('lr', model1), ('dt', model2), ('knn', model3)], 
    voting='soft'
)
soft_voting.fit(X_train_final, y_train_final)
soft_voting_score = soft_voting.score(X_test, y_test)
print(f"Soft Voting Accuracy: {soft_voting_score * 100:.2f}%")

# =============================================================================
# BAGGING ALGORITHMS
# =============================================================================

print("\n3. BAGGING ALGORITHMS")
print("-" * 50)

# Bagging with Decision Trees
print("Bagging Classifier (with Decision Trees):")
bagging_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=50,
    random_state=42,
    max_samples=0.8,
    max_features=0.8
)
bagging_clf.fit(X_train_final, y_train_final)
bagging_score = bagging_clf.score(X_test, y_test)
print(f"Bagging Classifier Accuracy: {bagging_score * 100:.2f}%")

# Random Forest (Specialized Bagging)
print("\nRandom Forest Classifier:")
rf_clf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10
)
rf_clf.fit(X_train_final, y_train_final)
rf_score = rf_clf.score(X_test, y_test)
print(f"Random Forest Accuracy: {rf_score * 100:.2f}%")

# =============================================================================
# BOOSTING ALGORITHMS
# =============================================================================

print("\n4. BOOSTING ALGORITHMS")
print("-" * 50)

# AdaBoost
print("AdaBoost Classifier:")
adaboost_clf = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    random_state=42,
    learning_rate=1.0
)
adaboost_clf.fit(X_train_final, y_train_final)
adaboost_score = adaboost_clf.score(X_test, y_test)
print(f"AdaBoost Accuracy: {adaboost_score * 100:.2f}%")

# XGBoost
print("\nXGBoost Classifier:")
xgb_clf = XGBClassifier(
    n_estimators=100,
    random_state=42,
    learning_rate=0.1,
    max_depth=6
)
xgb_clf.fit(X_train_final, y_train_final)
xgb_score = xgb_clf.score(X_test, y_test)
print(f"XGBoost Accuracy: {xgb_score * 100:.2f}%")

# =============================================================================
# TASK 1: COMPARE ALL THREE ALGORITHMS
# =============================================================================

print("\n" + "=" * 70)
print("TASK 1: COMPARISON OF RANDOM FOREST, XGBOOST, ADABOOST")
print("=" * 70)

# Train all three algorithms and compare their performance
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_final, y_train_final)
    
    # Training accuracy
    train_pred = model.predict(X_train_final)
    train_acc = accuracy_score(y_train_final, train_pred)
    
    # Testing accuracy
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    
    results[name] = {
        'Training Accuracy': train_acc,
        'Testing Accuracy': test_acc,
        'Model': model
    }
    
    print(f"{name} - Training Accuracy: {train_acc * 100:.2f}%")
    print(f"{name} - Testing Accuracy: {test_acc * 100:.2f}%")

# Display comparison
print("\n" + "=" * 50)
print("MODEL COMPARISON SUMMARY")
print("=" * 50)
comparison_df = pd.DataFrame(results).T
print(comparison_df[['Training Accuracy', 'Testing Accuracy']].round(4))

# =============================================================================
# TASK 2: VOTING CLASSIFIER WITH SPECIFIC FEATURES
# =============================================================================

print("\n" + "=" * 70)
print("TASK 2: VOTING CLASSIFIER WITH restEcg AND Oldpeak")
print("=" * 70)

# Extract only two features as specified
X_two_features = df_scaled[['restecg', 'oldpeak']]
y = df_scaled['Label']

# Split the data with two features
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
    X_two_features, y, test_size=0.2, random_state=0, stratify=y
)

X_train_final_2, X_val_2, y_train_final_2, y_val_2 = train_test_split(
    X_train_2, y_train_2, test_size=0.3, random_state=0, stratify=y_train_2
)

print(f"Training set shape with 2 features: {X_train_final_2.shape}")

# Define models for voting classifier
models_task2 = [
    ('dt', DecisionTreeClassifier(random_state=42)),
    ('knn', KNeighborsClassifier()),
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=50, random_state=42))
]

# Test both hard and soft voting
voting_types = ['hard', 'soft']
voting_results = {}

for voting_type in voting_types:
    print(f"\n{voting_type.upper()} VOTING with 2 features:")
    voting_clf = VotingClassifier(
        estimators=models_task2,
        voting=voting_type
    )
    voting_clf.fit(X_train_final_2, y_train_final_2)
    voting_accuracy = voting_clf.score(X_test_2, y_test_2)
    voting_results[voting_type] = voting_accuracy
    print(f"{voting_type.capitalize()} Voting Accuracy: {voting_accuracy * 100:.2f}%")

# Find best voting type
best_voting = max(voting_results, key=voting_results.get)
print(f"\nBest voting type: {best_voting} with {voting_results[best_voting] * 100:.2f}% accuracy")

# Test different weights (Manual approach)
print("\nTesting different weight combinations:")
weight_combinations = [
    [1, 1, 1, 1],  # Equal weights
    [2, 1, 3, 2],  # Higher weight for RF
    [1, 2, 2, 3],  # Higher weight for XGB
    [3, 1, 2, 2]   # Higher weight for DT
]

best_weights = None
best_accuracy = 0

for weights in weight_combinations:
    weighted_voting = VotingClassifier(
        estimators=models_task2,
        voting='soft',
        weights=weights
    )
    weighted_voting.fit(X_train_final_2, y_train_final_2)
    weighted_accuracy = weighted_voting.score(X_test_2, y_test_2)
    
    print(f"Weights {weights}: {weighted_accuracy * 100:.2f}%")
    
    if weighted_accuracy > best_accuracy:
        best_accuracy = weighted_accuracy
        best_weights = weights

print(f"\nBest weights: {best_weights} with {best_accuracy * 100:.2f}% accuracy")

# =============================================================================
# BIAS-VARIANCE TRADEOFF PLOT
# =============================================================================

print("\nPlotting Bias-Variance Tradeoff...")

def calculate_bias_variance(model, X_train, X_test, y_train, y_test, n_iterations=100):
    """
    Calculate bias and variance for a model
    """
    predictions = []
    
    for _ in range(n_iterations):
        # Create bootstrap sample
        indices = np.random.choice(len(X_train), len(X_train), replace=True)
        X_bootstrap = X_train.iloc[indices]
        y_bootstrap = y_train.iloc[indices]
        
        # Train model on bootstrap sample
        model_clone = clone(model)
        model_clone.fit(X_bootstrap, y_bootstrap)
        
        # Make predictions on test set
        pred = model_clone.predict(X_test)
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # Calculate bias and variance
    avg_predictions = np.mean(predictions, axis=0)
    bias = np.mean((avg_predictions - y_test) ** 2)
    variance = np.mean(np.var(predictions, axis=0))
    
    return bias, variance

from sklearn.base import clone

# Calculate bias and variance for individual models and voting classifier
models_bv = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=50, random_state=42),
    'Voting Classifier': VotingClassifier(estimators=models_task2, voting='soft')
}

bias_variance_results = {}

print("\nCalculating Bias and Variance for models...")
for name, model in models_bv.items():
    bias, variance = calculate_bias_variance(
        model, X_train_final_2, X_test_2, y_train_final_2, y_test_2, n_iterations=50
    )
    bias_variance_results[name] = {'bias': bias, 'variance': variance}
    print(f"{name}: Bias = {bias:.4f}, Variance = {variance:.4f}")

# Plot bias-variance tradeoff
plt.figure(figsize=(12, 8))
for name, results in bias_variance_results.items():
    plt.scatter(results['bias'], results['variance'], label=name, s=100)
    plt.annotate(name, (results['bias'], results['variance']), 
                xytext=(5, 5), textcoords='offset points')

plt.xlabel('Bias')
plt.ylabel('Variance')
plt.title('Bias-Variance Tradeoff for Different Models')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =============================================================================
# TASK 3: VOTING CLASSIFIER WITH DIFFERENT FEATURES
# =============================================================================

print("\n" + "=" * 70)
print("TASK 3: VOTING CLASSIFIER WITH restEcg AND Chol")
print("=" * 70)

# Extract different two features
X_features_3 = df_scaled[['restecg', 'chol']]
y = df_scaled['Label']

# Split the data
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(
    X_features_3, y, test_size=0.2, random_state=0, stratify=y
)

X_train_final_3, X_val_3, y_train_final_3, y_val_3 = train_test_split(
    X_train_3, y_train_3, test_size=0.3, random_state=0, stratify=y_train_3
)

print(f"Training set shape with features (restecg, chol): {X_train_final_3.shape}")

# Define models for this task
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
ada_model = AdaBoostClassifier(n_estimators=100, random_state=42)

# Train individual models
print("\nTraining individual models...")
rf_model.fit(X_train_final_3, y_train_final_3)
ada_model.fit(X_train_final_3, y_train_final_3)

# Individual model accuracies
rf_train_acc = rf_model.score(X_train_final_3, y_train_final_3)
rf_test_acc = rf_model.score(X_test_3, y_test_3)

ada_train_acc = ada_model.score(X_train_final_3, y_train_final_3)
ada_test_acc = ada_model.score(X_test_3, y_test_3)

print(f"Random Forest - Train: {rf_train_acc * 100:.2f}%, Test: {rf_test_acc * 100:.2f}%")
print(f"AdaBoost - Train: {ada_train_acc * 100:.2f}%, Test: {ada_test_acc * 100:.2f}%")

# Voting Classifier with RF and AdaBoost
print("\nVoting Classifier (Random Forest + AdaBoost):")
voting_clf_3 = VotingClassifier(
    estimators=[('rf', rf_model), ('ada', ada_model)],
    voting='soft'
)
voting_clf_3.fit(X_train_final_3, y_train_final_3)
voting_train_acc = voting_clf_3.score(X_train_final_3, y_train_final_3)
voting_test_acc = voting_clf_3.score(X_test_3, y_test_3)

print(f"Voting Classifier - Train: {voting_train_acc * 100:.2f}%, Test: {voting_test_acc * 100:.2f}%")

# Plot accuracy comparison
models_names = ['Random Forest', 'AdaBoost', 'Voting Ensemble']
train_accuracies = [rf_train_acc, ada_train_acc, voting_train_acc]
test_accuracies = [rf_test_acc, ada_test_acc, voting_test_acc]

x_pos = np.arange(len(models_names))

plt.figure(figsize=(10, 6))
bar_width = 0.35

plt.bar(x_pos - bar_width/2, train_accuracies, bar_width, 
        label='Training Accuracy', alpha=0.7, color='blue')
plt.bar(x_pos + bar_width/2, test_accuracies, bar_width, 
        label='Testing Accuracy', alpha=0.7, color='red')

plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Training vs Testing Accuracy: Individual Models vs Ensemble')
plt.xticks(x_pos, models_names)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# =============================================================================
# COMPREHENSIVE RESULTS SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("COMPREHENSIVE RESULTS SUMMARY")
print("=" * 70)

print("\nTASK 1 - Individual Algorithm Performance:")
print("-" * 50)
for model_name, result in results.items():
    print(f"{model_name}:")
    print(f"  Training Accuracy: {result['Training Accuracy'] * 100:.2f}%")
    print(f"  Testing Accuracy: {result['Testing Accuracy'] * 100:.2f}%")

print("\nTASK 2 - Voting Classifier Results:")
print("-" * 50)
print(f"Best Voting Type: {best_voting}")
print(f"Best Weights: {best_weights}")
print(f"Best Accuracy with 2 features: {best_accuracy * 100:.2f}%")

print("\nTASK 3 - Ensemble vs Individual Models:")
print("-" * 50)
print(f"Random Forest Test Accuracy: {rf_test_acc * 100:.2f}%")
print(f"AdaBoost Test Accuracy: {ada_test_acc * 100:.2f}%")
print(f"Voting Ensemble Test Accuracy: {voting_test_acc * 100:.2f}%")

print(f"\nEnsemble improvement over best individual model: "
      f"{(voting_test_acc - max(rf_test_acc, ada_test_acc)) * 100:.2f}%")

print("\n" + "=" * 70)
print("ENSEMBLE LEARNING LAB COMPLETED SUCCESSFULLY!")
print("=" * 70)
