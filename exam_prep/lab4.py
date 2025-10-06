# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import graphviz
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# TASK 2: ID3 ALGORITHM IMPLEMENTATION FROM SCRATCH
# =============================================================================

def entropy(target_col):
    """
    Calculate entropy of a target column
    Formula: Entropy = -Σ p(x) * log2(p(x))
    """
    elements, counts = np.unique(target_col, return_counts=True)
    entropy_val = 0
    for count in counts:
        probability = count / len(target_col)
        entropy_val += -probability * np.log2(probability)
    return entropy_val

def information_gain(data, split_attribute, target_name="CLASS"):
    """
    Calculate information gain for a given attribute
    Formula: IG = Entropy(parent) - Σ [Weight * Entropy(child)]
    """
    # Calculate total entropy
    total_entropy = entropy(data[target_name])
    
    # Calculate weighted entropy of children
    values, counts = np.unique(data[split_attribute], return_counts=True)
    weighted_entropy = 0
    
    for i, value in enumerate(values):
        subset = data[data[split_attribute] == value]
        weight = counts[i] / len(data)
        weighted_entropy += weight * entropy(subset[target_name])
    
    # Information gain
    information_gain_val = total_entropy - weighted_entropy
    return information_gain_val

def find_root_node_id3(data, features, target_name="CLASS"):
    """
    Find the root node using ID3 algorithm (maximum information gain)
    """
    print("Calculating Information Gain for all attributes:")
    print("=" * 50)
    
    ig_values = {}
    for feature in features:
        ig = information_gain(data, feature, target_name)
        ig_values[feature] = ig
        print(f"Information Gain for {feature}: {ig:.4f}")
    
    # Find attribute with maximum information gain
    root_node = max(ig_values, key=ig_values.get)
    print(f"\nRoot Node: {root_node} (Highest IG: {ig_values[root_node]:.4f})")
    
    return root_node, ig_values

# Create the loan approval dataset for Task 2
loan_data = pd.DataFrame({
    'ID': range(1, 16),
    'AGE': ['Young', 'Young', 'Young', 'Young', 'Young', 
            'Middle', 'Middle', 'Middle', 'Middle', 'Middle',
            'Old', 'Old', 'Old', 'Old', 'Old'],
    'JOB_STATUS': [False, False, True, True, False,
                  False, False, True, False, False,
                  False, False, True, True, False],
    'OWNS_HOUSE': [False, False, False, True, False,
                   False, False, True, True, True,
                   True, True, False, False, False],
    'CREDIT_RATING': ['Fair', 'Good', 'Good', 'Fair', 'Fair',
                     'Fair', 'Good', 'Good', 'Excellent', 'Excellent',
                     'Excellent', 'Good', 'Good', 'Excellent', 'Fair'],
    'CLASS': ['No', 'No', 'Yes', 'Yes', 'No',
             'No', 'No', 'Yes', 'Yes', 'Yes',
             'Yes', 'Yes', 'Yes', 'Yes', 'No']
})

print("Loan Approval Dataset:")
print(loan_data)
print("\n" + "="*50 + "\n")

# Find root node using ID3
features = ['AGE', 'JOB_STATUS', 'OWNS_HOUSE', 'CREDIT_RATING']
root_node_id3, ig_values = find_root_node_id3(loan_data, features)

# =============================================================================
# BASIC DECISION TREE IMPLEMENTATION USING SCIKIT-LEARN
# =============================================================================

def prepare_data_for_sklearn(data, categorical_columns, target_column):
    """
    Prepare data for sklearn by encoding categorical variables
    """
    data_encoded = data.copy()
    le_dict = {}
    
    for col in categorical_columns:
        le = LabelEncoder()
        data_encoded[col] = le.fit_transform(data_encoded[col])
        le_dict[col] = le
    
    X = data_encoded.drop(columns=[target_column, 'ID'])
    y = data_encoded[target_column]
    
    return X, y, le_dict

# Prepare loan data for sklearn
categorical_cols = ['AGE', 'CREDIT_RATING', 'CLASS']
X_loan, y_loan, label_encoders = prepare_data_for_sklearn(loan_data, categorical_cols, 'CLASS')

# Split data
X_train_loan, X_test_loan, y_train_loan, y_test_loan = train_test_split(
    X_loan, y_loan, test_size=0.2, random_state=0
)

print("\n" + "="*50)
print("DECISION TREE IMPLEMENTATIONS")
print("="*50)

# =============================================================================
# DECISION TREE WITH DIFFERENT PARAMETERS
# =============================================================================

# 1. Decision Tree using Entropy without Pruning
print("\n1. Decision Tree using Entropy without Pruning:")
dt_entropy_no_prune = DecisionTreeClassifier(criterion='entropy', random_state=0)
dt_entropy_no_prune.fit(X_train_loan, y_train_loan)

train_acc_entropy = dt_entropy_no_prune.score(X_train_loan, y_train_loan)
predictions_entropy = dt_entropy_no_prune.predict(X_test_loan)
test_acc_entropy = accuracy_score(y_test_loan, predictions_entropy)

print(f"Training Accuracy: {train_acc_entropy * 100:.2f}%")
print(f"Testing Accuracy: {test_acc_entropy * 100:.2f}%")

# 2. Decision Tree using Entropy with Pruning
print("\n2. Decision Tree using Entropy with Pruning:")
dt_entropy_prune = DecisionTreeClassifier(criterion='entropy', ccp_alpha=0.015, random_state=0)
dt_entropy_prune.fit(X_train_loan, y_train_loan)

train_acc_entropy_prune = dt_entropy_prune.score(X_train_loan, y_train_loan)
predictions_entropy_prune = dt_entropy_prune.predict(X_test_loan)
test_acc_entropy_prune = accuracy_score(y_test_loan, predictions_entropy_prune)

print(f"Training Accuracy: {train_acc_entropy_prune * 100:.2f}%")
print(f"Testing Accuracy: {test_acc_entropy_prune * 100:.2f}%")

# 3. Decision Tree using Gini without Pruning
print("\n3. Decision Tree using Gini without Pruning:")
dt_gini_no_prune = DecisionTreeClassifier(criterion='gini', random_state=0)
dt_gini_no_prune.fit(X_train_loan, y_train_loan)

train_acc_gini = dt_gini_no_prune.score(X_train_loan, y_train_loan)
predictions_gini = dt_gini_no_prune.predict(X_test_loan)
test_acc_gini = accuracy_score(y_test_loan, predictions_gini)

print(f"Training Accuracy: {train_acc_gini * 100:.2f}%")
print(f"Testing Accuracy: {test_acc_gini * 100:.2f}%")

# 4. Decision Tree using Gini with Pruning
print("\n4. Decision Tree using Gini with Pruning:")
dt_gini_prune = DecisionTreeClassifier(criterion='gini', ccp_alpha=0.015, random_state=0)
dt_gini_prune.fit(X_train_loan, y_train_loan)

train_acc_gini_prune = dt_gini_prune.score(X_train_loan, y_train_loan)
predictions_gini_prune = dt_gini_prune.predict(X_test_loan)
test_acc_gini_prune = accuracy_score(y_test_loan, predictions_gini_prune)

print(f"Training Accuracy: {train_acc_gini_prune * 100:.2f}%")
print(f"Testing Accuracy: {test_acc_gini_prune * 100:.2f}%")

# =============================================================================
# VISUALIZE DECISION TREE
# =============================================================================

def visualize_decision_tree(model, feature_names, class_names, title):
    """
    Visualize decision tree using graphviz
    """
    dot_data = export_graphviz(
        model,
        out_file=None,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True
    )
    graph = graphviz.Source(dot_data)
    graph.render(title, format='png', cleanup=True)
    print(f"\nDecision tree visualization saved as {title}.png")
    return graph

# Visualize one of the trees
feature_names = ['JOB_STATUS', 'OWNS_HOUSE', 'AGE', 'CREDIT_RATING']
class_names = ['No', 'Yes']
tree_graph = visualize_decision_tree(dt_entropy_no_prune, feature_names, class_names, "loan_decision_tree")

# =============================================================================
# TASK 3: COMPREHENSIVE DATASET ANALYSIS
# =============================================================================

print("\n" + "="*50)
print("TASK 3: COMPREHENSIVE DATASET ANALYSIS")
print("="*50)

# Create a sample dataset for demonstration
def create_sample_dataset():
    """
    Create a sample dataset for EDA and modeling
    """
    np.random.seed(0)
    n_samples = 1000
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=7,
        n_redundant=3,
        n_clusters_per_class=1,
        random_state=0
    )
    
    # Create DataFrame
    feature_names = [f'Feature_{i+1}' for i in range(10)]
    df = pd.DataFrame(X, columns=feature_names)
    df['Label'] = y
    
    # Add some missing values, duplicates, and categorical data for demonstration
    df.iloc[10:15, 0] = np.nan  # Add missing values
    df.iloc[20:25, 1] = np.nan
    
    # Add duplicate rows
    duplicates = df.iloc[50:55].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # Add categorical feature
    df['Category'] = np.random.choice(['A', 'B', 'C'], size=len(df))
    
    return df

# Create dataset
df = create_sample_dataset()
print("Dataset shape:", df.shape)
print("\nFirst 5 rows of the dataset:")
print(df.head())

# 1. Perform EDA
print("\n1. EXPLORATORY DATA ANALYSIS (EDA)")
print("-" * 40)

# Basic information
print("Dataset Info:")
print(df.info())

print("\nDescriptive Statistics:")
print(df.describe())

# 2. Check if dataset is balanced
print("\n2. DATASET BALANCE CHECK")
print("-" * 40)
label_counts = df['Label'].value_counts()
print("Label distribution:")
print(label_counts)
print(f"Dataset is {'balanced' if label_counts[0] == label_counts[1] else 'imbalanced'}")

# 3. Check for empty records, categorical features, duplicates
print("\n3. DATA QUALITY CHECKS")
print("-" * 40)

# Missing values
print("Missing values per column:")
print(df.isnull().sum())

# Handle missing values - using mean for numerical, mode for categorical
print("\nHandling missing values...")
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in ['float64', 'int64']:
            # Numerical columns: fill with mean
            df[col].fillna(df[col].mean(), inplace=True)
            print(f"Filled missing values in {col} with mean: {df[col].mean():.2f}")
        else:
            # Categorical columns: fill with mode
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"Filled missing values in {col} with mode: {mode_val}")

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate records: {duplicates}")

# Remove duplicates
if duplicates > 0:
    df = df.drop_duplicates()
    print(f"Removed {duplicates} duplicate records")

# Handle categorical variables using one-hot encoding
print("\nHandling categorical variables...")
categorical_columns = df.select_dtypes(include=['object']).columns
print(f"Categorical columns: {list(categorical_columns)}")

if len(categorical_columns) > 0:
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    print("Applied one-hot encoding to categorical variables")

print(f"Dataset shape after preprocessing: {df.shape}")

# 4. Check correlation and perform feature selection
print("\n4. CORRELATION ANALYSIS AND FEATURE SELECTION")
print("-" * 40)

# Calculate correlation matrix
correlation_matrix = df.corr()

# Plot correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# Feature selection using Pearson Correlation
target_correlations = correlation_matrix['Label'].abs().sort_values(ascending=False)
print("Correlation with target variable (Label):")
print(target_correlations)

# Select features with correlation above threshold
correlation_threshold = 0.1
selected_features = target_correlations[target_correlations > correlation_threshold].index.tolist()
selected_features.remove('Label')  # Remove target variable
print(f"\nSelected features (correlation > {correlation_threshold}): {selected_features}")

# 5. Check if feature scaling is required
print("\n5. FEATURE SCALING ANALYSIS")
print("-" * 40)

# Check feature distributions
numerical_features = df[selected_features].select_dtypes(include=['float64', 'int64']).columns
print("Feature ranges before scaling:")
print(df[numerical_features].describe().loc[['min', 'max', 'std']])

# Apply StandardScaler (z-score normalization)
print("\nApplying StandardScaler...")
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[numerical_features] = scaler.fit_transform(df[numerical_features])

print("Feature ranges after scaling:")
print(df_scaled[numerical_features].describe().loc[['min', 'max', 'std']])

# 6. Split dataset into train, test, and validation sets
print("\n6. DATA SPLITTING")
print("-" * 40)

# Separate features and target
X = df_scaled[selected_features]
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
print("A validation set is used to tune hyperparameters and prevent overfitting.")
print("It provides an unbiased evaluation of model fit during training.")
print("Without a validation set, we might overfit to the test set by repeatedly tuning parameters.")

# 7. Apply Decision Tree and check accuracy
print("\n7. DECISION TREE MODEL TRAINING AND EVALUATION")
print("-" * 40)

# Train Decision Tree on final training set
dt_final = DecisionTreeClassifier(random_state=0)
dt_final.fit(X_train_final, y_train_final)

# Evaluate on all sets
train_accuracy = dt_final.score(X_train_final, y_train_final)
val_accuracy = dt_final.score(X_val, y_val)
test_accuracy = dt_final.score(X_test, y_test)

print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# =============================================================================
# TASK 4: CART ALGORITHM IMPLEMENTATION FROM SCRATCH
# =============================================================================

def gini_index(target_col):
    """
    Calculate Gini index of a target column
    Formula: Gini = 1 - Σ p(x)²
    """
    elements, counts = np.unique(target_col, return_counts=True)
    gini = 1
    for count in counts:
        probability = count / len(target_col)
        gini -= probability ** 2
    return gini

def gini_gain(data, split_attribute, target_name="Liked"):
    """
    Calculate Gini gain for a given attribute
    Formula: Gini Gain = Gini(parent) - Σ [Weight * Gini(child)]
    """
    # Calculate parent Gini index
    parent_gini = gini_index(data[target_name])
    
    # Calculate weighted Gini index of children
    values, counts = np.unique(data[split_attribute], return_counts=True)
    weighted_gini = 0
    
    for i, value in enumerate(values):
        subset = data[data[split_attribute] == value]
        weight = counts[i] / len(data)
        weighted_gini += weight * gini_index(subset[target_name])
    
    # Gini gain
    gini_gain_val = parent_gini - weighted_gini
    return gini_gain_val

def find_root_node_cart(data, features, target_name="Liked"):
    """
    Find the root node using CART algorithm (maximum Gini gain)
    """
    print("\n" + "="*50)
    print("CART ALGORITHM - FINDING ROOT NODE")
    print("="*50)
    
    gini_gains = {}
    for feature in features:
        gg = gini_gain(data, feature, target_name)
        gini_gains[feature] = gg
        print(f"Gini Gain for {feature}: {gg:.4f}")
    
    # Find attribute with maximum Gini gain
    root_node = max(gini_gains, key=gini_gains.get)
    print(f"\nRoot Node: {root_node} (Highest Gini Gain: {gini_gains[root_node]:.4f})")
    
    return root_node, gini_gains

# Create the course liking dataset for Task 4
course_data = pd.DataFrame({
    'Student': range(1, 11),
    'Prior_Experience': ['Yes', 'No', 'Yes', 'No', 'Yes', 
                        'No', 'Yes', 'Yes', 'Yes', 'Yes'],
    'Course': ['Programming', 'Programming', 'History', 'Programming', 'English',
              'Programming', 'Programming', 'Mathematics', 'Programming', 'Programming'],
    'Time': ['Day', 'Day', 'Night', 'Night', 'Day',
            'Day', 'Day', 'Night', 'Night', 'Night'],
    'Liked': ['Yes', 'No', 'No', 'Yes', 'Yes',
             'No', 'No', 'Yes', 'Yes', 'No']
})

print("\nCourse Liking Dataset:")
print(course_data)

# Find root node using CART
features_cart = ['Prior_Experience', 'Course', 'Time']
root_node_cart, gini_gains = find_root_node_cart(course_data, features_cart)

# =============================================================================
# SUMMARY OF RESULTS
# =============================================================================

print("\n" + "="*50)
print("SUMMARY OF RESULTS")
print("="*50)

print(f"\nTASK 2 - ID3 Algorithm:")
print(f"Root Node: {root_node_id3}")
print("Information Gains:", {k: f"{v:.4f}" for k, v in ig_values.items()})

print(f"\nTASK 4 - CART Algorithm:")
print(f"Root Node: {root_node_cart}")
print("Gini Gains:", {k: f"{v:.4f}" for k, v in gini_gains.items()})

print("\nTASK 3 - Model Performance:")
print(f"Final Decision Tree Accuracy: {test_accuracy * 100:.2f}%")

print("\nAll tasks completed successfully!")
