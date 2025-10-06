# =============================================================================
# 1. MAX VOTING ENSEMBLE (Manual Implementation)
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy import stats

def max_voting_ensemble():
    """
    MAX VOTING ENSEMBLE TECHNIQUE
    - Multiple models make predictions for each data point
    - Each prediction is considered as a 'vote'
    - Final prediction is the majority vote (mode)
    - Used for classification problems
    """
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize multiple models
    model1 = DecisionTreeClassifier(random_state=42)
    model2 = KNeighborsClassifier(n_neighbors=5)
    model3 = LogisticRegression(random_state=42)
    
    # Train all models
    print("Training individual models for Max Voting...")
    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)
    model3.fit(X_train, y_train)
    
    # Get predictions from each model
    pred1 = model1.predict(X_test)
    pred2 = model2.predict(X_test)
    pred3 = model3.predict(X_test)
    
    # Apply Max Voting (Manual Implementation)
    final_predictions = []
    for i in range(len(X_test)):
        # Collect votes from all models
        votes = [pred1[i], pred2[i], pred3[i]]
        # Take the mode (most frequent prediction)
        final_vote = stats.mode(votes)[0][0]
        final_predictions.append(final_vote)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, final_predictions)
    print(f"Max Voting Ensemble Accuracy: {accuracy:.4f}")
    
    # Compare with individual models
    acc1 = accuracy_score(y_test, pred1)
    acc2 = accuracy_score(y_test, pred2)
    acc3 = accuracy_score(y_test, pred3)
    
    print(f"Individual Model Accuracies:")
    print(f"Decision Tree: {acc1:.4f}")
    print(f"K-Neighbors: {acc2:.4f}")
    print(f"Logistic Regression: {acc3:.4f}")
    
    return final_predictions, accuracy

# Run max voting ensemble
max_voting_predictions, max_voting_accuracy = max_voting_ensemble()


# =============================================================================
# 2. AVERAGING ENSEMBLE (Probability Based)
# =============================================================================

def averaging_ensemble():
    """
    AVERAGING ENSEMBLE TECHNIQUE
    - Multiple models make probability predictions
    - Average of all probability predictions is taken
    - Final prediction based on highest average probability
    - Can be used for both classification and regression
    """
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize multiple models
    model1 = DecisionTreeClassifier(random_state=42)
    model2 = KNeighborsClassifier(n_neighbors=5)
    model3 = LogisticRegression(random_state=42)
    
    # Train all models
    print("Training individual models for Averaging Ensemble...")
    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)
    model3.fit(X_train, y_train)
    
    # Get probability predictions from each model
    # predict_proba returns probabilities for each class
    prob1 = model1.predict_proba(X_test)
    prob2 = model2.predict_proba(X_test)
    prob3 = model3.predict_proba(X_test)
    
    # Average the probabilities across all models
    avg_probabilities = (prob1 + prob2 + prob3) / 3
    
    # Final prediction is the class with highest average probability
    final_predictions = np.argmax(avg_probabilities, axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, final_predictions)
    print(f"Averaging Ensemble Accuracy: {accuracy:.4f}")
    
    # Display sample probabilities
    print("\nSample Probability Predictions:")
    print("Model 1 probabilities:", prob1[0])
    print("Model 2 probabilities:", prob2[0])
    print("Model 3 probabilities:", prob3[0])
    print("Averaged probabilities:", avg_probabilities[0])
    print("Final prediction:", final_predictions[0])
    print("Actual class:", y_test[0])
    
    return final_predictions, accuracy, avg_probabilities

# Run averaging ensemble
avg_predictions, avg_accuracy, avg_probs = averaging_ensemble()



# =============================================================================
# 3. VOTING CLASSIFIER (Sklearn Implementation)
# =============================================================================

from sklearn.ensemble import VotingClassifier

def voting_classifier_sklearn():
    """
    VOTING CLASSIFIER USING SKLEARN
    - Built-in VotingClassifier for easy ensemble implementation
    - Supports both 'hard' and 'soft' voting
    - Hard: Majority vote of predictions
    - Soft: Average of probability predictions
    """
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Define individual models
    model1 = LogisticRegression(random_state=42)
    model2 = DecisionTreeClassifier(random_state=42)
    model3 = KNeighborsClassifier(n_neighbors=5)
    
    # Create estimator list with names
    estimators = [
        ('lr', model1),      # Logistic Regression
        ('dt', model2),      # Decision Tree
        ('knn', model3)      # K-Nearest Neighbors
    ]
    
    # HARD VOTING CLASSIFIER
    print("HARD VOTING CLASSIFIER")
    print("-" * 30)
    hard_voting = VotingClassifier(
        estimators=estimators,
        voting='hard'  # Use majority vote
    )
    
    hard_voting.fit(X_train, y_train)
    hard_accuracy = hard_voting.score(X_test, y_test)
    print(f"Hard Voting Accuracy: {hard_accuracy:.4f}")
    
    # SOFT VOTING CLASSIFIER
    print("\nSOFT VOTING CLASSIFIER")
    print("-" * 30)
    soft_voting = VotingClassifier(
        estimators=estimators,
        voting='soft'  # Use average probabilities
    )
    
    soft_voting.fit(X_train, y_train)
    soft_accuracy = soft_voting.score(X_test, y_test)
    print(f"Soft Voting Accuracy: {soft_accuracy:.4f}")
    
    # Compare individual model performances
    print("\nINDIVIDUAL MODEL PERFORMANCE")
    print("-" * 30)
    for name, model in estimators:
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        print(f"{name.upper()} Accuracy: {accuracy:.4f}")
    
    return hard_voting, soft_voting, hard_accuracy, soft_accuracy

# Run voting classifier
hard_voter, soft_voter, hard_acc, soft_acc = voting_classifier_sklearn()



# =============================================================================
# 4. BAGGING CLASSIFIER (Bootstrap Aggregating)
# =============================================================================

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

def bagging_ensemble():
    """
    BAGGING (Bootstrap Aggregating) ENSEMBLE
    - Creates multiple subsets of original data (with replacement)
    - Trains a base model on each subset
    - Combines predictions from all models
    - Reduces variance and prevents overfitting
    """
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create base estimator (weak learner)
    base_estimator = DecisionTreeClassifier(max_depth=5, random_state=42)
    
    # BAGGING CLASSIFIER
    print("BAGGING CLASSIFIER IMPLEMENTATION")
    print("-" * 35)
    
    bagging_clf = BaggingClassifier(
        estimator=base_estimator,    # Base model to use
        n_estimators=50,            # Number of base models
        max_samples=0.8,            # Sample size for each model (80% of data)
        max_features=0.8,           # Feature size for each model (80% of features)
        bootstrap=True,              # Sample with replacement
        bootstrap_features=False,    # Features without replacement
        random_state=42,
        n_jobs=-1                   # Use all available cores
    )
    
    # Train the bagging classifier
    print("Training Bagging Classifier...")
    bagging_clf.fit(X_train, y_train)
    
    # Make predictions
    bagging_accuracy = bagging_clf.score(X_test, y_test)
    print(f"Bagging Classifier Accuracy: {bagging_accuracy:.4f}")
    
    # Compare with single decision tree
    single_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
    single_tree.fit(X_train, y_train)
    single_tree_accuracy = single_tree.score(X_test, y_test)
    print(f"Single Decision Tree Accuracy: {single_tree_accuracy:.4f}")
    
    print(f"Improvement with Bagging: {bagging_accuracy - single_tree_accuracy:.4f}")
    
    # Display bagging parameters
    print("\nBagging Classifier Parameters:")
    print(f"Number of estimators: {bagging_clf.n_estimators}")
    print(f"Max samples per estimator: {bagging_clf.max_samples}")
    print(f"Max features per estimator: {bagging_clf.max_features}")
    
    return bagging_clf, bagging_accuracy

# Run bagging ensemble
bagging_model, bagging_acc = bagging_ensemble()



# =============================================================================
# 5. RANDOM FOREST (Special Bagging Implementation)
# =============================================================================

from sklearn.ensemble import RandomForestClassifier

def random_forest_ensemble():
    """
    RANDOM FOREST CLASSIFIER
    - Specialized implementation of bagging
    - Uses decision trees as base estimators
    - Random subset of features for each split
    - Built-in feature importance calculation
    - Highly effective for various problems
    """
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print("RANDOM FOREST CLASSIFIER")
    print("-" * 30)
    
    # Create Random Forest classifier
    rf_clf = RandomForestClassifier(
        n_estimators=100,        # Number of trees in the forest
        criterion='gini',        # Split quality measure
        max_depth=10,            # Maximum depth of trees
        min_samples_split=2,     # Minimum samples required to split
        min_samples_leaf=1,      # Minimum samples required at leaf node
        max_features='sqrt',     # Number of features for best split
        bootstrap=True,          # Use bootstrap samples
        random_state=42,
        n_jobs=-1,              # Use all available cores
        verbose=0
    )
    
    # Train the Random Forest
    print("Training Random Forest Classifier...")
    rf_clf.fit(X_train, y_train)
    
    # Make predictions
    rf_accuracy = rf_clf.score(X_test, y_test)
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    
    # Feature Importance
    feature_importance = rf_clf.feature_importances_
    print(f"\nTop 5 Most Important Features:")
    
    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'feature': [f'Feature_{i}' for i in range(X.shape[1])],
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print(importance_df.head())
    
    # Compare with single decision tree
    single_tree = DecisionTreeClassifier(max_depth=10, random_state=42)
    single_tree.fit(X_train, y_train)
    single_tree_accuracy = single_tree.score(X_test, y_test)
    
    print(f"\nSingle Decision Tree Accuracy: {single_tree_accuracy:.4f}")
    print(f"Random Forest Improvement: {rf_accuracy - single_tree_accuracy:.4f}")
    
    return rf_clf, rf_accuracy, feature_importance

# Run random forest
rf_model, rf_accuracy, feature_importance = random_forest_ensemble()



# =============================================================================
# 6. ADABOOST (Adaptive Boosting)
# =============================================================================

from sklearn.ensemble import AdaBoostClassifier

def adaboost_ensemble():
    """
    ADABOOST (Adaptive Boosting) CLASSIFIER
    - Sequential training of weak learners
    - Incorrectly predicted samples get higher weights
    - Each model tries to correct previous model's errors
    - Combines weak learners to form strong learner
    """
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print("ADABOOST CLASSIFIER")
    print("-" * 25)
    
    # Create base estimator (weak learner)
    base_estimator = DecisionTreeClassifier(
        max_depth=1,  # Very weak learner (decision stump)
        random_state=42
    )
    
    # Create AdaBoost classifier
    adaboost_clf = AdaBoostClassifier(
        estimator=base_estimator,    # Base weak learner
        n_estimators=50,            # Number of weak learners
        learning_rate=1.0,          # Contribution of each classifier
        algorithm='SAMME.R',        # Boosting algorithm
        random_state=42
    )
    
    # Train AdaBoost
    print("Training AdaBoost Classifier...")
    adaboost_clf.fit(X_train, y_train)
    
    # Make predictions
    adaboost_accuracy = adaboost_clf.score(X_test, y_test)
    print(f"AdaBoost Accuracy: {adaboost_accuracy:.4f}")
    
    # Compare with single weak learner
    weak_learner = DecisionTreeClassifier(max_depth=1, random_state=42)
    weak_learner.fit(X_train, y_train)
    weak_accuracy = weak_learner.score(X_test, y_test)
    
    print(f"Single Weak Learner Accuracy: {weak_accuracy:.4f}")
    print(f"AdaBoost Improvement: {adaboost_accuracy - weak_accuracy:.4f}")
    
    # Display estimator weights
    estimator_weights = adaboost_clf.estimator_weights_
    print(f"\nFirst 10 Estimator Weights: {estimator_weights[:10]}")
    
    # Display estimator errors
    estimator_errors = adaboost_clf.estimator_errors_
    print(f"First 10 Estimator Errors: {estimator_errors[:10]}")
    
    return adaboost_clf, adaboost_accuracy

# Run AdaBoost
adaboost_model, adaboost_acc = adaboost_ensemble()



# =============================================================================
# 7. XGBOOST (Extreme Gradient Boosting)
# =============================================================================

from xgboost import XGBClassifier

def xgboost_ensemble():
    """
    XGBOOST (Extreme Gradient Boosting)
    - Optimized gradient boosting implementation
    - Regularization to prevent overfitting
    - Handles missing values automatically
    - Parallel processing for faster training
    - Often wins machine learning competitions
    """
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print("XGBOOST CLASSIFIER")
    print("-" * 25)
    
    # Create XGBoost classifier
    xgb_clf = XGBClassifier(
        n_estimators=100,        # Number of boosting rounds
        max_depth=6,             # Maximum tree depth
        learning_rate=0.1,       # Step size shrinkage
        subsample=0.8,           # Subsample ratio of training instances
        colsample_bytree=0.8,    # Subsample ratio of features
        reg_alpha=0.1,           # L1 regularization term
        reg_lambda=1.0,          # L2 regularization term
        random_state=42,
        n_jobs=-1,              # Use all available cores
        eval_metric='logloss',   # Evaluation metric
        use_label_encoder=False
    )
    
    # Train XGBoost
    print("Training XGBoost Classifier...")
    xgb_clf.fit(X_train, y_train)
    
    # Make predictions
    xgb_accuracy = xgb_clf.score(X_test, y_test)
    print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")
    
    # Feature Importance
    feature_importance = xgb_clf.feature_importances_
    print(f"\nTop 5 Most Important Features:")
    
    importance_df = pd.DataFrame({
        'feature': [f'Feature_{i}' for i in range(X.shape[1])],
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print(importance_df.head())
    
    # Compare with other ensemble methods
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)
    rf_accuracy = rf_clf.score(X_test, y_test)
    
    print(f"\nRandom Forest Accuracy: {rf_accuracy:.4f}")
    print(f"XGBoost vs Random Forest Difference: {xgb_accuracy - rf_accuracy:.4f}")
    
    return xgb_clf, xgb_accuracy

# Run XGBoost
xgb_model, xgb_acc = xgboost_ensemble()



# =============================================================================
# 8. WEIGHTED AVERAGE VOTING
# =============================================================================

def weighted_average_voting():
    """
    WEIGHTED AVERAGE VOTING ENSEMBLE
    - Similar to averaging but with different weights for each model
    - Weights can be based on model performance or domain knowledge
    - More emphasis on better performing models
    """
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize multiple models
    model1 = DecisionTreeClassifier(random_state=42)
    model2 = KNeighborsClassifier(n_neighbors=5)
    model3 = LogisticRegression(random_state=42)
    
    # Train all models
    print("Training models for Weighted Average Voting...")
    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)
    model3.fit(X_train, y_train)
    
    # Get individual model accuracies (for weight assignment)
    acc1 = model1.score(X_test, y_test)
    acc2 = model2.score(X_test, y_test)
    acc3 = model3.score(X_test, y_test)
    
    print(f"Individual Model Accuracies:")
    print(f"Model 1 (Decision Tree): {acc1:.4f}")
    print(f"Model 2 (K-Neighbors): {acc2:.4f}")
    print(f"Model 3 (Logistic Regression): {acc3:.4f}")
    
    # Calculate weights based on performance
    total_acc = acc1 + acc2 + acc3
    weights = [acc1/total_acc, acc2/total_acc, acc3/total_acc]
    
    print(f"\nModel Weights: {[f'{w:.4f}' for w in weights]}")
    
    # Get probability predictions
    prob1 = model1.predict_proba(X_test)
    prob2 = model2.predict_proba(X_test)
    prob3 = model3.predict_proba(X_test)
    
    # Apply weighted average
    weighted_avg_prob = (weights[0] * prob1 + 
                        weights[1] * prob2 + 
                        weights[2] * prob3)
    
    # Final predictions
    weighted_predictions = np.argmax(weighted_avg_prob, axis=1)
    weighted_accuracy = accuracy_score(y_test, weighted_predictions)
    
    print(f"\nWeighted Average Voting Accuracy: {weighted_accuracy:.4f}")
    
    # Compare with simple averaging
    simple_avg_prob = (prob1 + prob2 + prob3) / 3
    simple_predictions = np.argmax(simple_avg_prob, axis=1)
    simple_accuracy = accuracy_score(y_test, simple_predictions)
    
    print(f"Simple Average Voting Accuracy: {simple_accuracy:.4f}")
    print(f"Weighted vs Simple Improvement: {weighted_accuracy - simple_accuracy:.4f}")
    
    return weighted_predictions, weighted_accuracy, weights

# Run weighted average voting
weighted_preds, weighted_acc, model_weights = weighted_average_voting()


# =============================================================================
# 9. COMPREHENSIVE ENSEMBLE COMPARISON
# =============================================================================

import matplotlib.pyplot as plt

def ensemble_comparison():
    """
    COMPREHENSIVE COMPARISON OF ALL ENSEMBLE METHODS
    - Compare all ensemble techniques on the same dataset
    - Visualize performance differences
    - Provide insights on when to use each method
    """
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print("COMPREHENSIVE ENSEMBLE METHOD COMPARISON")
    print("=" * 50)
    
    # Dictionary to store results
    results = {}
    
    # 1. Individual Models (Baseline)
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'K-Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Logistic Regression': LogisticRegression(random_state=42)
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        results[name] = accuracy
        print(f"{name}: {accuracy:.4f}")
    
    # 2. Ensemble Methods
    print("\nENSEMBLE METHODS:")
    print("-" * 20)
    
    # Max Voting
    estimators = [
        ('dt', DecisionTreeClassifier(random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('lr', LogisticRegression(random_state=42))
    ]
    
    # Hard Voting
    hard_voting = VotingClassifier(estimators=estimators, voting='hard')
    hard_voting.fit(X_train, y_train)
    results['Hard Voting'] = hard_voting.score(X_test, y_test)
    print(f"Hard Voting: {results['Hard Voting']:.4f}")
    
    # Soft Voting
    soft_voting = VotingClassifier(estimators=estimators, voting='soft')
    soft_voting.fit(X_train, y_train)
    results['Soft Voting'] = soft_voting.score(X_test, y_test)
    print(f"Soft Voting: {results['Soft Voting']:.4f}")
    
    # Bagging
    bagging = BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=50,
        random_state=42
    )
    bagging.fit(X_train, y_train)
    results['Bagging'] = bagging.score(X_test, y_test)
    print(f"Bagging: {results['Bagging']:.4f}")
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    results['Random Forest'] = rf.score(X_test, y_test)
    print(f"Random Forest: {results['Random Forest']:.4f}")
    
    # AdaBoost
    adaboost = AdaBoostClassifier(n_estimators=50, random_state=42)
    adaboost.fit(X_train, y_train)
    results['AdaBoost'] = adaboost.score(X_test, y_test)
    print(f"AdaBoost: {results['AdaBoost']:.4f}")
    
    # XGBoost
    xgb = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False)
    xgb.fit(X_train, y_train)
    results['XGBoost'] = xgb.score(X_test, y_test)
    print(f"XGBoost: {results['XGBoost']:.4f}")
    
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    methods = list(results.keys())
    accuracies = list(results.values())
    
    colors = ['lightblue' if 'Ensemble' not in method else 'lightcoral' 
             for method in methods]
    
    bars = plt.barh(methods, accuracies, color=colors, alpha=0.7)
    plt.xlabel('Accuracy')
    plt.title('Comparison of Ensemble Methods vs Individual Models')
    plt.xlim(0, 1)
    
    # Add value labels on bars
    for bar, accuracy in zip(bars, accuracies):
        plt.text(bar.get_width() - 0.05, bar.get_y() + bar.get_height()/2, 
                f'{accuracy:.4f}', ha='right', va='center', color='black', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    best_method = max(results, key=results.get)
    worst_method = min(results, key=results.get)
    
    print(f"\nSUMMARY:")
    print(f"Best Performing Method: {best_method} ({results[best_method]:.4f})")
    print(f"Worst Performing Method: {worst_method} ({results[worst_method]:.4f})")
    print(f"Range: {results[best_method] - results[worst_method]:.4f}")
    
    return results

# Run comprehensive comparison










#labtask2
# ● Perform EDA
# ● Check whether the dataset is balanced or not (using target variable “Label”)
# ● Check whether there is any empty records, categorical feature, or duplicate records, yes
# Then handle this and give a brief explanation why you have chosen this technique in a
# text cell or “jupyter/colab”
# ● Analyze your dataset and think if feature scaling is required or not. If yes then apply any
# scaling technique based on your distribution.
# ● Split your dataset in training, testing, and validation. The train split will be 80% and the
# test will be 20%. In the validation split your training samples will be 70% and the
# validation set will be 30%. Briefly describe why we use a validation set in a text cell.
# Declare Random_state=0
# ● Apply Random Forest, XGBoost, AdaBoost and check model training and testing
# accuracy.
# ● Compare the Training and Testing Results of all three algorithms


# EDA + Preprocessing + Train/Val/Test split + RandomForest, XGBoost, AdaBoost comparison
# Change FILEPATH to your actual CSV file name/path
FILEPATH = "dataset.csv"  # <-- replace with your CSV file (e.g., "heart.csv")

# -------------------------
# Imports
# -------------------------
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

# Try to import XGBoost; fallback to sklearn's GradientBoosting if not available
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    from sklearn.ensemble import GradientBoostingClassifier
    XGB_AVAILABLE = False

# -------------------------
# 1) Load dataset
# -------------------------
df = pd.read_csv(FILEPATH)   # ensure the CSV is in the working directory or use full path
print("Initial shape:", df.shape)
print(df.head())

# -------------------------
# 2) Quick EDA - structure & basic summaries
# -------------------------
print("\nColumns and dtypes:\n", df.dtypes)
print("\nBasic stats:\n", df.describe(include='all').T)

# Target column assumed to be named 'target'
if 'target' not in df.columns:
    raise ValueError("No 'target' column found. Rename your label column to 'target' or edit the script.")

# 2a) Check class balance
print("\nClass counts (target):")
print(df['target'].value_counts())
plt.figure(figsize=(5,3))
sns.countplot(x='target', data=df)
plt.title("Target class distribution")
plt.show()

# Explanation (include in your notebook / text cell):
# If classes are very imbalanced we would consider resampling methods (oversample/SMOTE/undersample).
# For moderately balanced datasets standard classifiers often still work but monitor metrics other than accuracy.

# -------------------------
# 3) Missing values, categorical features, duplicates
# -------------------------
print("\nMissing values per column:")
print(df.isnull().sum())

# Duplicate rows
dupes = df.duplicated().sum()
print("\nNumber of duplicate rows:", dupes)

# Categorical columns detection (based on domain knowledge)
# Based on typical heart dataset:
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']  # adjust if needed
# 'cs' might be 'ca' or 'c s' (if numeric), keep it numeric unless it is categorical
print("\nAssumed categorical columns:", categorical_cols)

# If any of these are not in df, remove them from the list
categorical_cols = [c for c in categorical_cols if c in df.columns]

# -------------------------
# 4) Handle duplicates and missing values
# -------------------------
# 4a: Handle duplicates - drop exact duplicates
if dupes > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print("Dropped duplicates. New shape:", df.shape)

# 4b: Missing values handling
# Strategy explanation (put in notebook text):
# - For numeric columns: use median imputation (robust to outliers)
# - For categorical columns: use mode (most frequent)
# These choices are simple, unbiased, reproducible and often effective for baseline models.

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# exclude target from numeric imputation
num_cols = [c for c in num_cols if c != 'target']

cat_cols = categorical_cols

# Numeric imputation with median
num_imputer = SimpleImputer(strategy='median')
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# Categorical imputation with mode (most frequent)
cat_imputer = SimpleImputer(strategy='most_frequent')
if len(cat_cols) > 0:
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

print("\nMissing values after imputation:\n", df.isnull().sum())

# -------------------------
# 5) Encoding categorical variables
# -------------------------
# We will one-hot encode nominal categories (safe approach).
# For binary categories like 'sex' and 'fbs' you could keep as 0/1; get_dummies will handle that as well.

df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=True)  # drop_first avoids multicollinearity
print("\nShape after encoding:", df_enc.shape)

# -------------------------
# 6) Feature scaling decision & application
# -------------------------
# Check skewness of numeric features to decide scaler
continuous_cols = [c for c in df_enc.columns if c not in ['target'] and (df_enc[c].dtype.kind in 'biufc')]
# note: after get_dummies some continuous_cols include dummy columns; we will scale only original numeric features
# Let's identify the original numeric columns we want to scale (common continuous ones)
original_numeric = ['age','trestbps','chol','thalac','oldpeak','cs']  # 'cs' maybe numeric; include if exists
original_numeric = [c for c in original_numeric if c in df_enc.columns]

print("\nNumeric columns considered for scaling:", original_numeric)

# Compute skewness
skewness = df_enc[original_numeric].skew().round(3)
print("\nSkewness of numeric columns:\n", skewness)

# Strategy:
# - If |skew| > 1 => use RobustScaler (robust to outliers) or log-transform when positive.
# - else use StandardScaler (mean=0, std=1) as baseline for KNN & tree-based models (tree-based don't need scaling, but others might)
skew_threshold = 1.0
cols_high_skew = skewness[abs(skewness) > skew_threshold].index.tolist()
print("\nColumns with high skewness (>|1|):", cols_high_skew)

# Apply scaling
# We'll apply RobustScaler to highly skewed cols, StandardScaler to the rest of original_numeric
from sklearn.preprocessing import RobustScaler
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

# Note explanation (to include in your notebook):
# - Trees (RandomForest, AdaBoost) don't require scaling; scaling benefits distance-based models (KNN) and gradient-based optimization.
# - We kept sensible scalers: Robust for heavy skew / outliers, Standard otherwise.

# -------------------------
# 7) Train / Test / Validation split
# -------------------------
# Instructions:
# - Overall split: Train = 80%, Test = 20% (random_state=0)
# - Then split Train into Train-subset (70%) and Validation (30%) of the TRAIN set.
# Implementation:
X = sc_df.drop('target', axis=1)
y = sc_df['target']

# 80/20 split
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

# Now split the 80% (X_train_full) into Train (70%) and Val (30%) relative to that 80%:
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.3, random_state=0, stratify=y_train_full
)

print("\nFinal sample counts:")
print("Train:", X_train.shape[0])
print("Validation:", X_val.shape[0])
print("Test:", X_test.shape[0])

# Explanation (to include in a text cell):
# - Validation set is used to tune hyperparameters (choose model, K, depth, etc.) without touching the final test set.
# - Test set is only used once to estimate final generalization performance.

# -------------------------
# 8) Train models: RandomForest, XGBoost (or GradientBoosting fallback), AdaBoost
# -------------------------
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

# -------------------------
# 9) Comparison of results (table)
# -------------------------
results_df = pd.DataFrame(results).sort_values(by='Test_Acc', ascending=False).reset_index(drop=True)
print("\nModel comparison (sorted by Test Acc):")
print(results_df)

# Short analysis guidance (write in your notebook):
# - Compare Train vs Test accuracy to check overfitting:
#    * If Train >> Test -> overfitting.
#    * If both low -> underfitting.
# - Use Validation accuracy to decide hyperparameters (e.g., n_estimators, max_depth).
# - Prefer models with higher Test accuracy and stable Val/Test performance.


#task2
# Use the Same dataset as in Task 1
# ● Extract Only two Attributes with independent variable to analyze your results (restEcg
# and Oldpeak)
# ● Now train a Voting Classifier using (Decision Tree, KNN, Random Forest and XGboost)
# ● Check which Voting Parameter will give you the best Accuracy either soft or hard
# ● Check the best weights for these models.
# ● Plot the Bias and Variance Tradeoff Graph after Voting Classifier
# ensemble_results = ensemble_comparison()




