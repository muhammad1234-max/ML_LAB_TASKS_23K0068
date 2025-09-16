import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load datasets
print("Loading datasets...")
#the datasets i selected with more then 10 attributes
wine = load_wine()
cancer = load_breast_cancer()

wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
wine_df['target'] = wine.target
cancer_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
cancer_df['target'] = cancer.target

print(f"Wine dataset shape: {wine_df.shape}")
print(f"Cancer dataset shape: {cancer_df.shape}")
print(f"Wine features: {len(wine.feature_names)}")
print(f"Cancer features: {len(cancer.feature_names)}")

#Prepare data for both datasets
datasets = {
    'Dataset1_Wine': (wine.data, wine.target, wine.feature_names),
    'Dataset2_Cancer': (cancer.data, cancer.target, cancer.feature_names)
}

#Initialize results dictionary
results = {
    'Dataset1': {},
    'Dataset2': {}
}

#Define the methods to test
methods = [
    {'criterion': 'gini', 'pruning': False, 'ccp_alpha': 0.0},
    {'criterion': 'gini', 'pruning': True, 'ccp_alpha': 0.015},
    {'criterion': 'entropy', 'pruning': False, 'ccp_alpha': 0.0},
    {'criterion': 'entropy', 'pruning': True, 'ccp_alpha': 0.015}
]

#Train and evaluate models for each dataset
for dataset_name, (X, y, feature_names) in datasets.items():
    print(f"\n{'='*50}")
    print(f"Processing {dataset_name}")
    print(f"{'='*50}")
    
    #Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    dataset_key = 'Dataset1' if 'Wine' in dataset_name else 'Dataset2'
    results[dataset_key]['X_train'] = X_train
    results[dataset_key]['X_test'] = X_test
    results[dataset_key]['y_train'] = y_train
    results[dataset_key]['y_test'] = y_test
    
    for i, method in enumerate(methods):
        #Create decision tree classifier
        clf = DecisionTreeClassifier(
            criterion=method['criterion'],
            ccp_alpha=method['ccp_alpha'],
            random_state=42
        )
        
        #Train the model
        clf.fit(X_train, y_train)
        
        #Make predictions
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        
        #calculate accuracies
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        #Store results
        method_name = f"DT using {method['criterion']} {'with' if method['pruning'] else 'without'} pruning"
        results[dataset_key][method_name] = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'classifier': clf,
            'y_test_pred': y_test_pred
        }
        
        print(f"{method_name}:")
        print(f"  Training Accuracy: {train_accuracy:.4f}")
        print(f"  Testing Accuracy: {test_accuracy:.4f}")
        print(f"  Difference: {abs(train_accuracy - test_accuracy):.4f}")

#Create the results table
print("\n" + "="*80)
print("FINAL RESULTS TABLE")
print("="*80)

table_data = []
headers = ["Method", "Dataset1 Training", "Dataset1 Testing", "Dataset2 Training", "Dataset2 Testing"]

for method in methods:
    method_name = f"DT using {method['criterion']} {'with' if method['pruning'] else 'without'} pruning"
    
    dataset1_train = results['Dataset1'][method_name]['train_accuracy']
    dataset1_test = results['Dataset1'][method_name]['test_accuracy']
    dataset2_train = results['Dataset2'][method_name]['train_accuracy']
    dataset2_test = results['Dataset2'][method_name]['test_accuracy']
    
    table_data.append([
        method_name,
        f"{dataset1_train:.4f}",
        f"{dataset1_test:.4f}",
        f"{dataset2_train:.4f}",
        f"{dataset2_test:.4f}"
    ])

#Display the table
print(f"\n{headers[0]:<40} {headers[1]:<20} {headers[2]:<20} {headers[3]:<20} {headers[4]:<20}")
print("-" * 120)
for row in table_data:
    print(f"{row[0]:<40} {row[1]:<20} {row[2]:<20} {row[3]:<20} {row[4]:<20}")

#Calculate correct predictions for Dataset2 with entropy and pruning
entropy_pruning_method = "DT using entropy with pruning"
y_test_actual = results['Dataset2']['y_test']
y_test_pred = results['Dataset2'][entropy_pruning_method]['y_test_pred']

correct_predictions = sum(y_test_actual == y_test_pred)
total_predictions = len(y_test_actual)

print(f"\nCorrect predictions for {entropy_pruning_method} on Dataset2:")
print(f"Correctly Predict: {correct_predictions} Out Of {total_predictions}")

#Plotting the graphs
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Decision Tree Performance Comparison', fontsize=16, fontweight='bold')

#dataset 1 Training and Testing Accuracy
methods_names = [method['criterion'] + (' + pruning' if method['pruning'] else '') for method in methods]
dataset1_train_acc = [results['Dataset1'][f"DT using {method['criterion']} {'with' if method['pruning'] else 'without'} pruning"]['train_accuracy'] for method in methods]
dataset1_test_acc = [results['Dataset1'][f"DT using {method['criterion']} {'with' if method['pruning'] else 'without'} pruning"]['test_accuracy'] for method in methods]

x = np.arange(len(methods_names))
width = 0.35

axes[0, 0].bar(x - width/2, dataset1_train_acc, width, label='Training Accuracy', alpha=0.8)
axes[0, 0].bar(x + width/2, dataset1_test_acc, width, label='Testing Accuracy', alpha=0.8)
axes[0, 0].set_title('Dataset 1 (Wine) - Accuracy Comparison')
axes[0, 0].set_xlabel('Methods')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(methods_names, rotation=45, ha='right')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

#Dataset 1 Overfitting Analysis (Difference between train and test)
dataset1_diff = [abs(train - test) for train, test in zip(dataset1_train_acc, dataset1_test_acc)]
axes[0, 1].bar(methods_names, dataset1_diff, alpha=0.7, color='red')
axes[0, 1].set_title('Dataset 1 (Wine) - Overfitting Measure (|Train-Test|)')
axes[0, 1].set_xlabel('Methods')
axes[0, 1].set_ylabel('Accuracy Difference')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(True, alpha=0.3)

#Dataset 2 Training and Testing Accuracy
dataset2_train_acc = [results['Dataset2'][f"DT using {method['criterion']} {'with' if method['pruning'] else 'without'} pruning"]['train_accuracy'] for method in methods]
dataset2_test_acc = [results['Dataset2'][f"DT using {method['criterion']} {'with' if method['pruning'] else 'without'} pruning"]['test_accuracy'] for method in methods]

axes[1, 0].bar(x - width/2, dataset2_train_acc, width, label='Training Accuracy', alpha=0.8)
axes[1, 0].bar(x + width/2, dataset2_test_acc, width, label='Testing Accuracy', alpha=0.8)
axes[1, 0].set_title('Dataset 2 (Cancer) - Accuracy Comparison')
axes[1, 0].set_xlabel('Methods')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(methods_names, rotation=45, ha='right')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

#Dataset 2 Overfitting Analysis (Difference between train and test)
dataset2_diff = [abs(train - test) for train, test in zip(dataset2_train_acc, dataset2_test_acc)]
axes[1, 1].bar(methods_names, dataset2_diff, alpha=0.7, color='red')
axes[1, 1].set_title('Dataset 2 (Cancer) - Overfitting Measure (|Train-Test|)')
axes[1, 1].set_xlabel('Methods')
axes[1, 1].set_ylabel('Accuracy Difference')
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#Additional detailed analysis
print("\n" + "="*80)
print("MODEL FIT ANALYSIS")
print("="*80)

for dataset_key in ['Dataset1', 'Dataset2']:
    print(f"\n{dataset_key} Analysis:")
    for method_name in results[dataset_key]:
        if method_name not in ['X_train', 'X_test', 'y_train', 'y_test']:
            train_acc = results[dataset_key][method_name]['train_accuracy']
            test_acc = results[dataset_key][method_name]['test_accuracy']
            diff = abs(train_acc - test_acc)
            
            if diff < 0.05:
                fit_status = "Generalized (Good fit)"
            elif train_acc > test_acc + 0.1:
                fit_status = "Overfit"
            else:
                fit_status = "Underfit or High Variance"
            
            print(f"  {method_name}: {fit_status} (Diff: {diff:.4f})")
