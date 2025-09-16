import pandas as pd
import numpy as np
import math

#Creating the dataset from the table given by miss in the document
data = {
    'ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    'AGE': ['Young', 'Young', 'Young', 'Young', 'Young', 'Middle', 'Middle', 'Middle', 'Middle', 'Middle', 'Old', 'Old', 'Old', 'Old', 'Old'],
    'JOB_STATUS': [False, False, True, True, False, False, False, True, False, False, False, False, True, True, False],
    'OWNS_HOUSE': [False, False, False, True, False, False, False, True, True, True, True, True, False, False, False],
    'CREDIT_RATING': ['Fair', 'Good', 'Good', 'Fair', 'Fair', 'Fair', 'Good', 'Good', 'Excellent', 'Excellent', 'Excellent', 'Good', 'Good', 'Excellent', 'Fair'],
    'CLASS': ['No', 'No', 'Yes', 'Yes', 'No', 'No', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)
print("Loan Approval Dataset:")
print(df.to_string(index=False))
print("\n")

#Calc entropy function
def calculate_entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = 0
    for count in counts:
        probability = count / len(target_col)
        entropy += -probability * math.log2(probability)
    return entropy

#Calc information gain function
def calculate_info_gain(data, feature, target_name="CLASS"):
    total_entropy = calculate_entropy(data[target_name])
    
    #get unique values of the feature
    values, counts = np.unique(data[feature], return_counts=True)
    
    weighted_entropy = 0
    for value, count in zip(values, counts):
        subset = data[data[feature] == value]
        subset_entropy = calculate_entropy(subset[target_name])
        weighted_entropy += (count / len(data)) * subset_entropy
    
    information_gain = total_entropy - weighted_entropy
    return information_gain

#calc entropy of the entire dataset
total_entropy = calculate_entropy(df['CLASS'])
print(f"Total Entropy of the dataset: {total_entropy:.4f}")

#calc information gain for each feature
features = ['AGE', 'JOB_STATUS', 'OWNS_HOUSE', 'CREDIT_RATING']
info_gains = {}

print("Information Gain Calculations:")
print("-" * 50)
for feature in features:
    ig = calculate_info_gain(df, feature)
    info_gains[feature] = ig
    print(f"{feature}: {ig:.4f}")

#find the feature with maximum information gain
root_node = max(info_gains, key=info_gains.get)
print(f"\nRoot Node should be: {root_node} (Highest Information Gain: {info_gains[root_node]:.4f})")

#let's also calculate gain ratio for completeness
def calculate_gain_ratio(data, feature, target_name="CLASS"):
    info_gain = calculate_info_gain(data, feature, target_name)
    
    values, counts = np.unique(data[feature], return_counts=True)
    split_info = 0
    for count in counts:
        probability = count / len(data)
        split_info += -probability * math.log2(probability)
    
    #Avoiding division by zero error
    if split_info == 0:
        return 0
    
    gain_ratio = info_gain / split_info
    return gain_ratio

print("\nGain Ratio Calculations:")
print("-" * 50)
for feature in features:
    gr = calculate_gain_ratio(df, feature)
    print(f"{feature}: {gr:.4f}")
