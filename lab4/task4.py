import numpy as np
import pandas as pd

#making the dataset manually from the table miss gave us
student_data = {
    'Student': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Prior_Experience': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes'],
    'Course': ['Programming', 'Programming', 'History', 'Programming', 'English', 'Programming', 'Programming', 'Mathematics', 'Programming', 'Programming'],
    'Time': ['Day', 'Day', 'Night', 'Night', 'Day', 'Day', 'Day', 'Night', 'Night', 'Night'],
    'Liked': ['Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No']
}

student_df = pd.DataFrame(student_data)
print("\nStudent Course Preference Dataset:")
print(student_df.to_string(index=False))
print("\n")

#calculate Gini impurity function
def calculate_gini(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    gini = 1
    for count in counts:
        probability = count / len(target_col)
        gini -= probability ** 2
    return gini

#calculate Gini gain function for CART
def calculate_gini_gain(data, feature, target_name="Liked"):
    total_gini = calculate_gini(data[target_name])
    
    #get unique values of the feature
    values, counts = np.unique(data[feature], return_counts=True)
    
    weighted_gini = 0
    for value, count in zip(values, counts):
        subset = data[data[feature] == value]
        subset_gini = calculate_gini(subset[target_name])
        weighted_gini += (count / len(data)) * subset_gini
    
    gini_gain = total_gini - weighted_gini
    return gini_gain

#calculate Gini impurity of the entire dataset
total_gini = calculate_gini(student_df['Liked'])
print(f"Total Gini Impurity of the dataset: {total_gini:.4f}")

#calculate Gini gain for each feature
student_features = ['Prior_Experience', 'Course', 'Time']
gini_gains = {}

print("Gini Gain Calculations (CART Algorithm):")
print("-" * 60)
for feature in student_features:
    gg = calculate_gini_gain(student_df, feature)
    gini_gains[feature] = gg
    print(f"{feature}: {gg:.4f}")

#find the feature with maximum Gini gain
root_node_cart = max(gini_gains, key=gini_gains.get)
print(f"\nRoot Node (CART) should be: {root_node_cart} (Highest Gini Gain: {gini_gains[root_node_cart]:.4f})")

#let's also analyze each feature in detail
print("\nDetailed Analysis of Each Feature:")
print("=" * 60)

for feature in student_features:
    print(f"\nFeature: {feature}")
    print("-" * 30)
    
    values = np.unique(student_df[feature])
    for value in values:
        subset = student_df[student_df[feature] == value]
        liked_counts = subset['Liked'].value_counts()
        total = len(subset)
        
        print(f"  {value}: Total={total}, Liked={liked_counts.get('Yes', 0)}, Not Liked={liked_counts.get('No', 0)}")
        print(f"    Gini Impurity: {calculate_gini(subset['Liked']):.4f}")

#let's see what happens if we split on the root node
print(f"\nIf we split on '{root_node_cart}':")
root_values = np.unique(student_df[root_node_cart])
for value in root_values:
    subset = student_df[student_df[root_node_cart] == value]
    print(f"  {value}: {len(subset)} samples, Gini: {calculate_gini(subset['Liked']):.4f}")
