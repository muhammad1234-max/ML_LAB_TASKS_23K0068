#import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#load the dataset
df = pd.read_csv("cancer patient data sets.csv")

# Histogram for all numerical columns
df.hist(figsize=(15,12), bins=20, color='skyblue', edgecolor='black')
plt.suptitle("Histograms of Features", fontsize=16)
plt.show()

# Boxplots for numerical features
plt.figure(figsize=(15,8))
sns.boxplot(data=df)
plt.title("Boxplots of Dataset Features")
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Age Distribution
plt.figure(figsize=(6,4))
sns.histplot(df['Age'], bins=20, kde=True, color='blue')
plt.title("Distribution of Age")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Frequent Cold Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Frequent Cold', data=df, palette="viridis")
plt.title("Distribution of Frequent Cold")
plt.xlabel("Frequent Cold")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x=df['Obesity'], color='orange')
plt.title("Outlier Detection in Obesity Feature")
plt.show()
