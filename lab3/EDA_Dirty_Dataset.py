import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#a random dirty dataset i got from the internet
df = pd.read_csv("dirty_dataset.csv")

# Basic Info
print("Shape:", df.shape)
print("\nInfo:")
print(df.info())
print("\nPreview:")
print(df.head())

#Missing Values
print("\nMissing Values per Column:")
print(df.isnull().sum())

#Duplicates
print("\nDuplicate Rows:", df.duplicated().sum())

#Unique Values (to check inconsistencies)
print("\nUnique Genders:", df["Gender"].unique())

#Summary Statistics
print("\nSummary Statistics:")
print(df.describe(include="all"))

#Outlier Detection
sns.boxplot(x=df["Age"])
plt.title("Boxplot - Age (Outlier Detection)")
plt.show()

sns.boxplot(x=df["StudyHours"])
plt.title("Boxplot - Study Hours")
plt.show()

sns.boxplot(x=df["GPA"])
plt.title("Boxplot - GPA")
plt.show()

#Correlation Heatmap
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
