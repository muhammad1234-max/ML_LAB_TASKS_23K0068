# Basic Python Data Types and Collections

# Integer, Float, String
num = 10             # Integer
pi = 3.14            # Float
name = "Machine Learning"  # String

# List - ordered, mutable collection
fruits = ["apple", "banana", "cherry"]

# Tuple - ordered, immutable collection
colors = ("red", "green", "blue")

# Set - unordered, unique elements
unique_numbers = {1, 2, 3, 4}

# Dictionary - key-value pairs
student = {"name": "Ali", "age": 21, "grade": "A"}

print(fruits)
print(colors)
print(unique_numbers)
print(student)

# Importing Pandas Library
import pandas as pd

# Creating a Series (1D data)
data = [10, 20, 30, 40]
series = pd.Series(data, index=['a', 'b', 'c', 'd'])
print("Pandas Series:")
print(series)

# Creating a DataFrame (2D data)
data = {
    "Name": ["Ali", "Sara", "Ahmed", "Zara"],
    "Age": [20, 21, 19, 22],
    "Score": [85, 90, 78, 88]
}
df = pd.DataFrame(data)
print("\nDataFrame:")
print(df)


# Reading a CSV file
data = pd.read_csv("sample.csv")   # make sure sample.csv exists in your directory
print(data.head())

# Saving (storing) a dataset to CSV
df.to_csv("output.csv", index=False)
print("DataFrame saved successfully as 'output.csv'")



# Display top and bottom records
print(df.head(3))   # Top 3 records
print(df.tail(2))   # Bottom 2 records

# Checking column names and dataset info
print("Column Names:", df.columns)
print("Shape of dataset:", df.shape)
print("Data Info:")
print(df.info())

# Checking null values and data types
print(df.isnull().sum())
print(df.dtypes)

# Checking data statistics
print(df.describe())


# Adding a new column
df["City"] = ["Karachi", "Lahore", "Islamabad", "Quetta"]
print(df)

# Removing a column
df = df.drop("City", axis=1)
print(df)

# Removing a row
df = df.drop(1, axis=0)
print(df)

# Changing specific values
df.loc[2, "Score"] = 95
print(df)


# Resetting and Setting Index
df.reset_index(drop=True, inplace=True)
print(df)

df.set_index("Name", inplace=True)
print(df)

# Extracting specific columns
print(df["Age"])  # Single column
print(df[["Age", "Score"]])  # Multiple columns

# Extracting using loc (label-based)
print(df.loc["Ahmed", ["Age", "Score"]])

# Extracting using iloc (index-based)
print(df.iloc[0:2, 0:2])  # First two rows and two columns

import matplotlib.pyplot as plt
import seaborn as sns

# Sample dataset
data = pd.DataFrame({
    "Age": [20, 25, 22, 30, 27],
    "Score": [85, 78, 90, 88, 95],
    "City": ["Karachi", "Lahore", "Islamabad", "Peshawar", "Quetta"]
})

# Histogram
plt.hist(data["Age"], bins=5)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Scatter Plot
plt.scatter(data["Age"], data["Score"])
plt.title("Age vs Score")
plt.xlabel("Age")
plt.ylabel("Score")
plt.show()

# Boxplot
sns.boxplot(x=data["Score"])
plt.title("Score Distribution")
plt.show()

# Correlation Heatmap
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


import numpy as np

print("Mean of Age:", data["Age"].mean())
print("Median of Age:", data["Age"].median())
print("Mode of Age:", data["Age"].mode()[0])
print("Standard Deviation of Age:", data["Age"].std())
print("Min Age:", data["Age"].min())
print("Max Age:", data["Age"].max())


# Creating a dataset for exam failure causes
failure_data = {
    "Test_Score": [45, 32, 50, 29, 60, 40, 35, 55, 48, 30],
    "Writing_Skills": [60, 55, 70, 50, 80, 65, 58, 72, 68, 45],
    "Reading_Skills": [65, 50, 75, 40, 85, 60, 55, 70, 66, 48],
    "Attendance": [80, 70, 90, 60, 95, 85, 75, 88, 82, 65],
    "Motivation_Level": [7, 5, 8, 4, 9, 6, 5, 8, 7, 4]
}

fail_df = pd.DataFrame(failure_data)
print(fail_df)


# Distribution of Age and Frequent Cold (sample columns)
sns.distplot(data["Age"], kde=True)
plt.title("Distribution of Age")
plt.show()

# Correlation plot
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Plot")
plt.show()

# Detecting outliers using boxplot
sns.boxplot(data["Score"])
plt.title("Outliers in Score Feature")
plt.show()


# How to organize your code (Create a Text block for Importing Libraries in next cell import required library
# now Create a Heading for Data Preprocessing and the add a new cell for Data analysis. Create headings for each
# cell )
# A. Download the dataset from Here
# B. Import your dataset in Cola or Jupyter notebook
# C. Find out the unique category in target variable
# D. Find out the total numbers of columns in entire dataset, Check out the first 50 records of the dataset +
# bottom 30 records. Find its Information related to memory size , its dimension and Explore the size and
# dimension of your dataset
# E. Find the numbers of columns and total number of samples
# F. Find out the number of records for each unique category of target variable using numerical and graphical
# visualization.
# G. Extract sample from 550th sample to 900th sample and take these features Gender to Coughing of Blood
# using Loc Method and store in a new variable. Save this dataset in a new place and apply statistical
# analysis on these extracted records In statistical analysis you have to find out the mean , median , mode ,
# standard deviation , minimum an maximum
# H. Extract sample of last 20 records , take 5 features in consideration from Shortness of Breath to Frequent
# cold column using iLoc Method Save this dataset in a new place and apply statistical analysis on these
# extracted records
# I. Extract sample of these records 110,220,360,440,656,778,202,889. Take all features in consideration.
# Check out the average of alcohol use in these records.

#import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#load the dataset
df = pd.read_csv("cancer patient data sets.csv")

#display first 5 records to confirm loading
df.head()
print("Unique categories in target variable (Level):")
print(df['Level'].unique())
#total number of columns
print("Total Columns:", df.shape[1])

#first 50 records
print("\nFirst 50 records:")
df.head(50)

#bottom 30 records
print("\nBottom 30 records:")
df.tail(30)

#information
print("\nDataset Info:")
print(df.info())

#dimension (rows, columns)
print("\nDataset Dimension:", df.shape)

#memory usage
print("\nMemory Usage:")
print(df.memory_usage(deep=True))

print("Number of columns:", len(df.columns))
print("Total number of samples:", len(df))

#numerical count
print("\nCount of each category in target variable:")
print(df['Level'].value_counts())

#graphical visualization
plt.figure(figsize=(6,4))
sns.countplot(x='Level', data=df, palette="Set2")
plt.title("Distribution of Target Variable (Level)")
plt.show()

#extract using loc (inclusive indexing)
subset1 = df.loc[550:900, "Gender":"Coughing of Blood"]

#save to new CSV
subset1.to_csv("subset1.csv", index=False)

#statistical analysis
stats1 = subset1.describe().T
stats1["median"] = subset1.median()
stats1["mode"] = subset1.mode().iloc[0]
stats1["std"] = subset1.std()
stats1["min"] = subset1.min()
stats1["max"] = subset1.max()

print("\nStatistical Analysis (Records 550-900, Gender → Coughing of Blood):")
print(stats1)

subset2 = df.iloc[-20:, df.columns.get_loc("Shortness of Breath"):df.columns.get_loc("Frequent Cold")+1]

#save to new CSV
subset2.to_csv("subset2.csv", index=False)

#statistical analysis
stats2 = subset2.describe().T
stats2["median"] = subset2.median()
stats2["mode"] = subset2.mode().iloc[0]
stats2["std"] = subset2.std()
stats2["min"] = subset2.min()
stats2["max"] = subset2.max()

print("\nStatistical Analysis (Last 20 Records, Shortness of Breath → Frequent Cold):")
print(stats2)

subset3 = df.loc[[110,220,360,440,656,778,202,889], :]

#average of Alcohol use
avg_alcohol_use = subset3["Alcohol use"].mean()
print("Average Alcohol Use for selected records:", avg_alcohol_use)
