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


