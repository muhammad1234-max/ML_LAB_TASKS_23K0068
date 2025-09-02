import pandas as pd
import numpy as np


D1A = pd.read_csv("Lab2 D1A.csv")

#customizing the dataset by adding attributes
customizedData = pd.DataFrame({
    "fid": D1A["fid"],
    "Size": np.random.choice(["Small", "Medium", "High"], len(D1A)),
    "Direction": np.random.choice(["North", "South", "East", "West"], len(D1A)),
    "Timing": np.random.choice(["Full-time", "Part-time"], len(D1A)),
    "CategoryAttr": np.random.choice(["A", "B", "C"], len(D1A)),  # categorical with alphabets
    "ContinuousAttr": np.random.randn(len(D1A)) * 100             # continuous with random numbers
})

print("Customized dataset shape:", customizedData.shape)

print("D1A columns:", D1A.columns.tolist())
print("customizedData columns:", customizedData.columns.tolist())


#Merging with D1A
modifiedData = pd.merge(D1A, customizedData, on="fid", how="inner")
print("Modified dataset shape:", modifiedData.shape)
