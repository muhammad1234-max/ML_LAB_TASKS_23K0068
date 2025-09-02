import pandas as pd

D1A = pd.read_csv("Lab2 D1A.csv")
D1B = pd.read_csv("Lab2 D1B.csv")
D1C = pd.read_csv("Lab2 D1C.csv")

#printing info abt the datasets
print("Shape of D1A:", D1A.shape)
print("Shape of D1B:", D1B.shape)
print("Shape of D1C:", D1C.shape)

#Combining D1A and D1B without duplicate columns
combined_AB = pd.concat([D1A, D1B.loc[:, ~D1B.columns.isin(D1A.columns)]], axis=1)
print("Combined AB shape:", combined_AB.shape)

#Merging D1A with D1C for similar records (inner join)
comboAC = pd.merge(D1A, D1C, on="county", how="inner")  
print("ComboAC shape:", comboAC.shape)
