# Importing essential libraries for Data Preprocessing & EDA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


#combining datasets

# Sample DataFrames
df1 = pd.DataFrame({"ID": [1, 2, 3], "Name": ["Ali", "Sara", "Ahmed"]})
df2 = pd.DataFrame({"ID": [4, 5, 6], "Name": ["Zara", "Bilal", "Fatima"]})
df3 = pd.DataFrame({"ID": [1, 2, 3], "City": ["Karachi", "Lahore", "Islamabad"]})

# 1️⃣ Append - adds rows to another DataFrame
df_append = df1.append(df2, ignore_index=True)
print("Append Example:\n", df_append)

# 2️⃣ Concat - joins vertically (rows) or horizontally (columns)
df_concat = pd.concat([df1, df2], axis=0)  # vertical
print("\nConcat Vertically:\n", df_concat)

df_concat_h = pd.concat([df1, df3], axis=1)  # horizontal
print("\nConcat Horizontally:\n", df_concat_h)

# 3️⃣ Merge - joins based on a common key
df_merge = pd.merge(df1, df3, on="ID", how="inner")
print("\nMerge Example:\n", df_merge)

# 4️⃣ Join - joins by index
df1.set_index("ID", inplace=True)
df3.set_index("ID", inplace=True)
df_join = df1.join(df3)
print("\nJoin Example:\n", df_join)



#handling missing values

data = pd.DataFrame({
    "Age": [22, np.nan, 25, np.nan, 30],
    "City": ["Karachi", "Lahore", np.nan, "Islamabad", np.nan]
})

# 1️⃣ Drop missing values
drop_rows = data.dropna(axis=0)   # drop rows
drop_cols = data.dropna(axis=1)   # drop columns

# 2️⃣ Fill missing values
data["Age"].fillna(data["Age"].mean(), inplace=True)     # mean for numeric
data["City"].fillna(data["City"].mode()[0], inplace=True) # mode for categorical

# 3️⃣ Forward Fill / Backward Fill
data_ffill = data.fillna(method="ffill")
data_bfill = data.fillna(method="bfill")

# 4️⃣ Using SimpleImputer
num_imputer = SimpleImputer(strategy='median')
data["Age"] = num_imputer.fit_transform(data[["Age"]])

# 5️⃣ KNN Imputer (uses nearest neighbors to estimate missing values)
knn = KNNImputer(n_neighbors=2)
data_knn = knn.fit_transform(data[["Age"]])
print(pd.DataFrame(data_knn, columns=["Age"]))



#handling duplicate records
df_dup = pd.DataFrame({
    "ID": [1, 1, 2, 3, 3],
    "Name": ["Ali", "Ali", "Sara", "Ahmed", "Ahmed"]
})

# 1️⃣ Remove complete duplicate rows
df_no_dup = df_dup.drop_duplicates()

# 2️⃣ Remove duplicates based on specific column
df_no_dup_col = df_dup.drop_duplicates(subset=["ID"], keep="first")

# 3️⃣ Keep last occurrence
df_keep_last = df_dup.drop_duplicates(subset=["ID"], keep="last")

print(df_no_dup_col)


#feature scaling
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, Normalizer
import pandas as pd

data = pd.DataFrame({
    "Height": [150, 160, 170, 180],
    "Weight": [50, 60, 70, 80]
})

# 1️⃣ Min-Max Scaling → range [0, 1]
scaler = MinMaxScaler()
scaled_minmax = scaler.fit_transform(data)

# 2️⃣ Standard Scaling (Z-score normalization)
std = StandardScaler()
scaled_std = std.fit_transform(data)

# 3️⃣ Robust Scaling (less affected by outliers)
robust = RobustScaler()
scaled_robust = robust.fit_transform(data)

# 4️⃣ MaxAbs Scaling (scales by max absolute value)
maxabs = MaxAbsScaler()
scaled_maxabs = maxabs.fit_transform(data)

# 5️⃣ Normalization (each row unit norm)
normalizer = Normalizer()
scaled_norm = normalizer.fit_transform(data)

print(pd.DataFrame(scaled_std, columns=["Height", "Weight"]))



#outlier detection and treatment
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore

data = pd.DataFrame({"Score": [50, 52, 51, 100, 49, 48, 200]})

# 1️⃣ Detect outliers using Z-score
data["Zscore"] = zscore(data["Score"])
outliers = data[np.abs(data["Zscore"]) > 3]

# 2️⃣ Detect outliers using IQR
Q1 = data["Score"].quantile(0.25)
Q3 = data["Score"].quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = data[(data["Score"] < Q1 - 1.5 * IQR) | (data["Score"] > Q3 + 1.5 * IQR)]

# 3️⃣ Visual Detection
sns.boxplot(x=data["Score"])
plt.title("Boxplot for Outliers")
plt.show()

# 4️⃣ Outlier Treatment - remove or clip
data_clipped = data.copy()
data_clipped["Score"] = np.clip(data_clipped["Score"], Q1, Q3)

# 5️⃣ Log Transform (for right-skewed positive data)
data["Score_log"] = np.log1p(data["Score"])




#feature encoding
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import category_encoders as ce

data = pd.DataFrame({
    "Size": ["Small", "Medium", "Large", "Medium"],
    "City": ["Karachi", "Lahore", "Islamabad", "Lahore"]
})

# 1️⃣ Label Encoding (for ordinal data)
le = LabelEncoder()
data["Size_encoded"] = le.fit_transform(data["Size"])

# 2️⃣ One-Hot Encoding (for nominal data)
onehot = pd.get_dummies(data["City"], prefix="City")

# 3️⃣ Ordinal Encoding (manual mapping)
ordinal_map = {"Small": 1, "Medium": 2, "Large": 3}
data["Size_ordinal"] = data["Size"].map(ordinal_map)

# 4️⃣ Frequency Encoding
freq_encode = data["City"].value_counts().to_dict()
data["City_freq"] = data["City"].map(freq_encode)

# 5️⃣ Target Encoding (using mean of target variable)
df = pd.DataFrame({
    "City": ["Karachi", "Lahore", "Islamabad", "Lahore", "Karachi"],
    "Purchased": [1, 0, 1, 0, 1]
})
target_mean = df.groupby("City")["Purchased"].mean().to_dict()
df["City_target"] = df["City"].map(target_mean)

print(df)



#eda
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

data = pd.DataFrame({
    "Age": [22, 25, 30, 35, 28, 40],
    "Salary": [40000, 45000, 50000, 55000, 48000, 60000],
    "Gender": ["M", "F", "M", "F", "M", "F"]
})

# 1️⃣ Univariate Analysis
sns.histplot(data["Age"], kde=True)
plt.title("Distribution of Age")
plt.show()

# 2️⃣ Bivariate Analysis
sns.scatterplot(x="Age", y="Salary", hue="Gender", data=data)
plt.title("Age vs Salary")
plt.show()

# 3️⃣ Multivariate Analysis
sns.pairplot(data, hue="Gender")
plt.show()

# Correlation Heatmap
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()




#handling imabalanced data
import pandas as pd
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Creating imbalanced dataset
data = pd.DataFrame({
    "Feature": range(100),
    "Target": [0]*90 + [1]*10
})

# 1️⃣ Upsampling (Random)
majority = data[data.Target == 0]
minority = data[data.Target == 1]
minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
data_balanced = pd.concat([majority, minority_upsampled])

# 2️⃣ Downsampling
majority_downsampled = resample(majority, replace=False, n_samples=len(minority), random_state=42)
data_downsampled = pd.concat([majority_downsampled, minority])

# 3️⃣ SMOTE (synthetic minority oversampling)
X = data[["Feature"]]
y = data["Target"]
sm = SMOTE(random_state=42)
X_sm, y_sm = sm.fit_resample(X, y)
print(pd.Series(y_sm).value_counts())

# 4️⃣ NearMiss (undersampling method)
rus = RandomUnderSampler(random_state=42)
X_rus, y_rus = rus.fit_resample(X, y)
print(pd.Series(y_rus).value_counts())



#feature selection
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif, RFE, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

X, y = load_breast_cancer(return_X_y=True, as_frame=True)

# 1️⃣ Pearson Correlation (manual threshold)
corr = X.corr()
selected = corr[abs(corr["mean radius"]) > 0.65].index

# 2️⃣ Variance Threshold
vt = VarianceThreshold(threshold=0.05)
X_vt = vt.fit_transform(X)

# 3️⃣ ANOVA F-test
skb = SelectKBest(score_func=f_classif, k=10)
X_anova = skb.fit_transform(X, y)

# 4️⃣ Mutual Information
mi = SelectKBest(score_func=mutual_info_classif, k=10)
X_mi = mi.fit_transform(X, y)

# 5️⃣ Recursive Feature Elimination
rfe = RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=10)
X_rfe = rfe.fit_transform(X, y)

# 6️⃣ Tree-Based (Random Forest Importance)
selector = SelectFromModel(RandomForestClassifier(n_estimators=100))
selector.fit(X, y)
selected_features = X.columns[selector.get_support()]
print("Selected Features (Tree-based):", selected_features.tolist())


