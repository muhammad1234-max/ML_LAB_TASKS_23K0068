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



#lab tasks

# A. Download the dataset and explore how to merge these dataset
# B. Combine dataset Lab2 D1A with Lab2 D1B in such a way that it doesn’t contain any
# duplicate column. The resultant dataset consist of these columns and the final shape will be

# C. Combine dataset Lab2 D1A with Lab2 D1C using merge method to extract similar records
# in a new Dataframe “comboAC” having 4221333 records


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


# A. Customized you own dataset with the name “customizedData”, add at least one attribute
# that should be similar to Lab2 D1A, Lab2 D1B, Lab2 D1C dataset , now add 3 attributes
# of Size (small, ,medium, and high), cardinal direction ( North, South, East and West) ,
# Timings (full time , part time) and add 2 attributes of your own choice, one attribute should
# be categorical and one should be continuous.
# B. Merge “customizedData” with Lab2 D1A, Lab2 D1B, Lab2 D1C and produce a resultand
# dataset with the name of “modifiedData” and explore/ analyze its number of records and
# features before and after merging with the technique of similar records joining.

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


# How to organize your code (Create a Text block for Importing Libraries in next cell import
# required library now Create a Heading for Data Preprocessing and the add a new cell for Data
# analysis. Create headings for each cell)
# A. Download the dataset from Here
# B. Import your dataset in Colab or Jupyter notebook
# C. Calculate the correlation between these variables,Var3,Var38,Var15,
# imp_op_var39_comer_ult1
# D. Check whether the data is linear or not and write a brief explanation what you
# have analyzed in text cell
# E. Check whether the data contain any missing record, if yes then perform imputation using
# an average method.
# F. In your dataset, you have some interesting variables. Think of multi-variable research
# questions that you can explore with these data and explore. You need to do at least 5
# explorations that include data visualizations, numerical summary.
# G. Find out the unique category in target variable and check whether your dataset is balanced
# or not.
# H. If dataset is not balanced, then handle your dataset and balance it using Up sampling
# I. Find out the total number of features and records and perform feature selection using
# Pearson Correlation having threshold equal to 65%.
# J. Make a copy of your dataset and perform feature selection other than Pearson and Variance
# threshold.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from sklearn.feature_selection import mutual_info_classif

dataset = pd.read_csv("train.csv")  

#Correlations between selected variables
corr_vars = dataset[["var3", "var38", "var15", "imp_op_var39_comer_ult1"]]
print("Correlation Matrix:\n", corr_vars.corr())

#linearity check with scatter plots + heatmap
sns.pairplot(corr_vars)
plt.show()
sns.heatmap(corr_vars.corr(), annot=True, cmap="coolwarm")
plt.show()

#handling missing values with mean imputation
imputer = SimpleImputer(strategy="mean")
dataset_imputed = pd.DataFrame(imputer.fit_transform(dataset), columns=dataset.columns)

#example explorations
sns.histplot(dataset_imputed["var38"], bins=30, kde=True)
plt.show()

sns.boxplot(x=dataset_imputed["var3"], y=dataset_imputed["var38"])
plt.show()

sns.scatterplot(x=dataset_imputed["var15"], y=dataset_imputed["var38"], hue=dataset_imputed["var3"])
plt.show()

#checking class balance
print("Target distribution:\n", dataset["TARGET"].value_counts())

#handling imbalance with upsampling
majority = dataset[dataset["TARGET"] == 0]
minority = dataset[dataset["TARGET"] == 1]
minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
balanced_dataset = pd.concat([majority, minority_upsampled])
print("Balanced dataset shape:", balanced_dataset.shape)

#feature selection with Pearson correlation
cor_matrix = balanced_dataset.corr()
selected_features = cor_matrix.index[abs(cor_matrix["TARGET"]) > 0.65]
print("Features selected (Pearson > 65%):", selected_features)

#feature selection using Mutual Information
X = balanced_dataset.drop("TARGET", axis=1).fillna(0)
y = balanced_dataset["TARGET"]
mi_scores = mutual_info_classif(X, y)
mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
print("Top MI Features:\n", mi_series.head())


#using a scatterplot we can analyze that the data in the train.csv dataset is not linear as the data points do not follow the 
#pattern of a straight line or any semblance of it as well



# A. Create a survey form (ask for the approval in order to avoid duplicate content) , ask some
# questions , Make sure to choose your question wisely, your attributes should reflect more to
# your problem statement. User have to answer atleast 5 questions and the remaining one will
# be depend on user, whether to answer or not.
# B. The questionnaire will contains at least 10 questions and you have to collect dataset from
# minimum 100 individuals
# C. After collecting dataset, Perform some Statistical analysis over it, Do some Graphical
# Visualization, Check whether the dataset you have collected is balanced or not.
# D. Perform Data wrangling, If the dataset contain any missing records try to handle these
# missing values wisely.
# E. If you found your dataset is not balanced then choose a technique other than smote or
# NearMiss
# F. Perform feature selection technique other than Variance Threshold and Pearson correlation
# and explain in a text cell its working.
# G. If the dataset contain any categorical feature then encode it using Dummy Encoding, and
# explain the difference between dummy encoding and one hot encoding
# H. Check whether your dataset contain any duplicate records, if it does, then handle these
# records with atleast 2 techniques.


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.feature_selection import chi2

#a theoretical file that will contain data from my survey
survey_df = pd.read_csv("SurveyData.csv")  

#Statistical summary
print(survey_df.describe(include="all"))

#visualization 
sns.countplot(x="Q1", data=survey_df)
plt.show()


survey_df.fillna(survey_df.mode().iloc[0], inplace=True)


class_counts = survey_df["TARGET"].value_counts()
minority = survey_df[survey_df["TARGET"] == class_counts.idxmin()]
majority = survey_df[survey_df["TARGET"] == class_counts.idxmax()]
majority_downsampled = resample(majority, replace=False, n_samples=len(minority), random_state=42)
balanced_survey = pd.concat([minority, majority_downsampled])

#feature selection with Chi-Square
categorical_features = pd.get_dummies(balanced_survey.drop("TARGET", axis=1))
chi_scores, p_vals = chi2(categorical_features, balanced_survey["TARGET"])
chi_series = pd.Series(chi_scores, index=categorical_features.columns).sort_values(ascending=False)
print("Top Chi-Square Features:\n", chi_series.head())

#dummy Encoding
dummy_encoded = pd.get_dummies(survey_df, drop_first=True)
print("Dummy Encoded Shape:", dummy_encoded.shape)

#handle duplicates
survey_df = survey_df.drop_duplicates()                        
survey_df = survey_df.drop_duplicates(subset=["Q1", "Q2"])     


#survey example
# Questions:

# Q1: Age (Required)

# Q2: Gender (Required)

# Q3: Study Hours per Day (Required)

# Q4: Do you have a part-time job? (Required)

# Q5: GPA Range (Required)

# Q6: Do you exercise regularly? (Optional)

# Q7: Preferred Study Time (Optional)

# Q8: Internet Usage (Optional)

# Q9: Preferred Learning Mode (Optional)

# Q10: Academic Life Satisfaction (Optional)

