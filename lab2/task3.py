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
