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
