import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#random clustering dataset i picked up of kaggle
df = pd.read_csv("student_dropout_behavior_dataset.csv")

# Remove column with all NaN values as this dataset i have choosen has this issue
df = df.drop(columns=["assignments_submitted"])

# Select numeric features
features = [
    "quiz1_marks", "quiz2_marks", "quiz3_marks",
    "total_assignments",
    "midterm_marks", "final_marks",
    "previous_gpa",
    "total_lectures", "lectures_attended",
    "total_lab_sessions", "labs_attended"
]

X = df[features]

X = X.replace("", float("nan"))
X = X.fillna(X.mean())

#standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#elbow Method
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

#i choose this optimal number of k after observing the plot
optimal_k = 6
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df["cluster"] = kmeans.fit_predict(X_scaled)

plt.scatter(df["midterm_marks"], df["final_marks"], c=df["cluster"])
plt.xlabel("Midterm Marks")
plt.ylabel("Final Marks")
plt.title("Student Clusters")
plt.show()

print(df.groupby("cluster")[features].mean())
