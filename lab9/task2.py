import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#load iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#for comparison 
solvers = ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']

results = []

for solver in solvers:
    try:
        model = LogisticRegression(
            solver=solver,
            max_iter=1000
        )
        model.fit(X_train_scaled, y_train)

        train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
        test_acc = accuracy_score(y_test, model.predict(X_test_scaled))

        results.append([solver, train_acc, test_acc])

    except Exception as e:
        # Some solvers may not support multinomial by default
        results.append([solver, "Error", str(e)])


df_results = pd.DataFrame(
    results,
    columns=["Solver", "Training Accuracy", "Testing Accuracy"]
)

print(df_results)
