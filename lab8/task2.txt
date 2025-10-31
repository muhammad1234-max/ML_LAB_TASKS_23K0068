Assumptions of Common Classification Algorithms
1. Logistic Regression

• Linearity: Assumes a linear relationship between the independent variables and the log-odds of the dependent variable.
• Independence: Observations should be independent of each other.
• No or Little Multicollinearity: Predictor variables should not be highly correlated.
• Large Sample Size: Performs better with large datasets to ensure stable estimates.
• Normally Distributed Errors: Assumes residuals are approximately normally distributed.

2. K-Nearest Neighbors (KNN)

• No Specific Distribution: KNN is non-parametric and makes no assumptions about data distribution.
• Feature Scaling: Features should be normalized or standardized for distance-based computation.
• Relevant Features: Works best when irrelevant features are minimized or removed.
• Balanced Data: Sensitive to imbalanced class distributions.

3. Support Vector Machine (SVM)

• Linearity (for Linear SVM): Assumes data is linearly separable, or near-linear with a soft margin.
• Independence: Observations should be independent and representative of the population.
• Feature Scaling: Features must be scaled for optimal margin separation.
• No Multicollinearity: Highly correlated features can affect the decision boundary.
• Choice of Kernel: The kernel function must appropriately capture non-linear relationships (if used).

4. Naive Bayes

• Conditional Independence: Assumes all features are independent given the class label.
• Distribution Assumption: Each variant assumes a distribution (e.g., Gaussian NB assumes normal distribution).
• Feature Relevance: Irrelevant or redundant features can degrade performance.
• No Multicollinearity: Strong correlations between features can violate the independence assumption.

5. Decision Tree

• No Linearity Required: Does not assume linear relationships between variables.
• No Normality or Homoscedasticity: Works well without distributional assumptions.
• Independence: Assumes training samples are independent.
• Feature Relevance: Handles both categorical and numerical features effectively.
• No Multicollinearity Assumption: However, correlated features may reduce interpretability.

6. Random Forest

• Independence: Observations should be independent.
• No Specific Distribution: Non-parametric and does not assume feature distributions.
• Balanced Data: Works best with balanced datasets.
• Feature Importance: Performs better when important features are present.
• Reduced Multicollinearity Impact: Random feature selection reduces collinearity effects.

7. Neural Networks

• Large Data Requirement: Needs a large dataset for effective learning.
• Feature Scaling: Input features should be normalized or standardized.
• Independence: Observations should be independent.
• No Multicollinearity: Highly correlated features can slow down learning.
• Data Stationarity (for time series): Input data should be stable over time.

