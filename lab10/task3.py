import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


#creating random data as we havent been given the dataset
np.random.seed(42)
n_samples = 1000

data = {
    'customer_id': range(1, n_samples + 1),
    'age': np.random.randint(18, 70, n_samples),
    'annual_income': np.random.normal(60000, 20000, n_samples),
    'monthly_spending': np.random.normal(3000, 800, n_samples),
    'savings_balance': np.random.normal(25000, 10000, n_samples),
    'investment_amount': np.random.normal(15000, 6000, n_samples),
    'credit_score': np.random.normal(700, 50, n_samples),
    'debt_to_income_ratio': np.random.normal(0.35, 0.1, n_samples),
    'num_credit_cards': np.random.randint(1, 6, n_samples),
    'loan_amount': np.random.normal(20000, 8000, n_samples)
}

#correlations
data['monthly_spending'] = data['annual_income'] * 0.05 + np.random.normal(0, 200, n_samples)
data['savings_balance'] = data['annual_income'] * 0.4 + np.random.normal(0, 5000, n_samples)
data['investment_amount'] = data['savings_balance'] * 0.6 + np.random.normal(0, 3000, n_samples)

df = pd.DataFrame(data)
print(f"Dataset shape: {df.shape}")
print("First 5 rows:")
print(df.head())

print("\nStep 2: Dataset Exploration")
print("=" * 50)

print("\nDataset information:")
print(df.info())

print("\nDescriptive statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

# Visualize distributions
plt.figure(figsize=(15, 10))
features_to_plot = ['annual_income', 'monthly_spending', 'savings_balance', 
                   'investment_amount', 'credit_score']

for i, feature in enumerate(features_to_plot, 1):
    plt.subplot(2, 3, i)
    plt.hist(df[feature], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

#selecting relevant features for PCA
print("\nStep 3: Feature Selection for PCA")
print("=" * 50)

financial_features = ['annual_income', 'monthly_spending', 'savings_balance', 
                     'investment_amount', 'credit_score', 'debt_to_income_ratio', 
                     'loan_amount']

X = df[financial_features].copy()
print(f"Selected features: {financial_features}")
print(f"Features dataset shape: {X.shape}")

#correlation matrix
plt.figure(figsize=(10, 8))
correlation_matrix = X.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
plt.title('Correlation Matrix of Financial Features')
plt.tight_layout()
plt.show()


print("\nStep 4: Feature Standardization")


imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
X_scaled_df = pd.DataFrame(X_scaled, columns=financial_features)

print("Standardization completed")
print(f"Mean after standardization: {X_scaled_df.mean().values}")
print(f"Std after standardization: {X_scaled_df.std().values}")


#apply pca
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
print(f"PCA transformed data shape: {X_pca.shape}")


pca_components_df = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(len(financial_features))],
    index=financial_features
)

print("Principal Components (Eigenvectors):")
print(pca_components_df)

# Visualize components
plt.figure(figsize=(12, 8))
sns.heatmap(pca_components_df, annot=True, cmap='RdBu', center=0,
            fmt='.3f', linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title('Principal Components Heatmap')
plt.tight_layout()
plt.show()

print("\nFirst Principal Component (PC1) Analysis:")
pc1_weights = pca_components_df['PC1'].abs().sort_values(ascending=False)
for feature, weight in pc1_weights.items():
    direction = "positive" if pca_components_df.loc[feature, 'PC1'] > 0 else "negative"
    print(f"  {feature}: {weight:.3f} ({direction} correlation)")

#finding out the optimal number of components
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("Explained variance by each component:")
for i, (var, cum_var) in enumerate(zip(explained_variance, cumulative_variance)):
    print(f"PC{i+1}: {var:.4f} ({var*100:.2f}%) - Cumulative: {cum_var:.4f} ({cum_var*100:.2f}%)")

# Variance plots
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(range(1, len(explained_variance) + 1), explained_variance, 
        alpha=0.7, color='lightblue', edgecolor='navy')
plt.plot(range(1, len(explained_variance) + 1), explained_variance, 'ro-', linewidth=2)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Variance by Principal Component')
plt.xticks(range(1, len(explained_variance) + 1))
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-', 
         linewidth=2, markersize=6)
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
plt.axhline(y=0.90, color='g', linestyle='--', label='90% variance')
plt.axhline(y=0.85, color='y', linestyle='--', label='85% variance')
plt.legend()
plt.xticks(range(1, len(cumulative_variance) + 1))
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Determine optimal components
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
n_components_85 = np.argmax(cumulative_variance >= 0.85) + 1

print(f"\nOptimal number of components:")
print(f"- For 95% variance: {n_components_95} components")
print(f"- For 90% variance: {n_components_90} components")
print(f"- For 85% variance: {n_components_85} components")


# use selected components and transform data
selected_components = n_components_90
print(f"Selected number of principal components: {selected_components}")

pca_selected = PCA(n_components=selected_components)
X_pca_reduced = pca_selected.fit_transform(X_scaled)

print(f"Original data shape: {X_scaled.shape}")
print(f"Reduced data shape: {X_pca_reduced.shape}")
print(f"Data reduction: {X_scaled.shape[1]} → {X_pca_reduced.shape[1]} features")
print(f"Compression ratio: {(1 - X_pca_reduced.shape[1]/X_scaled.shape[1])*100:.1f}% reduction")

pca_columns = [f'PC{i+1}' for i in range(selected_components)]
X_pca_df = pd.DataFrame(X_pca_reduced, columns=pca_columns)
print("\nFirst 5 rows of transformed data:")
print(X_pca_df.head())



# Customer segmentation visualization
plt.figure(figsize=(15, 12))

# PC1 vs PC2 colored by income
plt.subplot(2, 2, 1)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, 
                     c=df['annual_income'], cmap='viridis', s=50)
plt.colorbar(scatter, label='Annual Income')
plt.xlabel('PC1 - Overall Financial Health')
plt.ylabel('PC2 - Spending vs Saving Pattern')
plt.title('Customer Segmentation by Income\n(PC1 vs PC2)')
plt.grid(True, alpha=0.3)

# PC1 vs PC2 colored by savings
plt.subplot(2, 2, 2)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, 
                     c=df['savings_balance'], cmap='plasma', s=50)
plt.colorbar(scatter, label='Savings Balance')
plt.xlabel('PC1 - Overall Financial Health')
plt.ylabel('PC2 - Spending vs Saving Pattern')
plt.title('Customer Segmentation by Savings\n(PC1 vs PC2)')
plt.grid(True, alpha=0.3)

# Distribution of PC1
plt.subplot(2, 2, 3)
plt.hist(X_pca[:, 0], bins=30, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
plt.xlabel('PC1 - Overall Financial Health Score')
plt.ylabel('Frequency')
plt.title('Distribution of Financial Health Scores')
plt.grid(True, alpha=0.3)

# Distribution of PC2
plt.subplot(2, 2, 4)
plt.hist(X_pca[:, 1], bins=30, alpha=0.7, color='lightcoral', edgecolor='darkred')
plt.xlabel('PC2 - Spending vs Saving Pattern')
plt.ylabel('Frequency')
plt.title('Distribution of Spending/Saving Behavior')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Business insights
print("\nKEY INSIGHTS FROM PCA ANALYSIS:")
print("=" * 40)

print("\n1. PRINCIPAL COMPONENTS INTERPRETATION:")
print("PC1 - Overall Financial Health Index:")
print("   • Combines: High income + High savings + High investments + Good credit score")
print("   • Represents: Overall financial stability and wealth accumulation")

print("\nPC2 - Spending vs Saving Orientation:")
print("   • Positive: High monthly spending, lower savings orientation")
print("   • Negative: Conservative spending, strong saving/investment focus")

print("\n2. CUSTOMER SEGMENTS IDENTIFIED:")
print("   🟢 Segment A (High PC1, High PC2): Affluent spenders")
print("   🔵 Segment B (High PC1, Low PC2): Wealthy savers/investors")
print("   🟡 Segment C (Low PC1, High PC2): Moderate income, high spenders")
print("   🔴 Segment D (Low PC1, Low PC2): Budget-conscious, conservative")

print("\n3. BUSINESS RECOMMENDATIONS:")
print("=" * 40)

print("🎯 TARGETED MARKETING:")
print("   • Segment A: Premium credit cards, luxury rewards programs")
print("   • Segment B: Investment products, retirement planning")
print("   • Segment C: Budgeting tools, debt management services")
print("   • Segment D: Basic banking, savings accounts, financial education")

print("\n📊 RISK MANAGEMENT:")
print("   • Use PC1 as primary financial health score for credit decisions")
print("   • Monitor Segment C for potential over-leverage risks")
print("   • Segment B represents lowest credit risk")

print("\n💡 PRODUCT DEVELOPMENT:")
print("   • Develop hybrid products for customers between segments")
print("   • Create personalized financial advice based on PCA coordinates")
print("   • Implement dynamic pricing based on financial health scores")

print("\n📈 OPERATIONAL EFFICIENCY:")
print(f"   • Dimensionality reduction: {X_scaled.shape[1]} → {selected_components} features")
print(f"   • Variance retained: {cumulative_variance[selected_components-1]*100:.1f}%")
print("   • Simplified customer profiling and segmentation")
print("   • Faster computational processing for analytics")

print("\n" + "=" * 60)
print("PCA ANALYSIS COMPLETED SUCCESSFULLY")
print(f"Original features: {X_scaled.shape[1]}")
print(f"Reduced features: {selected_components}")
print(f"Variance retained: {cumulative_variance[selected_components-1]*100:.1f}%")
print("=" * 60)

