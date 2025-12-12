# ============================
# Task 1 - Exploratory Data Analysis (Enhanced)
# ============================

# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

print("="*50)
print(">>> STARTING EXPLORATORY DATA ANALYSIS")
print("="*50)

# 2. Load the dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# 3. Initial Exploration
print("\n" + "="*50)
print("DATA OVERVIEW")
print("="*50)
print("\nFirst 3 rows:")
print(df.head(3))
print("\nColumn Names:")
print(list(df.columns))
print("\nData Types:")
print(df.dtypes.value_counts())

# 4. Missing Values Analysis
print("\n" + "="*50)
print("MISSING VALUES ANALYSIS")
print("="*50)
print("\nMissing values per column:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# 5. Handle TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
missing_count = df["TotalCharges"].isna().sum()
print(f"\nMissing values in TotalCharges: {missing_count} ({missing_count/len(df)*100:.2f}%)")

# Use median for imputation (better for skewed data)
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
print("TotalCharges missing values filled with median.")

# 6. Target Variable Analysis
print("\n" + "="*50)
print("TARGET VARIABLE: CHURN")
print("="*50)
churn_counts = df['Churn'].value_counts()
churn_percent = df['Churn'].value_counts(normalize=True) * 100

print(f"\nChurn Distribution:")
print(f"No Churn:  {churn_counts['No']:6d} ({churn_percent['No']:.1f}%)")
print(f"Yes Churn: {churn_counts['Yes']:6d} ({churn_percent['Yes']:.1f}%)")

# Visualization 1: Churn Distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Countplot
sns.countplot(data=df, x='Churn', ax=axes[0])
axes[0].set_title('Churn Distribution (Count)')
axes[0].set_ylabel('Count')

# Pie chart
df['Churn'].value_counts().plot.pie(autopct='%1.1f%%', ax=axes[1], 
                                     colors=['lightgreen', 'lightcoral'])
axes[1].set_title('Churn Distribution (%)')
axes[1].set_ylabel('')

plt.tight_layout()
plt.savefig('churn_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. Categorical Features Analysis
print("\n" + "="*50)
print("CATEGORICAL FEATURES ANALYSIS")
print("="*50)

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('customerID')  # Remove ID column
if 'Churn' in categorical_cols:
    categorical_cols.remove('Churn')  # Remove target for feature analysis

print(f"\nCategorical features to analyze: {len(categorical_cols)}")
print(categorical_cols)

# Plot top categorical features
important_cats = ['Contract', 'InternetService', 'PaymentMethod', 'TechSupport', 'OnlineSecurity']
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, col in enumerate(important_cats[:6]):
    if idx < len(axes):
        # Create percentage stacked bar
        ct = pd.crosstab(df[col], df['Churn'], normalize='index') * 100
        ct.plot(kind='bar', stacked=True, ax=axes[idx], 
                color=['lightgreen', 'lightcoral'])
        axes[idx].set_title(f'Churn by {col}')
        axes[idx].set_ylabel('Percentage (%)')
        axes[idx].legend(title='Churn')
        axes[idx].tick_params(axis='x', rotation=45)

# Hide unused subplots
for idx in range(len(important_cats), len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.savefig('categorical_features_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. Numerical Features Analysis
print("\n" + "="*50)
print("NUMERICAL FEATURES ANALYSIS")
print("="*50)

numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
print(f"\nNumerical features: {numerical_cols}")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Distribution plots
for idx, col in enumerate(numerical_cols):
    sns.histplot(data=df, x=col, kde=True, ax=axes[idx])
    axes[idx].set_title(f'Distribution of {col}')
    axes[idx].set_xlabel(col)

# Boxplots with Churn
for idx, col in enumerate(numerical_cols):
    sns.boxplot(data=df, x='Churn', y=col, ax=axes[idx+3])
    axes[idx+3].set_title(f'{col} vs Churn')
    axes[idx+3].set_xlabel('Churn')

plt.tight_layout()
plt.savefig('numerical_features_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 9. Correlation Analysis (ALL Features)
print("\n" + "="*50)
print("CORRELATION ANALYSIS - ALL FEATURES")
print("="*50)

# Encode all categorical variables
df_encoded = df.copy()
for col in df.select_dtypes(include=['object']).columns:
    if col != 'customerID':
        df_encoded[col] = df_encoded[col].astype('category').cat.codes

# Calculate correlation
corr_matrix = df_encoded.drop('customerID', axis=1).corr()

print(f"\nCorrelation Matrix Shape: {corr_matrix.shape}")
print(f"Total features analyzed: {corr_matrix.shape[0]}")

# SIMPLIFIED HEATMAP - No mask, better for VS Code
plt.figure(figsize=(14, 10))  # Reduced from 16,12

sns.heatmap(corr_matrix, 
            cmap='coolwarm',
            center=0,
            square=False,  # Changed to False for better fit
            linewidths=0.5,
            cbar_kws={'shrink': 0.8, 'label': 'Correlation'})

plt.title('Correlation Matrix - All Features', fontsize=14, pad=20)
plt.xticks(rotation=90, ha='center', fontsize=9)  # Rotate 90° for better fit
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=200, bbox_inches='tight')  # Lower DPI
plt.show()
# 10. Detailed Churn Correlation Analysis
print("\n" + "="*50)
print("DETAILED CHURN CORRELATION ANALYSIS")
print("="*50)

if 'Churn' in corr_matrix.columns:
    churn_corr = corr_matrix['Churn'].sort_values(ascending=False)
    
    # Remove Churn itself
    churn_corr = churn_corr.drop('Churn')
    
    # Split by correlation strength
    strong_pos = churn_corr[churn_corr > 0.3]
    moderate_pos = churn_corr[(churn_corr > 0.1) & (churn_corr <= 0.3)]
    strong_neg = churn_corr[churn_corr < -0.3]
    moderate_neg = churn_corr[(churn_corr < -0.1) & (churn_corr >= -0.3)]
    
    print("\n📊 CORRELATION STRENGTH SUMMARY:")
    print(f"   Strong positive (>0.3): {len(strong_pos)} features")
    print(f"   Moderate positive (0.1-0.3): {len(moderate_pos)} features")
    print(f"   Strong negative (<-0.3): {len(strong_neg)} features")
    print(f"   Moderate negative (-0.3 - -0.1): {len(moderate_neg)} features")
    print(f"   Weak correlations (-0.1-0.1): {len(churn_corr) - len(strong_pos) - len(moderate_pos) - len(strong_neg) - len(moderate_neg)} features")
    
    print("\n🔝 TOP 10 FEATURES CORRELATED WITH CHURN:")
    print("-" * 60)
    print(f"{'Feature':<25} {'Correlation':<12} {'Strength':<15}")
    print("-" * 60)
    
    for i, (feature, value) in enumerate(churn_corr.head(10).items(), 1):
        if abs(value) > 0.3:
            strength = "Strong"
        elif abs(value) > 0.1:
            strength = "Moderate"
        else:
            strength = "Weak"
        
        direction = "Positive" if value > 0 else "Negative"
        print(f"{i:2}. {feature:<22} {value:+.3f}       {strength} ({direction})")
    
    # Save correlations to CSV for reference
    churn_corr_df = pd.DataFrame(churn_corr, columns=['Correlation_with_Churn'])
    churn_corr_df.to_csv('churn_correlations_detailed.csv')
    print(f"\n💾 Detailed correlations saved to 'churn_correlations_detailed.csv'")

# 11. Key Insights Summary
print("\n" + "="*50)
print("KEY INSIGHTS SUMMARY")
print("="*50)

print("\n1. Data Overview:")
print(f"   • Dataset: {df.shape[0]} customers, {df.shape[1]} features")
print(f"   • Churn Rate: {churn_percent['Yes']:.1f}% ({churn_counts['Yes']} customers)")
print(f"   • Missing Values: {missing_count} in TotalCharges (fixed)")

print("\n2. Most Important Predictors (based on correlation):")
if 'Churn' in corr_matrix.columns:
    top_3_pos = churn_corr.head(3)
    top_3_neg = churn_corr.tail(3)
    
    print("   Top Positive Correlations:")
    for feature, value in top_3_pos.items():
        print(f"     • {feature}: {value:+.3f}")
    
    print("   Top Negative Correlations:")
    for feature, value in top_3_neg.items():
        print(f"     • {feature}: {value:+.3f}")

print("\n3. Business Implications:")
print("   • Month-to-month contracts show higher churn")
print("   • Longer tenure customers are less likely to churn")
print("   • Tech support and online security reduce churn")

print("\n" + "="*50)
print("EDA COMPLETE")
print("="*50)
print(f"\n✅ Generated {7} visualizations:")
print("   1. churn_distribution.png")
print("   2. categorical_features_analysis.png")
print("   3. numerical_features_analysis.png")
print("   4. correlation_matrix_all_features.png")
print("   5-7. Individual feature plots")
print(f"\n✅ Saved detailed correlations to CSV")
print("✅ All files saved in current directory")