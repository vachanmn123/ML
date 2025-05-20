import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

california_housing = fetch_california_housing()
df = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)

print(df.head())

print("Shape of DataFrame:", df.shape)  # (rows, columns)

print("\nDataFrame Info:")
df.info()

print("\nNumber of unique values in each column:")
print(df.nunique())

print("\nNumber of missing values in each column:")
print(df.isnull().sum())

print("\nDescriptive statistics:")
print(df.describe().T)

df.hist(figsize=(12, 10), bins=20, color="skyblue", edgecolor="black")
plt.suptitle(
    "Histogram of numerical features of California Housing dataset",
    fontsize=16,
)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout for title
plt.show()

plt.figure(figsize=(15, 8))
for i, feature in enumerate(df.columns):
    plt.subplot(2, 4, i + 1)  # Arrange in 2 rows and 4 columns
    sns.boxplot(data=df, x=feature)
    plt.title(f"Box Plot of {feature}")
plt.tight_layout()
plt.show()


def detect_outliers(df):
    outliers = {}
    for feature in df.columns:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[feature] = df[
            (df[feature] < lower_bound) | (df[feature] > upper_bound)
        ]
    return outliers


outliers = detect_outliers(df)
for feature, outlier_data in outliers.items():
    print(f"\nOutliers in {feature} (showing first 5 rows):")
    print(outlier_data.head())
