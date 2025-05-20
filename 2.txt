import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()

df = pd.DataFrame(data.data, columns=data.feature_names)
df["Target"] = data.target

print("DataFrame shape:", df.shape)
print(df.describe().T)
print("Missing values per column:\n", df.isnull().sum())

corr_matrix = df.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

sns.pairplot(df[["MedInc", "HouseAge", "AveRooms", "Target"]], diag_kind="kde")
plt.show()
