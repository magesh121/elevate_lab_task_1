# task1_preprocessing.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Load Dataset
df = pd.read_csv('Titanic-Dataset.csv')

# 2. Basic Exploration
print("Shape:", df.shape)
print("Info:\n", df.info())
print("Null values:\n", df.isnull().sum())

# 3. Handling Missing Values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop 'Cabin' due to excessive missing values
df.drop(columns=['Cabin'], inplace=True)

# 4. Encoding Categorical Columns
label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])  # male:1, female:0
df['Embarked'] = label_encoder.fit_transform(df['Embarked'])

# 5. Feature Scaling (Standardization)
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# 6. Outlier Visualization and Removal
def remove_outliers(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[col] >= lower) & (data[col] <= upper)]

# Before removing outliers
sns.boxplot(df['Fare'])
plt.title("Fare Before Outlier Removal")
plt.show()

df = remove_outliers(df, 'Fare')
df = remove_outliers(df, 'Age')

# After removing outliers
sns.boxplot(df['Fare'])
plt.title("Fare After Outlier Removal")
plt.show()

# Final Dataset Summary
print("Final Shape:", df.shape)
print("Cleaned Dataset Preview:\n", df.head())

# Save Cleaned Data
df.to_csv('Cleaned_Titanic_Dataset.csv', index=False)
print("✅ Preprocessing Complete. Cleaned dataset saved.")
