# Task 1: Data Cleaning & Preprocessing - Titanic Dataset

## 📌 Objective
To clean and preprocess the Titanic dataset using key data science techniques. This task is part of an AI & ML internship assignment focused on real-world data handling.

---

## 🔧 Steps Performed

1. **Dataset Loading**: Titanic CSV was loaded using Pandas.
2. **Exploratory Data Analysis**:
   - Checked for null values and data types.
3. **Missing Value Handling**:
   - 'Age' and 'Fare': Filled with median.
   - 'Embarked': Filled with mode.
   - 'Cabin': Dropped due to high null count.
4. **Encoding**:
   - Categorical columns like 'Sex' and 'Embarked' encoded using Label Encoding.
5. **Feature Scaling**:
   - Applied Standardization to 'Age' and 'Fare' for better ML performance.
6. **Outlier Detection and Removal**:
   - Used IQR method to clean outliers from 'Age' and 'Fare'.
7. **Exported Clean Dataset**:
   - Saved the final cleaned CSV file.

---

## 📁 Files in this Repo

| File                      | Description                                 |
|---------------------------|---------------------------------------------|
| `Titanic-Dataset.csv`     | Original dataset                            |
| `task1_preprocessing.py`  | Python script with all preprocessing steps  |
| `Cleaned_Titanic_Dataset.csv` | Output file after preprocessing       |
| `README.md`               | This file                                   |

---

## 🧠 Key Learnings

- How to handle null values
- Difference between label encoding and one-hot encoding
- Importance of scaling and how it improves model learning
- Removing outliers to avoid skewed results

---

## ▶️ How to Run

```bash
python task1_preprocessing.py
