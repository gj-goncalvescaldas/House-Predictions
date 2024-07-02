
# House Prices Prediction using the Ames Housing Dataset

This repository contains a Jupyter notebook that guides you through the process of predicting house prices using the Ames Housing dataset. The notebook uses Python for data exploration, cleaning, feature engineering, and modeling.

## Introduction

This project aims to predict house prices using the Ames Housing dataset. We will use various Python libraries to explore, clean, and model the data, and ultimately make predictions on house prices.

## Dataset Description

The dataset includes various features related to the properties such as sale price, building class, zoning classification, lot area, street type, and many more. These features are used to predict the sale price of the properties.

## Getting Started

### Prerequisites

Make sure you have the following libraries installed:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

You can install these using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Data Loading

Load the dataset using pandas:

```python
import pandas as pd

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
```

### Data Exploration

Explore the dataset to understand its structure and summary statistics:

```python
train.head()
train.describe()
```

### Handling Missing Values

Handle missing values by filling numerical features with the median and categorical features with the mode:

```python
# Fill missing values for numerical features with the median
num_features = train.select_dtypes(include=[np.number])
num_features = num_features.fillna(num_features.median())

# Fill missing values for categorical features with the mode
cat_features = train.select_dtypes(include=[object])
cat_features = cat_features.fillna(cat_features.mode().iloc[0])

# Combine numerical and categorical features
train_cleaned = pd.concat([num_features, cat_features], axis=1)
```

### Feature Engineering

Create new features to improve the model:

```python
train_cleaned['TotalBathrooms'] = (train_cleaned['FullBath'] +
                                   0.5 * train_cleaned['HalfBath'] +
                                   train_cleaned['BsmtFullBath'] +
                                   0.5 * train_cleaned['BsmtHalfBath'])
train_cleaned['HouseAge'] = train_cleaned['YrSold'] - train_cleaned['YearBuilt']
train_cleaned['RemodelAge'] = train_cleaned['YrSold'] - train_cleaned['YearRemodAdd']
train_cleaned['IsNew'] = (train_cleaned['YearBuilt'] == train_cleaned['YrSold']).astype(int)
```

### Encoding Categorical Features

Convert categorical features into dummy variables:

```python
train_final = pd.get_dummies(train_cleaned)
```

### Data Visualization

Visualize the distribution of house prices:

```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.histplot(train_final['SalePrice'], kde=True, bins=30)
plt.title('Distribution of House Prices')
plt.xlabel('Sale Price')
plt.ylabel('Frequency')
plt.show()
```

### Model Training

Split the data into training and validation sets, and train a Random Forest model:

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Splitting the data into features and target variable
X = train_final.drop('SalePrice', axis=1)
y = np.log1p(train_final['SalePrice'])  # Use log1p for better handling of skewed distribution

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)
```

### Model Evaluation

Evaluate the model using various metrics:

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Make predictions on the validation set
preds = model.predict(X_val)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_val, preds))
print(f'Root Mean Squared Error: {rmse}')

# Calculate MAE
mae = mean_absolute_error(y_val, preds)
print(f'Mean Absolute Error: {mae}')

# Calculate R² Score
r2 = r2_score(y_val, preds)
print(f'R² Score: {r2}')

# Calculate Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((y_val - preds) / y_val)) * 100
print(f'Mean Absolute Percentage Error: {mape}%')
```

## Interpretation of Results

- **RMSE (Root Mean Squared Error)**: This indicates the average deviation of the log-transformed predictions from the actual log-transformed house prices. Lower values indicate better performance.
- **MAE (Mean Absolute Error)**: This indicates the average absolute error between the predicted and actual log-transformed house prices.
- **R² Score**: Indicates the proportion of variance in the dependent variable that is predictable from the independent variables. Higher values indicate better performance.
- **MAPE (Mean Absolute Percentage Error)**: This indicates the average percentage error between the predicted and actual values. Lower values indicate better accuracy.

## Conclusion

This project demonstrates a step-by-step process to predict house prices using the Ames Housing dataset. By following this guide, you can understand the key steps involved in data preprocessing, feature engineering, model training, and evaluation.

Feel free to explore the notebook and modify it to improve the model's performance. Happy coding!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
