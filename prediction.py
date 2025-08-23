# House Price Prediction - Complete Pipeline
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------------
# Step 1: Load Dataset
# ------------------------------
# CSV version of Boston Housing dataset
url = 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv'
df = pd.read_csv(url)

# Check first few rows
print("First 5 rows of dataset:")
print(df.head())

# Check data info
print("\nDataset Info:")
print(df.info())

# Check for missing values
print("\nMissing values in dataset:")
print(df.isnull().sum())

# ------------------------------
# Step 2: Exploratory Data Analysis (EDA)
# ------------------------------
# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Pairplot of selected features
sns.pairplot(df[['medv', 'rm', 'lstat', 'ptratio']])
plt.show()

# ------------------------------
# Step 3: Preprocessing
# ------------------------------
# Split data into features and target
X = df.drop('medv', axis=1)  # Features
y = df['medv']               # Target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Preprocessing done. Shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# ------------------------------
# Step 4: Train Models
# ------------------------------
# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Random Forest Regression (optional, better performance)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# ------------------------------
# Step 5: Evaluate Models
# ------------------------------
def evaluate_model(y_test, y_pred, model_name="Model"):
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} Mean Squared Error: {mse:.2f}")
    print(f"{model_name} R2 Score: {r2:.2f}")

# Evaluate Linear Regression
evaluate_model(y_test, y_pred_lr, "Linear Regression")

# Evaluate Random Forest
evaluate_model(y_test, y_pred_rf, "Random Forest Regression")

# ------------------------------
# Step 6: Visualize Predictions
# ------------------------------
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_lr, color='blue', label='Linear Regression')
plt.scatter(y_test, y_pred_rf, color='green', label='Random Forest')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.show()

# ------------------------------
# Step 7: Insights
# ------------------------------
print("\nTop Features contributing to price prediction (Linear Regression coefficients):")
coefficients = pd.DataFrame({"Feature": X.columns, "Coefficient": lr_model.coef_})
print(coefficients.sort_values(by="Coefficient", ascending=False))
