"""
Hospital Cost Estimator - Model Comparison & Validation Utility
==============================================================
This script serves as a diagnostics tool. It evaluates and compares the accuracy 
of three different Machine Learning algorithms:
1. Linear Regression (baseline parametric model)
2. Decision Tree Regressor (non-linear tree structure)
3. Random Forest Regressor (ensemble method aggregating multiple trees)

It splits the source dataset into training (80%) and testing (20%) sets. 
Testing on unseen data helps evaluate how well the models will generalize 
to new user-submitted records in production.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# 1. Load the dataset
# ==========================================
data = pd.read_csv('../dataset/hospital_data.csv')

# ==========================================
# 2. Preprocess the data (Encode text categories to numbers)
# ==========================================
le_disease = LabelEncoder()
le_city = LabelEncoder()
le_type = LabelEncoder()

data['Disease'] = le_disease.fit_transform(data['Disease'])
data['City'] = le_city.fit_transform(data['City'])
data['Hospital_Type'] = le_type.fit_transform(data['Hospital_Type'])

# Input features (Disease, City, Hospital Type) and prediction target (Cost)
X = data[['Disease', 'City', 'Hospital_Type']]
y = data['Cost']

# ==========================================
# 3. Train-Test Split (80% Training, 20% Testing)
# ==========================================
# random_state=42 guarantees that the splits are identical across multiple runs.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 4. Initialize Models
# ==========================================
linear_model = LinearRegression()
decision_tree_model = DecisionTreeRegressor(random_state=5)
random_forest_model = RandomForestRegressor(random_state=42)

# ==========================================
# 5. Fit (Train) Models on the training set
# ==========================================
linear_model.fit(X_train, y_train)
decision_tree_model.fit(X_train, y_train)
random_forest_model.fit(X_train, y_train)


# ==========================================
# 6. Predict on the test set 
# ==========================================
linear_predictions = linear_model.predict(X_test)
tree_predictions = decision_tree_model.predict(X_test)
forest_predictions = random_forest_model.predict(X_test)

# ==========================================
# 7. Evaluate and Compare Performance Metrics
# ==========================================
# - R-Squared Score (coefficient of determination): Measures the percentage of variance
#   in the target variable explained by the model features. Closer to 1.0 (or 100%) is better.
# - Mean Absolute Error (MAE): Measures the average absolute difference between predicted costs 
#   and actual costs. Lower error is better.

# Linear Regression evaluation
linear_r2 = r2_score(y_test, linear_predictions)
linear_mae = mean_absolute_error(y_test, linear_predictions)

# Decision Tree evaluation
tree_r2 = r2_score(y_test, tree_predictions)
tree_mae = mean_absolute_error(y_test, tree_predictions)

# Random Forest evaluation
forest_r2 = r2_score(y_test, forest_predictions)
forest_mae = mean_absolute_error(y_test, forest_predictions)

# ==========================================
# 8. Output Results 
# ==========================================
print("=== Validation Results ===")
print("\n[1] Linear Regression:")
print(f"    Accuracy (R-Squared): {linear_r2 * 100:.2f}%")
print(f"    Average Prediction Error: +/- {linear_mae:.2f}")

print("\n[2] Decision Tree Regressor:")
print(f"    Accuracy (R-Squared): {tree_r2 * 100:.2f}%")
print(f"    Average Prediction Error: +/- {tree_mae:.2f}")

print("\n[3] Random Forest Regressor:")
print(f"    Accuracy (R-Squared): {forest_r2 * 100:.2f}%")
print(f"    Average Prediction Error: +/- {forest_mae:.2f}")

print("\nConclusion: The model with the higher Accuracy (%) and lower Average Error is better!")


