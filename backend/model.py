
"""
Hospital Cost Estimator - Model Training & Export Script
======================================================

This script loads the raw healthcare dataset, performs preprocessing by encoding 
textual/categorical features to numerical codes, trains a Random Forest Regressor 
model to estimate costs, and serializes both the trained model and label encoders 
to pickle files for subsequent API production usage.

Execute this script whenever:
1. The underlying dataset '../dataset/hospital_data.csv' is updated or replaced.
2. Model parameters (hyperparameters) or selected algorithms are modified.

"""


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# ==========================================
# 1. Dataset Loading  
# ==========================================
# Reads the training dataset containing records of Disease, City, Hospital_Type, Cost, and Hospital_Name
data = pd.read_csv('../dataset/hospital_data.csv')

# ==========================================
# 2. Preprocessing (Label Encoding)
# ==========================================
# Since Machine Learning algorithms require numerical inputs, LabelEncoder transforms
# non-numerical categorical columns (like "Malaria", "Mangaluru", "Govt") into integer values (0, 1, 2, ...).
le_disease = LabelEncoder()
le_city = LabelEncoder()
le_type = LabelEncoder()

# Fit the encoders on the categories and transform the columns
data['Disease'] = le_disease.fit_transform(data['Disease'])
data['City'] = le_city.fit_transform(data['City'])
data['Hospital_Type'] = le_type.fit_transform(data['Hospital_Type'])

# ==========================================
# 3. Feature Selection & Target Definition
# ==========================================
# X: Independent variables (inputs)
X = data[['Disease', 'City', 'Hospital_Type']]
# y: Target variable to predict (output)
y = data['Cost']

# ==========================================
# 4. Model Selection & Fitting
# ==========================================
# Let's use Random Forest Regressor: An ensemble learning method that trains multiple 
# decision trees and averages their predictions for better generalizability and accuracy.
model = RandomForestRegressor(random_state=42) # Set seed for reproducible training
model.fit(X, y)

# ==========================================
# 5. Serialization and File Export
# ==========================================
# Ensure output model directory exists
os.makedirs('../model', exist_ok=True)

# Save the trained model and the encoders using Pickle.
# Saving the encoders is critical because any new user input string must be converted
# to the exact same numerical format during inference.
pickle.dump(model, open('../model/model.pkl', 'wb'))
pickle.dump(le_disease, open('../model/le_disease.pkl', 'wb'))
pickle.dump(le_city, open('../model/le_city.pkl', 'wb'))
pickle.dump(le_type, open('../model/le_type.pkl', 'wb'))

print("Model trained and saved successfully!")
