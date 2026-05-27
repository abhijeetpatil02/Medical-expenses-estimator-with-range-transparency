
# this is after adding hospital name prediction
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Load dataset
data = pd.read_csv('../dataset/hospital_data.csv')

# Create encoders
le_disease = LabelEncoder()
le_city = LabelEncoder()
le_type = LabelEncoder()

# Convert text columns into numbers
data['Disease'] = le_disease.fit_transform(data['Disease'])
data['City'] = le_city.fit_transform(data['City'])
data['Hospital_Type'] = le_type.fit_transform(data['Hospital_Type'])

# Features and target
X = data[['Disease', 'City', 'Hospital_Type']]
y = data['Cost']

# Train model
# --- linear regression model ---
# model = LinearRegression()
# model.fit(X, y)

# Decision Tree Regressor Model
# model = DecisionTreeRegressor(random_state=42)
# model.fit(X, y)

# Random Forest Regressor Model
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Create model folder if not exists
os.makedirs('../model', exist_ok=True)

# Save model and encoders
pickle.dump(model, open('../model/model.pkl', 'wb'))
pickle.dump(le_disease, open('../model/le_disease.pkl', 'wb'))
pickle.dump(le_city, open('../model/le_city.pkl', 'wb'))
pickle.dump(le_type, open('../model/le_type.pkl', 'wb'))

print("Model trained and saved successfully!")




