import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load the dataset
data = pd.read_csv('../dataset/hospital_data.csv')

# 2. Preprocess the data (Encode text to numbers)
le_disease = LabelEncoder()
le_city = LabelEncoder()
le_type = LabelEncoder()

data['Disease'] = le_disease.fit_transform(data['Disease'])
data['City'] = le_city.fit_transform(data['City'])
data['Hospital_Type'] = le_type.fit_transform(data['Hospital_Type'])

X = data[['Disease', 'City', 'Hospital_Type']]
y = data['Cost']

# 3. Split the data into Training and Testing sets (80% train, 20% test)
# This is how we "validate" the machine predicting unseen results!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize all models
linear_model = LinearRegression()
decision_tree_model = DecisionTreeRegressor(random_state=5)
random_forest_model = RandomForestRegressor(random_state=42)

# 5. Train all models
linear_model.fit(X_train, y_train)
decision_tree_model.fit(X_train, y_train)
random_forest_model.fit(X_train, y_train)

# 6. Make Predictions on the test data
linear_predictions = linear_model.predict(X_test)
tree_predictions = decision_tree_model.predict(X_test)
forest_predictions = random_forest_model.predict(X_test)

# 7. Calculate Accuracy (R-squared) and Error (Mean Absolute Error)
linear_r2 = r2_score(y_test, linear_predictions)
linear_mae = mean_absolute_error(y_test, linear_predictions)

tree_r2 = r2_score(y_test, tree_predictions)
tree_mae = mean_absolute_error(y_test, tree_predictions)

forest_r2 = r2_score(y_test, forest_predictions)
forest_mae = mean_absolute_error(y_test, forest_predictions)

# 8. Print Results!
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

