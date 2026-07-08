"""
Hospital Cost Estimator - Main Flask Server & Inference Application
==================================================================

This is the main entry point for the backend server. It executes the following roles:
1. Loads the source hospital dataset and trains two ML models on startup:
   - RandomForestRegressor: Predicts the cost of a disease given a city and hospital type.
   - RandomForestClassifier: Recommends the specific hospital name.
2. Serves the static HTML, CSS, and image files to the frontend client.
3. Provides a search suggestions endpoint (/api/suggestions) for UI autocompletes.
4. Hosts the core cost estimator endpoint (/predict) which matches spelling errors 
   using difflib and performs inference.
   
5. Implements user signups and logins in coordination with the database module.
"""

from database import get_db_connection
from flask import Flask, render_template, request, jsonify, send_from_directory, session, redirect
from flask_cors import CORS
import difflib
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Determine the paths for the templates and static resources
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, 'frontend')

# Instantiate Flask app and map it to look in the frontend folder for html and css files
app = Flask(
    __name__,
    template_folder=FRONTEND_DIR,  # Finds auth.html and index.html
    static_folder=FRONTEND_DIR,    # Finds auth.css, style.css, logo.png, etc.
    static_url_path=''
)
CORS(app) # Enable Cross-Origin Resource Sharing for API accessibility

# =====================================================================
# 1. Machine Learning Setup & Training (On Server Startup)
# =====================================================================

# Load the source dataset from the CSV file
raw_data = pd.read_csv("../dataset/hospital_data.csv")
data = raw_data.copy()

# Initialize scikit-learn Label Encoders to translate strings into numbers
le_disease = LabelEncoder()
le_city = LabelEncoder()
le_type = LabelEncoder()
le_hospital = LabelEncoder()

# Fit and transform the categorical textual columns to continuous integers
data["Disease"] = le_disease.fit_transform(data["Disease"])
data["City"] = le_city.fit_transform(data["City"])
data["Hospital_Type"] = le_type.fit_transform(data["Hospital_Type"])
data["Hospital_Name"] = le_hospital.fit_transform(data["Hospital_Name"])

# Extract feature columns (inputs) and target columns (outputs) 
X = data[["Disease", "City", "Hospital_Type"]]
y_cost = data["Cost"]
y_hospital = data["Hospital_Name"]

# Train the Cost Predictor Regressor Model
cost_model = RandomForestRegressor()
cost_model.fit(X, y_cost)

# Train the Hospital Recommendation Classifier Model
hospital_model = RandomForestClassifier()
hospital_model.fit(X, y_hospital)


# =====================================================================
# 2. Static Asset Hosting
# =====================================================================
@app.route("/<path:filename>")
def serve_static(filename):
    """
    Serves static assets (CSS, JS, images) from the frontend folder.
    """
    return send_from_directory(FRONTEND_DIR, filename)


# =====================================================================
# 3. Autocomplete Search Suggestions API
# =====================================================================
@app.route("/api/suggestions", methods=["GET"])
def api_suggestions():
    """
    Returns unique lists of diseases and cities in alphabetical order
    to populate the frontend dropdown list search suggestions.
    """
    try:
        diseases = sorted(list(le_disease.classes_))
        cities = sorted(list(le_city.classes_))
        return jsonify({
            "diseases": diseases,
            "cities": cities
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =====================================================================
# 4. Estimation and Comparison API (/predict)
# =====================================================================
@app.route("/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint. Takes a JSON request containing:
      - disease: name of the illness/procedure
      - city: name of the city
    
    Processes the request, handles typos using difflib matching, 
    and predicts costs for both Private and Government hospital types.
    """
    user_data = request.json

    # Helper function to fix spelling mistakes using fuzzy matching
    def get_match(user_input, valid_classes):
        """
        Compares user input to a list of known valid categories (classes)
        and returns the closest matching class name.
        """
        capitalized_input = user_input.strip().title()
        
        # 1. Check for exact match first
        if capitalized_input in valid_classes:
            return capitalized_input

        # 2. Check for case-insensitive exact match
        lower_input = user_input.strip().lower()
        for c in valid_classes:
            if c.lower() == lower_input:
                return c

        # 3. Search for closest match using difflib fuzzy matching
        matches = difflib.get_close_matches(capitalized_input, valid_classes, n=1, cutoff=0.4)
        if matches:
            return matches[0]

        # 4. Final fallback: loop and match by SequenceMatcher similarity ratio
        best_match = None
        best_ratio = -1.0
        for c in valid_classes:
            ratio = difflib.SequenceMatcher(None, lower_input, c.lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = c

        if best_ratio > 0.2:
            return best_match

        return capitalized_input # Return original if no matches are close enough

    try:
        # Resolve the closest valid disease name and encode it
        matched_disease = get_match(user_data["disease"], le_disease.classes_)
        disease = le_disease.transform([matched_disease])[0]
        
        # Resolve the closest valid city name and encode it
        matched_city = get_match(user_data["city"], le_city.classes_)
        city = le_city.transform([matched_city])[0]
    except ValueError:
        # Raised if label encoder doesn't recognize the resulting match
        return jsonify({"error": "Unrecognized category. Please check your spelling and try again."}), 400

    results = {}

    # Compute predictions for all available hospital types in label encoder (e.g. 'Govt', 'Private')
    for h_type in le_type.classes_:
        h_type_encoded = le_type.transform([h_type])[0]
        input_data = [[disease, city, h_type_encoded]]
        
        # Predict estimated cost using the Random Forest Regressor
        predicted_cost = int(cost_model.predict(input_data)[0])
        
        # Filter raw dataset rows matching this city and hospital type to restrict recommendations to local facilities
        city_type_data = data[(data["City"] == city) & (data["Hospital_Type"] == h_type_encoded)]
        
        if not city_type_data.empty:
            valid_hospitals = city_type_data["Hospital_Name"].unique()
            # Get probability scores for all hospital classifications
            probs = hospital_model.predict_proba(input_data)[0]
            
            best_hospital = None
            best_prob = -1
            classes_list = list(hospital_model.classes_)
            
            # Find the hospital from the valid list with the highest model probability
            for h in valid_hospitals:
                idx = classes_list.index(h)
                if probs[idx] > best_prob:
                    best_prob = probs[idx]
                    best_hospital = h
                    
            predicted_hospital_encoded = best_hospital
        else:
            # Fallback classifier prediction if no hospital matches the exact city/type pair
            predicted_hospital_encoded = int(hospital_model.predict(input_data)[0])

        # Decode numerical hospital code back to its original name string
        predicted_hospital = le_hospital.inverse_transform([predicted_hospital_encoded])[0]
        
        # Extract demographic information (address, contact) from database rows matching the recommended hospital
        matched_rows = raw_data[(raw_data["Hospital_Name"] == predicted_hospital) & (raw_data["City"] == matched_city)]
        if matched_rows.empty:
            # Fallback query if city names do not match perfectly
            matched_rows = raw_data[raw_data["Hospital_Name"] == predicted_hospital]
            
        if not matched_rows.empty:
            address = matched_rows.iloc[0]["Address"]
            contact = matched_rows.iloc[0]["Contact"]
            h_type_str = matched_rows.iloc[0]["Hospital_Type"]
        else:
            address = "N/A"
            contact = "N/A"
            h_type_str = h_type

        # =====================================================================
        # Cost Range Calculations (Range Transparency)
        # =====================================================================
        # Attempts to calculate real min/max boundaries from historical dataset records.
        # If no records exist, we fall back to statistical scale bounds.
        history_match = data[
            (data["Disease"] == disease) & 
            (data["City"] == city) & 
            (data["Hospital_Type"] == h_type_encoded)
        ]
        
        if not history_match.empty:
            min_cost = int(history_match["Cost"].min())
            max_cost = int(history_match["Cost"].max())
            if min_cost == max_cost:
                # Add default variance if only a single historical data point exists
                min_cost = int(predicted_cost * 0.85)
                max_cost = int(predicted_cost * 1.15)
        else:
            # Match disease and city across all hospital types
            disease_city_match = data[
                (data["Disease"] == disease) & 
                (data["City"] == city)
            ]
            if not disease_city_match.empty:
                min_cost = int(disease_city_match["Cost"].min())
                max_cost = int(disease_city_match["Cost"].max())
                if min_cost == max_cost or h_type.lower() == "govt":
                    if h_type.lower() == "govt":
                        # Government facilities generally have a tighter budget range
                        min_cost = int(predicted_cost * 0.8)
                        max_cost = int(predicted_cost * 1.1)
                    else:
                        min_cost = int(predicted_cost * 0.9)
                        max_cost = int(predicted_cost * 1.25)
            else:
                # Universal fallback scale bounds based on hospital class
                if h_type.lower() == "govt":
                    min_cost = int(predicted_cost * 0.8)
                    max_cost = int(predicted_cost * 1.1)
                else:
                    min_cost = int(predicted_cost * 0.9)
                    max_cost = int(predicted_cost * 1.25)

        # Sanity validation bounds checks
        min_cost = min(min_cost, predicted_cost)
        max_cost = max(max_cost, predicted_cost)
        min_cost = max(min_cost, 1) # Prevent negative or zero prices
        max_cost = max(max_cost, min_cost + 1)
            
        # Add result object categorized by hospital type (e.g. results['govt'], results['private'])
        results[h_type.lower()] = {
            "hospital_name": predicted_hospital,
            "predicted_cost": predicted_cost,
            "min_cost": min_cost,
            "max_cost": max_cost,
            "hospital_type": h_type_str,
            "address": address,
            "contact": contact
        }

    # Return predictions JSON to UI client
    return jsonify({
        "city": matched_city,
        "disease": matched_disease,
        "original_city": user_data["city"],
        "original_disease": user_data["disease"],
        "predictions": results
    })


# =====================================================================
# 5. DB Connectivity, Session Management, and User Authentication APIs
# =====================================================================

app.secret_key = 'hospital_predictor_secret_key'

@app.route('/')
def auth_page():
    """
    Serves the landing login screen.
    """
    return render_template('auth.html')

@app.route('/index.html')
def index_page():
    """
    Serves the estimator dashboard layout. Blocks access if user is not authenticated.
    """
    if 'user_email' in session:
        return render_template('index.html')
    return "<h1>Access Denied. Please <a href='/'>Login</a> first.</h1>"

@app.route('/logout')
def logout():
    """
    Clears user session variables and redirects user back to login.
    """
    session.clear()
    return redirect('/')

@app.route('/api/signup', methods=['POST'])
def api_signup():
    """
    User Account Signup API. Inserts registration credentials to MySQL database.
    """
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')

    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (name, email, password) VALUES (%s, %s, %s)", (name, email, password))
        conn.commit()
        session['user_email'] = email
        return jsonify({"success": True, "message": "Account created successfully!"})
    except Exception as err:
        return jsonify({"success": False, "message": f"Database error during signup: {str(err)}"}), 400
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

@app.route('/api/login', methods=['POST'])
def api_login():
    """
    User Account Login API. Queries database and sets up application session cookies on success.
    """
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    conn = None
    cursor = None
    try:
        import pymysql.cursors
        conn = get_db_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        cursor.execute("SELECT * FROM users WHERE email = %s AND password = %s", (email, password))
        user = cursor.fetchone()
        if user:
            session['user_email'] = user['email']
            return jsonify({"success": True, "message": "Login successful!"})
        else:
            return jsonify({"success": False, "message": "Invalid email or password."}), 401
    except Exception as err:
        return jsonify({"success": False, "message": f"Database error during login: {str(err)}"}), 400
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


if __name__ == "__main__":
    # Start web server
    app.run(debug=True)
