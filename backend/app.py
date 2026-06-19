# database
from database import get_db_connection
# this is adding some new requirment
from flask import Flask, render_template, request, jsonify, send_from_directory, session, redirect
from flask_cors import CORS
# apelling
import difflib
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, 'frontend')

app = Flask(
    __name__,
    template_folder=FRONTEND_DIR,  # Finds auth.html and index.html
    static_folder=FRONTEND_DIR,    # Finds auth.css, style.css, logo.png, etc.
    static_url_path=''
)
CORS(app)

# Load dataset
raw_data = pd.read_csv("../dataset/hospital_data.csv")
data = raw_data.copy()

# Label Encoders
le_disease = LabelEncoder()
le_city = LabelEncoder()
le_type = LabelEncoder()
le_hospital = LabelEncoder()

# Encode columns
data["Disease"] = le_disease.fit_transform(data["Disease"])
data["City"] = le_city.fit_transform(data["City"])
data["Hospital_Type"] = le_type.fit_transform(data["Hospital_Type"])
data["Hospital_Name"] = le_hospital.fit_transform(data["Hospital_Name"])

# Features and targets
X = data[["Disease", "City", "Hospital_Type"]]
y_cost = data["Cost"]
y_hospital = data["Hospital_Name"]

# Train models
cost_model = RandomForestRegressor()
cost_model.fit(X, y_cost)

hospital_model = RandomForestClassifier()
hospital_model.fit(X, y_hospital)


@app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory(FRONTEND_DIR, filename)

@app.route("/api/suggestions", methods=["GET"])
def api_suggestions():
    try:
        diseases = sorted(list(le_disease.classes_))
        cities = sorted(list(le_city.classes_))
        return jsonify({
            "diseases": diseases,
            "cities": cities
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    user_data = request.json


    # Helper function to fix spelling mistakes using fuzzy matching
    def get_match(user_input, valid_classes):
        capitalized_input = user_input.strip().title()
        # 1. Exact match check
        if capitalized_input in valid_classes:
            return capitalized_input

        # 2. Case-insensitive exact match
        lower_input = user_input.strip().lower()
        for c in valid_classes:
            if c.lower() == lower_input:
                return c

        # 3. Find the closest match to handle typos using a lower cutoff
        matches = difflib.get_close_matches(capitalized_input, valid_classes, n=1, cutoff=0.4)
        if matches:
            return matches[0]

        # 4. Fallback: find the single best match by SequenceMatcher ratio
        best_match = None
        best_ratio = -1.0
        for c in valid_classes:
            ratio = difflib.SequenceMatcher(None, lower_input, c.lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = c

        if best_ratio > 0.2:
            return best_match

        return capitalized_input # Fallback if no close match is found

    try:
        # Match user input to the closest valid category and encode it
        matched_disease = get_match(user_data["disease"], le_disease.classes_)
        disease = le_disease.transform([matched_disease])[0]
        
        matched_city = get_match(user_data["city"], le_city.classes_)
        city = le_city.transform([matched_city])[0]
    except ValueError:
        return jsonify({"error": "Unrecognized category. Please check your spelling and try again."}), 400

    results = {}

    # Predict for all possible hospital types
    for h_type in le_type.classes_:
        h_type_encoded = le_type.transform([h_type])[0]
        input_data = [[disease, city, h_type_encoded]]
        
        predicted_cost = int(cost_model.predict(input_data)[0])
        
        # Filter data for the exact city and hospital type
        city_type_data = data[(data["City"] == city) & (data["Hospital_Type"] == h_type_encoded)]
        
        if not city_type_data.empty:
            valid_hospitals = city_type_data["Hospital_Name"].unique()
            probs = hospital_model.predict_proba(input_data)[0]
            
            best_hospital = None
            best_prob = -1
            classes_list = list(hospital_model.classes_)
            
            for h in valid_hospitals:
                idx = classes_list.index(h)
                if probs[idx] > best_prob:
                    best_prob = probs[idx]
                    best_hospital = h
                    
            predicted_hospital_encoded = best_hospital
        else:
            # Fallback if no hospital of that type exists in the city
            predicted_hospital_encoded = int(hospital_model.predict(input_data)[0])

        predicted_hospital = le_hospital.inverse_transform([predicted_hospital_encoded])[0]
        
        # Look up additional details from raw_data matching the predicted hospital name and city
        matched_rows = raw_data[(raw_data["Hospital_Name"] == predicted_hospital) & (raw_data["City"] == matched_city)]
        if matched_rows.empty:
            # Fallback to name only if city mismatch (should not happen normally)
            matched_rows = raw_data[raw_data["Hospital_Name"] == predicted_hospital]
            
        if not matched_rows.empty:
            address = matched_rows.iloc[0]["Address"]
            contact = matched_rows.iloc[0]["Contact"]
            h_type_str = matched_rows.iloc[0]["Hospital_Type"]
        else:
            address = "N/A"
            contact = "N/A"
            h_type_str = h_type

        # Range calculations
        history_match = data[
            (data["Disease"] == disease) & 
            (data["City"] == city) & 
            (data["Hospital_Type"] == h_type_encoded)
        ]
        
        if not history_match.empty:
            min_cost = int(history_match["Cost"].min())
            max_cost = int(history_match["Cost"].max())
            if min_cost == max_cost:
                min_cost = int(predicted_cost * 0.85)
                max_cost = int(predicted_cost * 1.15)
        else:
            # Check for disease and city across all hospital types
            disease_city_match = data[
                (data["Disease"] == disease) & 
                (data["City"] == city)
            ]
            if not disease_city_match.empty:
                min_cost = int(disease_city_match["Cost"].min())
                max_cost = int(disease_city_match["Cost"].max())
                if min_cost == max_cost or h_type.lower() == "govt":
                    if h_type.lower() == "govt":
                        min_cost = int(predicted_cost * 0.8)
                        max_cost = int(predicted_cost * 1.1)
                    else:
                        min_cost = int(predicted_cost * 0.9)
                        max_cost = int(predicted_cost * 1.25)
            else:
                if h_type.lower() == "govt":
                    min_cost = int(predicted_cost * 0.8)
                    max_cost = int(predicted_cost * 1.1)
                else:
                    min_cost = int(predicted_cost * 0.9)
                    max_cost = int(predicted_cost * 1.25)

        # Bounds and sanity checks
        min_cost = min(min_cost, predicted_cost)
        max_cost = max(max_cost, predicted_cost)
        min_cost = max(min_cost, 1)
        max_cost = max(max_cost, min_cost + 1)
            
        results[h_type.lower()] = {
            "hospital_name": predicted_hospital,
            "predicted_cost": predicted_cost,
            "min_cost": min_cost,
            "max_cost": max_cost,
            "hospital_type": h_type_str,
            "address": address,
            "contact": contact
        }

    return jsonify({
        "city": matched_city,
        "disease": matched_disease,
        "original_city": user_data["city"],
        "original_disease": user_data["disease"],
        "predictions": results
    })


# for database connection
app.secret_key = 'hospital_predictor_secret_key'

# 2. Serve the Auth Page
@app.route('/')
def auth_page():
    return render_template('auth.html')

# 3. Serve the Homepage (Dashboard) after successful login
@app.route('/index.html')
def index_page():
    if 'user_email' in session:
        return render_template('index.html')
    return "<h1>Access Denied. Please <a href='/'>Login</a> first.</h1>"

# Logout Route
@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

# 4. API Endpoint for Signup
@app.route('/api/signup', methods=['POST'])
def api_signup():
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

# 5. API Endpoint for Login
@app.route('/api/login', methods=['POST'])
def api_login():
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
    app.run(debug=True)