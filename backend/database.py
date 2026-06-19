"""
Hospital Cost Estimator - Database & Basic Authentication Module
================================================================

This module handles the core database configurations and standard authentication operations 
(Login, Signup, and Session Verification) for the Flask backend application using PyMySQL.

Prerequisite Database Setup:
----------------------------
Make sure to create the database and user table in your MySQL server before running:

    CREATE DATABASE IF NOT EXISTS hospital_db;
    USE hospital_db;

    CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        email VARCHAR(255) UNIQUE NOT NULL,
        password VARCHAR(255) NOT NULL
    );
"""

from flask import Flask, render_template, request, jsonify, session, send_from_directory
import pymysql
import pymysql.cursors
import os

# Initialize Flask Application with templates served from the local directory
app = Flask(__name__, template_folder='.') 
# Secret key used for signing session cookies to prevent client-side tampering
app.secret_key = 'hospital_predictor_secret_key'

# ==========================================
# MySQL Database Connection configuration
# ==========================================
def get_db_connection():
    """
    Establishes and returns a new connection to the MySQL database.
    Note: Replace credentials below with your local MySQL setup details.
    """
    return pymysql.connect(
        host="localhost",
        user="root",             # Change to your MySQL username
        password="Abhijeet@123", # Change to your MySQL password
        database="hospital_db"
    )

# ==========================================
# Static Resource Routing
# ==========================================
@app.route('/auth.css')
def serve_css():
    """
    Serves the CSS stylesheet directly from the working directory.
    Normally Flask serves styles from a 'static' directory, but this acts as a helper route.
    """
    return send_from_directory(os.getcwd(), 'auth.css')

# ==========================================
# Navigation Routes
# ==========================================
@app.route('/')
def auth_page():
    """
    Serves the main landing page, which houses the authentication (login/signup) interface.
    """
    return render_template('auth.html')

@app.route('/index.html')
def index_page():
    """
    Homepage/dashboard route. Accessible only to authenticated users who have
    their email stored in the active Flask session.
    """
    if 'user_email' in session:
        return f"<h1>Welcome to the Hospital Cost Predictor Dashboard, {session['user_name']}!</h1><br><a href='/logout'>Logout</a>"
    # Block access if session does not exist
    return "<h1>Access Denied. Please <a href='/'>Login</a> first.</h1>"

# ==========================================
# API Endpoint Routes
# ==========================================
@app.route('/api/signup', methods=['POST'])
def api_signup():
    """
    Handles user signup requests. Receives JSON payload with name, email, and password.
    Inserts data into the 'users' table and handles duplicate email errors.
    """
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    password = data.get('password') # Note: In production, hash this using a library like bcrypt or werkzeug.security!

    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Perform user registration insert
        cursor.execute("INSERT INTO users (name, email, password) VALUES (%s, %s, %s)", (name, email, password))
        conn.commit() # Save changes to the database
        return jsonify({"success": True, "message": "Account created successfully!"})
    except pymysql.MySQLError as err:
        # Check for Duplicate Entry error code (MySQL Error 1062)
        if err.args[0] == 1062:
            return jsonify({"success": False, "message": "Email already registered."}), 400
        return jsonify({"success": False, "message": str(err)}), 500
    finally:
        cursor.close()
        conn.close()

@app.route('/api/login', methods=['POST'])
def api_login():
    """
    Handles user login verification. Validates user credentials from the database.
    If valid, sets user session variables to track login state.
    """
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    conn = get_db_connection()
    # DictCursor allows accessing query output columns as dictionary keys (e.g. user['email'])
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    
    # Query database to find a matching email and password pair
    cursor.execute("SELECT * FROM users WHERE email = %s AND password = %s", (email, password))
    user = cursor.fetchone()
    
    cursor.close()
    conn.close()

    if user:
        # Store user details in session cookie dictionary to keep track of logged-in state
        session['user_email'] = user['email']
        session['user_name'] = user['name']
        return jsonify({"success": True, "message": "Login successful!"})
    else:
        # Authentication failure response
        return jsonify({"success": False, "message": "Invalid email or password."}), 401

@app.route('/logout')
def logout():
    """
    Clears all active session storage variables, logging out the user,
    and returns a link redirection back to the login screen.
    """
    session.clear()
    return f"Logged out. <a href='/'>Go back to Login</a>"

if __name__ == '__main__':
    # Run the debug server locally on default port 5000
    app.run(debug=True)

    
