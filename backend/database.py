from flask import Flask, render_template, request, jsonify, session, send_from_directory
import pymysql
import pymysql.cursors
import os

app = Flask(__name__, template_folder='.') # Serves templates from the current directory
app.secret_key = 'hospital_predictor_secret_key'

# MySQL Connection Function
def get_db_connection():
    return pymysql.connect(
        host="localhost",
        user="root",             # Change to your MySQL username
        password="Abhijeet@123", # Change to your MySQL password
        database="hospital_db"
    )

# Route to serve the Auth CSS file if it's in the same directory
@app.route('/auth.css')
def serve_css():
    return send_from_directory(os.getcwd(), 'auth.css')

# 1. Main Route - Serves your login/signup page
@app.route('/')
def auth_page():
    return render_template('auth.html') # Ensure your HTML file is named auth.html

# 2. Homepage Route - Where users go after logging in
@app.route('/index.html')
def index_page():
    if 'user_email' in session:
        return f"<h1>Welcome to the Hospital Cost Predictor Dashboard, {session['user_name']}!</h1><br><a href='/logout'>Logout</a>"
    return "<h1>Access Denied. Please <a href='/'>Login</a> first.</h1>"

# 3. Handle Signup Data
@app.route('/api/signup', methods=['POST'])
def api_signup():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    password = data.get('password') # Note: In production, hash this using werkzeug.security!

    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("INSERT INTO users (name, email, password) VALUES (%s, %s, %s)", (name, email, password))
        conn.commit()
        return jsonify({"success": True, "message": "Account created successfully!"})
    except pymysql.MySQLError as err:
        # Catch duplicate email error
        if err.args[0] == 1062:
            return jsonify({"success": False, "message": "Email already registered."}), 400
        return jsonify({"success": False, "message": str(err)}), 500
    finally:
        cursor.close()
        conn.close()

# 4. Handle Login Data
@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    conn = get_db_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    
    cursor.execute("SELECT * FROM users WHERE email = %s AND password = %s", (email, password))
    user = cursor.fetchone()
    
    cursor.close()
    conn.close()

    if user:
        # Store user details in session
        session['user_email'] = user['email']
        session['user_name'] = user['name']
        return jsonify({"success": True, "message": "Login successful!"})
    else:
        return jsonify({"success": False, "message": "Invalid email or password."}), 401

# 5. Logout
@app.route('/logout')
def logout():
    session.clear()
    return f"Logged out. <a href='/'>Go back to Login</a>"

if __name__ == '__main__':
    app.run(debug=True)
    
# datase