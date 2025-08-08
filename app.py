from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import pandas as pd
import joblib

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Replace with a secure key in production

# Configure Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# In-memory user store
users = {'user@example.com': {'password': 'password', 'name': 'Sample User'}}  # Replace with your user data

class User(UserMixin):
    def __init__(self, email):
        self.id = email

@login_manager.user_loader
def load_user(user_id):
    if user_id in users:
        return User(user_id)
    return None

# Load pre-trained model and preprocessor
pipeline, preprocessor = joblib.load('pipeline.pkl')  # Load the pipeline and preprocessor

# Define the expected feature names in the correct order
expected_features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud', 'type']

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if email in users and users[email]['password'] == password:
            user = User(email)
            login_user(user, remember='remember' in request.form)
            return redirect(url_for('index'))
        else:
            flash('Invalid email or password')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        if password != confirm_password:
            flash('Passwords do not match')
        elif email in users:
            flash('Email already registered')
        else:
            users[email] = {'password': password, 'name': name}
            user = User(email)
            login_user(user)
            return redirect(url_for('index'))
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        data = request.get_json(force=True)
        data_df = pd.DataFrame([data])
        
        # Ensure all expected features are present and in the correct order
        for feature in expected_features:
            if feature not in data_df.columns:
                data_df[feature] = 0  # Assign a default value (e.g., 0) for missing features
        
        # Reorder columns to match the model's expected input
        data_df = data_df[expected_features]
        
        # Preprocess the data
        data_processed = preprocessor.transform(data_df)
        
        # Predict using the pipeline
        prediction = pipeline.predict(data_processed)
        result = 'fraudulent' if prediction[0] == 1 else 'approved'
        
        return jsonify(result=result)
    except Exception as e:
        return jsonify(result='error', error=str(e)), 500

if __name__ == '__main__':
    app.run(debug=True)
