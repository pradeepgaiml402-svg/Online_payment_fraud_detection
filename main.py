import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import time

# Load dataset
data = pd.read_csv('online_payments_fraud.csv')

# Print the first few rows to understand data structure
print("First few rows of the dataset:\n", data.head())

# Assuming all non-numeric columns are not crucial, we drop them
non_numeric_columns = data.select_dtypes(include=['object']).columns
print("Non-numeric columns to drop:", non_numeric_columns)
data = data.drop(non_numeric_columns, axis=1)

# Define features and target
X = data.drop('isFraud', axis=1)
y = data['isFraud']

# Check for missing values and fill them
print("Checking for missing values...")
X = X.fillna(X.mean())
print("Missing values handled.")

# Use a smaller sample of the data for testing
data_sample = data.sample(frac=0.1, random_state=42)
X_sample = data_sample.drop('isFraud', axis=1)
y_sample = data_sample['isFraud']

# Split data into training and test sets
print("Splitting the data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)
print("Data split complete.")

# Scale the features
print("Scaling the features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Feature scaling complete.")

# Train the model with timing
start_time = time.time()
print("Training the model...")
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
print("Model training complete. Time taken: {:.2f} seconds".format(time.time() - start_time))

# Make predictions
print("Making predictions...")
y_pred = model.predict(X_test)
print("Predictions complete.")

# Evaluate the model
print("Evaluating the model...")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the model and scaler
print("Saving the model and scaler...")
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model and scaler saved.")
