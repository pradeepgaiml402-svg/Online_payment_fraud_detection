import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
data = pd.read_csv('online_payments_fraud.csv')

# Check class distribution
print("Class distribution before balancing:")
print(data['isFraud'].value_counts(normalize=True))

# Feature selection
features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'type']
X = data[features]
y = data['isFraud']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("Train and test data split complete.")

# Preprocessing: Encoding 'type' feature before SMOTE
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['type'])
    ])

X_train_transformed = preprocessor.fit_transform(X_train)

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_transformed, y_train)

# Define the classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000, C=0.1, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Train and evaluate each classifier
for name, classifier in classifiers.items():
    print(f"\nTraining and evaluating {name}...")
    # Create a pipeline that preprocesses data and then fits the model
    pipeline = Pipeline(steps=[
        ('classifier', classifier)
    ])
    
    # Fit the pipeline on the training data
    pipeline.fit(X_train_balanced, y_train_balanced)
    print(f"{name} model training complete.")
    
    # Cross-validation
    cv_scores = cross_val_score(pipeline, X_train_balanced, y_train_balanced, cv=5)
    print(f'{name} cross-validation scores: {cv_scores}')
    print(f'{name} mean cross-validation score: {cv_scores.mean():.2f}')
    
    # Evaluate the model on the test set
    X_test_transformed = preprocessor.transform(X_test)
    y_pred = pipeline.predict(X_test_transformed)
    test_score = pipeline.score(X_test_transformed, y_test)
    print(f'{name} model accuracy: {test_score:.2f}')
    
    # Classification metrics
    print(f"{name} classification report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    print(f"{name} confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save the pipeline and preprocessor
    joblib.dump((pipeline, preprocessor), f'pipeline_{name.replace(" ", "_").lower()}.pkl')
    print(f"{name} pipeline saved as pipeline_{name.replace(' ', '_').lower()}.pkl")
