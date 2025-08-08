# Online Payment Fraud Detection using Machine Learning

Project Overview
This project focuses on detecting fraudulent online payment transactions using machine learning techniques. With over **200,000+ transaction records**, the goal was to build robust models capable of identifying fraud with high accuracy, even under significant **class imbalance**.

Key Contributions
- Trained and evaluated models using Random Forest and XGBoost, achieving up to 94% accuracy.
- Engineered 25+ features including:
  - Transaction time and amount
  - User behavior metrics (e.g., frequency, velocity)
  - Historical fraud indicators
- Applied cross-validation and hyperparameter tuning to optimize model performance.
- Addressed class imbalance using techniques like SMOTE and class weighting.

Dataset
- Source: Simulated or anonymized the transaction data
- Size: 200,000+ records
- Features: Time, amount, user ID, transaction type, location, device info, etc.
- Target: Binary label indicating fraud (1) or legitimate (0)

Models Used
| Model           | Accuracy | Precision | Recall | F1 Score |
|----------------|----------|-----------|--------|----------|
| Random Forest   | 92%      | 89%       | 90%    | 89.5%    |
| XGBoost         | 94%      | 91%       | 92%    | 91.5%    |

> Note: Performance metrics are based on stratified cross-validation and test set evaluation.

Technologies & Tools
- Languages: Python
- Libraries: scikit-learn, XGBoost, pandas, NumPy, matplotlib, seaborn
- Techniques: Feature engineering, class balancing, model tuning, evaluation metrics

Results & Insights
- XGBoost outperformed Random Forest in both precision and recall.
- Feature importance analysis revealed transaction time and user velocity as key indicators.
- Class imbalance mitigation significantly improved recall for minority class (fraud).
