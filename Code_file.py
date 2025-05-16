import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ========== Config ==========
DATA_PATH = 'customer_booking.csv'
MODEL_PATH = 'models/ticket_booking_model.pkl'
ENCODERS_PATH = 'models/label_encoders.pkl'
REPORT_DIR = 'reports'

# ========== Create directories ==========
os.makedirs('models', exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ========== Load and Preprocess Data ==========
def load_and_clean_data(path):
    logging.info("Loading data...")
    data = pd.read_csv(path, encoding="ISO-8859-1")
    data = data.ffill()

    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    return data, label_encoders

# ========== Train Model ==========
def train_model(X_train, y_train):
    logging.info("Training RandomForest model...")
    model = RandomForestClassifier(random_state=42)
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)
    model = grid.best_estimator_

    model.fit(X_train, y_train)
    return model

# ========== Evaluate Model ==========
def evaluate_model(model, X_test, y_test):
    logging.info("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Accuracy: {accuracy:.4f}")
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    logging.info(f"\nClassification Report:\n{report}")
    logging.info(f"\nConfusion Matrix:\n{matrix}")

    with open(f'{REPORT_DIR}/classification_report.txt', 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(report)
        f.write(f"\nConfusion Matrix:\n{matrix}\n")

    return accuracy

# ========== Feature Importance ==========
def plot_feature_importance(model, features):
    logging.info("Plotting feature importances...")
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(f'{REPORT_DIR}/feature_importances.png')
    plt.close()
    logging.info("Feature importance plot saved.")
    return importance_df

# ========== Cross Validation ==========
def cross_validate(model, x, y):
    logging.info("Performing cross-validation...")
    scores = cross_val_score(model, x, y, cv=5)
    logging.info(f"CV Scores: {scores}")
    logging.info(f"Mean CV Score: {scores.mean():.4f}")
    return scores

# ========== Save Artifacts ==========
def save_model_and_encoders(model, encoders):
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    logging.info(f"Model saved to {MODEL_PATH}")
    logging.info(f"Label encoders saved to {ENCODERS_PATH}")

# ========== Main Pipeline ==========
def main():
    data, label_encoders = load_and_clean_data(DATA_PATH)
    x = data.drop('booking_complete', axis=1)
    y = data['booking_complete']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    cross_validate(model, x, y)
    plot_feature_importance(model, x.columns)
    save_model_and_encoders(model, label_encoders)
    logging.info("Pipeline completed successfully.")

if __name__ == '__main__':
    main()
