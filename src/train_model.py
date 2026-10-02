"""
Model training module for EcoGuardian AI.
Trains and evaluates a Random Forest classifier.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os


FEATURE_COLUMNS = [
    'temperature', 'humidity', 'wind_speed',
    'pm25', 'pm10',
    'Heat_Risk', 'Pollution_Risk', 'Climate_Severity'
]

TARGET_COLUMN = 'Recommendation'


def load_labeled_data(filepath):
    """Load labeled dataset."""
    return pd.read_csv(filepath)


def prepare_features(df):
    """Split data into features and target."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y


def train_random_forest(X_train, y_train, n_estimators=300, random_state=42):
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    return accuracy


def save_model(model, filepath):
    """Save trained model to disk."""
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'environmental_dataset.csv')
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pkl')

    df = load_labeled_data(data_path)
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    model = train_random_forest(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model, model_path)
