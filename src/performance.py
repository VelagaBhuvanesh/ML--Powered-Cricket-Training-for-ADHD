"""
Model Evaluation & Prediction Demo
Loads the trained XGBoost model and evaluates it on test data.
Shows performance metrics and sample predictions.
"""

import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from config_mysql import MYSQL_CONFIG

def load_data():
    """Load data from MySQL"""
    print("📊 Loading data from MySQL...")
    engine = create_engine(
        f"mysql+mysqlconnector://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}@"
        f"{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}"
    )
    df = pd.read_sql_query("SELECT * FROM windows", engine)
    print(f"✓ Loaded {len(df)} windows")
    return df

def prepare_features(df):
    """Prepare features and labels"""
    # Drop non-feature columns
    drop_cols = ['window_id', 'participant_id', 'label', 'window_index', 'created_at', 'adhd_prob']
    existing_drop = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=existing_drop, errors='ignore')
    y = df['label']
    
    # Keep only numeric columns
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
    X = X[numeric_cols].fillna(0)
    
    print(f"✓ Features: {X.shape[1]} numeric features")
    return X, y

def load_model():
    """Load trained model and scaler"""
    print("🔄 Loading model and scaler...")
    model = joblib.load('models/xgboost_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    print("✓ Model and scaler loaded")
    return model, scaler

def evaluate_model(model, scaler, X, y):
    """Evaluate model on test set"""
    print("\n📊 Evaluating model...")
    
    # Split data (same as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale test data
    X_test_scaled = scaler.transform(X_test)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print("\n" + "="*60)
    print("📈 MODEL PERFORMANCE SUMMARY")
    print("="*60)
    print(f"✅ Accuracy:  {accuracy:.4f}")
    print(f"✅ Precision: {precision:.4f}")
    print(f"✅ Recall:    {recall:.4f}")
    print(f"✅ F1-Score:  {f1:.4f}")
    print(f"✅ AUC-ROC:   {auc:.4f}")
    
    # Classification Report
    print("\n" + "="*60)
    print("📋 CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_test, y_pred, target_names=['Control', 'ADHD']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n" + "="*60)
    print("📊 CONFUSION MATRIX")
    print("="*60)
    print(f"                  Predicted")
    print(f"                  ADHD    Control")
    print(f"Actual ADHD      {cm[1,1]:>5}     {cm[1,0]:>5}")
    print(f"Actual Control   {cm[0,1]:>5}     {cm[0,0]:>5}")
    
    # Detailed breakdown
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  True Positives (correct ADHD):  {tp}")
    print(f"  True Negatives (correct Control): {tn}")
    print(f"  False Positives (false alarm):   {fp}")
    print(f"  False Negatives (missed ADHD):   {fn}")
    
    return y_test, y_pred, y_prob

def show_sample_predictions(model, scaler, X, y):
    """Show sample predictions with actual labels"""
    print("\n" + "="*60)
    print("🎯 SAMPLE PREDICTIONS")
    print("="*60)
    
    # Take 10 random samples
    np.random.seed(42)
    indices = np.random.choice(len(X), size=10, replace=False)
    
    X_sample = X.iloc[indices]
    y_sample = y.iloc[indices]
    
    X_scaled = scaler.transform(X_sample)
    probs = model.predict_proba(X_scaled)[:, 1]
    preds = model.predict(X_scaled)
    
    # Create DataFrame
    results = pd.DataFrame({
        'Actual': y_sample.values,
        'Predicted': preds,
        'Probability': probs
    })
    
    results['Correct'] = results['Actual'] == results['Predicted']
    
    print("\n  Actual  Predicted  Probability  Correct")
    print("  " + "-"*45)
    for i, row in results.iterrows():
        actual = 'ADHD' if row['Actual'] == 1 else 'Control'
        pred = 'ADHD' if row['Predicted'] == 1 else 'Control'
        prob = f"{row['Probability']:.3f}"
        correct = '✅' if row['Correct'] else '❌'
        print(f"  {actual:>7}  {pred:>9}   {prob:>10}      {correct}")
    
    return results

def plot_roc_curve(y_test, y_prob):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'XGBoost (AUC = {auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - ADHD Detection')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('models/roc_curve.png', dpi=150)
    print("\n✓ ROC curve saved to 'models/roc_curve.png'")
    plt.show()

def main():
    """Run evaluation"""
    print("\n" + "🚀"*30)
    print(" MODEL EVALUATION & PREDICTION DEMO")
    print("🚀"*30)
    
    # Load data
    df = load_data()
    X, y = prepare_features(df)
    
    # Load model
    model, scaler = load_model()
    
    # Evaluate
    y_test, y_pred, y_prob = evaluate_model(model, scaler, X, y)
    
    # Show sample predictions
    show_sample_predictions(model, scaler, X, y)
    
    # Plot ROC curve
    plot_roc_curve(y_test, y_prob)
    
    print("\n" + "✅"*30)
    print(" EVALUATION COMPLETE")
    print("✅"*30)

if __name__ == "__main__":
    main()