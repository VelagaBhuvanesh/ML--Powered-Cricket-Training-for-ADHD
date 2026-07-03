import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mysql.connector
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_both_models():
    print("="*70)
    print("🧪 FINAL TEST: BOTH MODELS (NO ENSEMBLE)")
    print("   Model 1: IMU (XGBoost) | Model 2: Clinical (Random Forest)")
    print("="*70)
    
    # ============================================================
    # 1. LOAD MODELS
    # ============================================================
    print("\n📂 Loading models...")
    try:
        model1 = joblib.load('models/xgboost_model.pkl')
        scaler1 = joblib.load('models/scaler.pkl')
        print("✓ Model 1 (IMU) loaded")
    except Exception as e:
        print(f"❌ Model 1 not found: {e}")
        return
    
    try:
        model2 = joblib.load('models/model2_final.pkl')
        print("✓ Model 2 (Clinical) loaded")
    except Exception as e:
        print(f"❌ Model 2 not found: {e}")
        return
    
    # Load features for Model 2
    try:
        with open('models/model2_features.txt', 'r') as f:
            clinical_features = f.read().split(',')
    except:
        clinical_features = ['gender', 'age', 'handedness', 'adhd_measure', 'adhd_index',
                            'inattentive', 'hyper_impulsive', 'verbal_iq', 
                            'performance_iq', 'full4_iq']
    print(f"✓ Clinical features: {clinical_features}")
    
    # ============================================================
    # 2. LOAD TEST CSV (for Model 2)
    # ============================================================
    test_paths = [
        '../data/raw/adhd200_test.csv',
        'data/raw/adhd200_test.csv',
        'adhd200_test.csv'
    ]
    test_path = None
    for path in test_paths:
        if os.path.exists(path):
            test_path = path
            print(f"\n✅ Found test CSV at: {path}")
            break
    
    if test_path is None:
        print("❌ Test CSV not found!")
        return
    
    df_test = pd.read_csv(test_path)
    df_test.columns = df_test.columns.str.lower()
    if 'dx' in df_test.columns:
        df_test = df_test.rename(columns={'dx': 'label'})
    
    print(f"\n📥 Loaded test CSV: {len(df_test)} rows")
    print(f"  Columns: {df_test.columns.tolist()}")
    
    # ============================================================
    # 3. TEST MODEL 2 (Clinical) on CSV
    # ============================================================
    print("\n" + "="*70)
    print("📊 MODEL 2 (CLINICAL) - Random Forest")
    print("="*70)
    
    # Ensure all clinical features exist
    for col in clinical_features:
        if col not in df_test.columns:
            match = next((c for c in df_test.columns if c.lower() == col), None)
            if match:
                df_test = df_test.rename(columns={match: col})
            else:
                print(f"⚠️ Missing '{col}', adding zeros")
                df_test[col] = 0
    
    X2 = df_test[clinical_features]
    y2 = df_test['label']
    
    print(f"\n📊 Test data for Model 2:")
    print(f"  ADHD (1): {sum(y2==1)}")
    print(f"  Control (0): {sum(y2==0)}")
    print(f"  Total: {len(y2)}")
    
    y2_pred = model2.predict(X2)
    y2_prob = model2.predict_proba(X2)[:, 1]
    
    acc2 = accuracy_score(y2, y2_pred)
    print(f"\n✅ Model 2 Accuracy on Test CSV: {acc2:.4f}")
    
    print("\n📋 Model 2 Classification Report:")
    print(classification_report(y2, y2_pred, target_names=['Control', 'ADHD']))
    
    cm2 = confusion_matrix(y2, y2_pred)
    print("\n📊 Model 2 Confusion Matrix:")
    print(f"  True Negatives:  {cm2[0,0]}")
    print(f"  False Positives: {cm2[0,1]}")
    print(f"  False Negatives: {cm2[1,0]}")
    print(f"  True Positives:  {cm2[1,1]}")
    
    # ============================================================
    # 4. TEST MODEL 1 (IMU) on Windows Table
    # ============================================================
    print("\n" + "="*70)
    print("📊 MODEL 1 (IMU) - XGBoost")
    print("   Testing on Windows table from MySQL")
    print("="*70)
    
    try:
        from config_mysql import MYSQL_CONFIG
        connection = mysql.connector.connect(**MYSQL_CONFIG)
        df_windows = pd.read_sql_query("SELECT * FROM windows", connection)
        connection.close()
        
        print(f"\n📥 Loaded: {len(df_windows)} rows from 'windows' table")
        
        # Prepare features for Model 1
        drop_cols = ['window_id', 'participant_id', 'label', 'window_index', 'created_at', 'adhd_prob']
        existing_drop = [c for c in drop_cols if c in df_windows.columns]
        X1 = df_windows.drop(columns=existing_drop, errors='ignore')
        X1 = X1.select_dtypes(include=['float64', 'int64']).fillna(0)
        y1 = df_windows['label']
        
        X1_scaled = scaler1.transform(X1)
        y1_pred = model1.predict(X1_scaled)
        y1_prob = model1.predict_proba(X1_scaled)[:, 1]
        
        acc1 = accuracy_score(y1, y1_pred)
        print(f"\n✅ Model 1 Accuracy on Windows Table: {acc1:.4f}")
        
        print("\n📋 Model 1 Classification Report:")
        print(classification_report(y1, y1_pred, target_names=['Control', 'ADHD']))
        
        cm1 = confusion_matrix(y1, y1_pred)
        print("\n📊 Model 1 Confusion Matrix:")
        print(f"  True Negatives:  {cm1[0,0]}")
        print(f"  False Positives: {cm1[0,1]}")
        print(f"  False Negatives: {cm1[1,0]}")
        print(f"  True Positives:  {cm1[1,1]}")
        
    except Exception as e:
        print(f"⚠️ Could not test Model 1: {e}")
        acc1 = None
    
    # ============================================================
    # 5. COMBINED RESULTS (Only Model 2 for CSV)
    # ============================================================
    print("\n" + "="*70)
    print("🎯 FINAL VERDICT FOR EACH SAMPLE (Model 2 - Clinical)")
    print("="*70)
    print(f"{'Sample':6} {'Actual':8} {'Predicted':10} {'Prob':8} {'Correct'}")
    print("-"*50)
    
    # Also count how many predicted ADHD vs Control
    pred_adhd_count = sum(y2_pred)
    pred_control_count = len(y2_pred) - pred_adhd_count
    actual_adhd_count = sum(y2)
    actual_control_count = len(y2) - actual_adhd_count
    
    for idx in range(len(df_test)):
        row = df_test.iloc[idx]
        actual = "ADHD" if row['label'] == 1 else "Control"
        pred = "ADHD" if y2_pred[idx] == 1 else "Control"
        prob = y2_prob[idx]
        correct = "✅" if y2_pred[idx] == row['label'] else "❌"
        print(f"{idx+1:6} {actual:8} {pred:10} {prob:.2%}   {correct}")
    
    print("\n" + "="*70)
    print("📊 COMBINED RESULT SUMMARY")
    print("="*70)
    print(f"Model 1 (IMU / XGBoost) Accuracy  : {acc1:.4f if acc1 else 'N/A'}")
    print(f"Model 2 (Clinical / RF) Accuracy  : {acc2:.4f}")
    print(f"\n📌 For the test CSV (29 samples):")
    print(f"   Actual ADHD samples  : {actual_adhd_count}")
    print(f"   Actual Control samples: {actual_control_count}")
    print(f"   Predicted ADHD        : {pred_adhd_count}")
    print(f"   Predicted Control     : {pred_control_count}")
    print(f"   Correct predictions   : {sum(y2_pred == y2)}")
    print(f"   Incorrect predictions : {sum(y2_pred != y2)}")
    
    # Final combined interpretation
    print("\n" + "="*70)
    print("💡 FINAL INTERPRETATION (for this CSV)")
    print("="*70)
    if pred_adhd_count > pred_control_count:
        print("   ⚠️ Majority of samples predicted as ADHD.")
    else:
        print("   ✅ Majority of samples predicted as Control.")
    
    print("\n   ⚠️ This is a screening tool. Consult a healthcare professional for diagnosis.")
    print("   ✅ Model 2 is recommended when you have questionnaire data.")
    print("   ✅ Model 1 is recommended when you have IMU data from the sleeve.")

if __name__ == "__main__":
    test_both_models()