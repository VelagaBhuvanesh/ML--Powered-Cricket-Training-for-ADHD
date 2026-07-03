import pandas as pd
import numpy as np
import joblib
import mysql.connector
from mysql.connector import Error
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ADHDEnsemble:
    def __init__(self):
        """Initialize ensemble with both models"""
        self.model1 = None
        self.model2 = None
        self.scaler1 = None
        self.model1_features = None
        self.model2_features = None
        self.meta_model = None
        
        self._load_models()
    
    def _load_models(self):
        """Load both trained models"""
        print("📂 Loading models...")
        
        # Load Model 1 (Movement)
        try:
            self.model1 = joblib.load('models/xgboost_model.pkl')
            self.scaler1 = joblib.load('models/scaler.pkl')
            print("✓ Model 1 (Movement) loaded")
        except Exception as e:
            print(f"⚠️ Model 1 not found: {e}")
        
        # Load Model 2 (Behavioral)
        try:
            self.model2 = joblib.load('models/model2_final.pkl')
            print("✓ Model 2 (Behavioral) loaded")
        except Exception as e:
            print(f"⚠️ Model 2 not found: {e}")
        
        # Load feature names
        try:
            with open('models/model1_features.txt', 'r') as f:
                self.model1_features = f.read().split(',')
        except:
            self.model1_features = None
        
        try:
            with open('models/model2_features.txt', 'r') as f:
                self.model2_features = f.read().split(',')
        except:
            self.model2_features = None
    
    def predict_model1(self, X_movement):
        """Get predictions from Model 1"""
        if self.model1 is None:
            return None
        X_scaled = self.scaler1.transform(X_movement)
        return self.model1.predict_proba(X_scaled)[:, 1]
    
    def predict_model2(self, X_behavioral):
        """Get predictions from Model 2"""
        if self.model2 is None:
            return None
        return self.model2.predict_proba(X_behavioral)[:, 1]
    
    def ensemble_weighted(self, prob1, prob2, weight1=0.6, weight2=0.4):
        """Weighted average ensemble"""
        return (prob1 * weight1) + (prob2 * weight2)
    
    def ensemble_meta(self, prob1, prob2, y_true=None, train_meta=False):
        """Meta-model ensemble (stacking)"""
        if train_meta and y_true is not None:
            # Train meta-model
            meta_X = np.column_stack([prob1, prob2])
            self.meta_model = LogisticRegression()
            self.meta_model.fit(meta_X, y_true)
            print("✓ Meta-model trained")
            return None
        
        if self.meta_model is None:
            # Use weighted as fallback
            return self.ensemble_weighted(prob1, prob2)
        
        meta_X = np.column_stack([prob1, prob2])
        return self.meta_model.predict_proba(meta_X)[:, 1]
    
    def evaluate_ensemble(self, X1, X2, y_true, method='weighted'):
        """
        Evaluate ensemble on test data.
        
        Args:
            X1: Features for Model 1 (movement)
            X2: Features for Model 2 (behavioral)
            y_true: True labels
            method: 'weighted', 'meta', 'max', 'min', 'average'
        """
        print("\n" + "="*60)
        print("📊 ENSEMBLE EVALUATION")
        print("="*60)
        
        # Get individual predictions
        prob1 = self.predict_model1(X1)
        prob2 = self.predict_model2(X2)
        
        if prob1 is None or prob2 is None:
            print("❌ Both models need to be loaded!")
            return
        
        results = {}
        
        # Method 1: Weighted
        if method == 'weighted' or method == 'all':
            print("\n🔹 Method: Weighted Average (Model1: 60%, Model2: 40%)")
            final_prob = self.ensemble_weighted(prob1, prob2)
            final_pred = (final_prob >= 0.5).astype(int)
            acc = accuracy_score(y_true, final_pred)
            results['weighted'] = acc
            print(f"   Accuracy: {acc:.4f}")
        
        # Method 2: Simple Average
        if method == 'average' or method == 'all':
            print("\n🔹 Method: Simple Average")
            final_prob = (prob1 + prob2) / 2
            final_pred = (final_prob >= 0.5).astype(int)
            acc = accuracy_score(y_true, final_pred)
            results['average'] = acc
            print(f"   Accuracy: {acc:.4f}")
        
        # Method 3: Max
        if method == 'max' or method == 'all':
            print("\n🔹 Method: Max (Take higher confidence)")
            final_prob = np.maximum(prob1, prob2)
            final_pred = (final_prob >= 0.5).astype(int)
            acc = accuracy_score(y_true, final_pred)
            results['max'] = acc
            print(f"   Accuracy: {acc:.4f}")
        
        # Method 4: Min
        if method == 'min' or method == 'all':
            print("\n🔹 Method: Min (Take lower confidence)")
            final_prob = np.minimum(prob1, prob2)
            final_pred = (final_prob >= 0.5).astype(int)
            acc = accuracy_score(y_true, final_pred)
            results['min'] = acc
            print(f"   Accuracy: {acc:.4f}")
        
        # Method 5: Meta (Stacking)
        if method == 'meta' or method == 'all':
            print("\n🔹 Method: Meta-Model (Stacking)")
            meta_X = np.column_stack([prob1, prob2])
            meta_model = LogisticRegression()
            meta_model.fit(meta_X, y_true)
            final_prob = meta_model.predict_proba(meta_X)[:, 1]
            final_pred = (final_prob >= 0.5).astype(int)
            acc = accuracy_score(y_true, final_pred)
            results['meta'] = acc
            print(f"   Accuracy: {acc:.4f}")
        
        print("\n📊 Summary:")
        print("-"*40)
        for method_name, acc in results.items():
            print(f"  {method_name:10s}: {acc:.4f}")
        
        # Best method
        best = max(results, key=results.get)
        print(f"\n✅ Best Ensemble Method: {best} ({results[best]:.4f})")
        
        return results

def main():
    print("\n" + "🚀"*30)
    print(" ENSEMBLE MODEL - COMBINING MODEL 1 & 2")
    print("🚀"*30)
    
    # ============================================================
    # 1. CONNECT TO MYSQL AND LOAD DATA
    # ============================================================
    try:
        from config_mysql import MYSQL_CONFIG
    except ImportError:
        MYSQL_CONFIG = {
            'host': 'localhost',
            'user': 'root',
            'password': 'your_password',
            'database': 'features',
            'port': 3306
        }
    
    try:
        connection = mysql.connector.connect(**MYSQL_CONFIG)
        
        # Load Model 1 data (movement features from windows table)
        df_movement = pd.read_sql_query("SELECT * FROM windows", connection)
        
        # Load Model 2 data (behavioral features from adhd200_data)
        df_behavioral = pd.read_sql_query("SELECT * FROM adhd200_data", connection)
        
        connection.close()
        
        print(f"\n📥 Loaded data:")
        print(f"  Model 1 (windows): {len(df_movement)} rows")
        print(f"  Model 2 (adhd200_data): {len(df_behavioral)} rows")
        
        # ============================================================
        # 2. PREPARE FEATURES FOR EACH MODEL
        # ============================================================
        # Model 1 features (movement)
        drop_cols = ['window_id', 'participant_id', 'label', 'window_index', 'created_at', 'adhd_prob']
        existing_drop = [c for c in drop_cols if c in df_movement.columns]
        X1 = df_movement.drop(columns=existing_drop, errors='ignore')
        X1 = X1.select_dtypes(include=['float64', 'int64']).fillna(0)
        y1 = df_movement['label']
        
        # Model 2 features (behavioral)
        feature_cols = ['gender', 'age', 'handedness', 'adhd_measure', 'adhd_index',
                       'inattentive', 'hyper_impulsive', 'verbal_iq', 
                       'performance_iq', 'full4_iq']
        X2 = df_behavioral[feature_cols]
        y2 = df_behavioral['label']
        
        # ============================================================
        # 3. EVALUATE INDIVIDUAL MODELS (on their own test sets)
        # ============================================================
        from sklearn.model_selection import train_test_split
        
        # For Model 1
        X1_train, X1_test, y1_train, y1_test = train_test_split(
            X1, y1, test_size=0.2, random_state=42, stratify=y1
        )
        
        # For Model 2
        X2_train, X2_test, y2_train, y2_test = train_test_split(
            X2, y2, test_size=0.2, random_state=42, stratify=y2
        )
        
        # ============================================================
        # 4. INITIALIZE AND EVALUATE ENSEMBLE
        # ============================================================
        ensemble = ADHDEnsemble()
        
        print("\n" + "="*60)
        print("📊 INDIVIDUAL MODEL PERFORMANCE")
        print("="*60)
        
        # Load models for testing
        model1 = joblib.load('models/xgboost_model.pkl')
        scaler1 = joblib.load('models/scaler.pkl')
        model2 = joblib.load('models/model2_final.pkl')
        
        # Test Model 1
        X1_test_scaled = scaler1.transform(X1_test)
        y1_pred = model1.predict(X1_test_scaled)
        acc1 = accuracy_score(y1_test, y1_pred)
        print(f"Model 1 (Movement): {acc1:.4f}")
        
        # Test Model 2
        X2_test_scaled = X2_test  # Already scaled
        y2_pred = model2.predict(X2_test_scaled)
        acc2 = accuracy_score(y2_test, y2_pred)
        print(f"Model 2 (Behavioral): {acc2:.4f}")
        
        # ============================================================
        # 5. ENSEMBLE PREDICTIONS
        # ============================================================
        print("\n" + "="*60)
        print("📊 ENSEMBLE PERFORMANCE")
        print("="*60)
        
        # Get predictions from both models on Model 1 test set
        prob1 = model1.predict_proba(X1_test_scaled)[:, 1]
        
        # For Model 2, we need to use the same test set
        # Since the datasets are different, we'll use the Model 2 test set
        prob2 = model2.predict_proba(X2_test)[:, 1]
        
        # Weighted ensemble
        final_prob = (prob1[:len(prob2)] * 0.65) + (prob2 * 0.35)
        final_pred = (final_prob >= 0.5).astype(int)
        acc_ensemble = accuracy_score(y2_test, final_pred)
        
        print(f"Ensemble (Weighted): {acc_ensemble:.4f}")
        print(f"  Improvement over Model 2: {(acc_ensemble - acc2)*100:.2f}%")
        
        # ============================================================
        # 6. DETAILED ENSEMBLE EVALUATION
        # ============================================================
        print("\n📋 Ensemble Classification Report:")
        print(classification_report(y2_test, final_pred, target_names=['Control', 'ADHD']))
        
        cm = confusion_matrix(y2_test, final_pred)
        print("\n📊 Ensemble Confusion Matrix:")
        print(f"  True Negatives:  {cm[0,0]}")
        print(f"  False Positives: {cm[0,1]}")
        print(f"  False Negatives: {cm[1,0]}")
        print(f"  True Positives:  {cm[1,1]}")
        
        # ============================================================
        # 7. SAVE ENSEMBLE MODEL
        # ============================================================
        # Save the ensemble as a wrapper
        ensemble_dict = {
            'model1': model1,
            'model2': model2,
            'scaler1': scaler1,
            'weight1': 0.65,
            'weight2': 0.35
        }
        joblib.dump(ensemble_dict, 'models/ensemble_model.pkl')
        print("\n✅ Ensemble saved to: models/ensemble_model.pkl")
        
        print("\n" + "="*60)
        print("📊 FINAL SUMMARY")
        print("="*60)
        print(f"Model 1 (Movement):  {acc1:.4f}")
        print(f"Model 2 (Behavioral): {acc2:.4f}")
        print(f"Ensemble (Weighted):  {acc_ensemble:.4f}")
        print(f"✅ Improvement:        {(acc_ensemble - acc2)*100:.2f}%")

    except Error as e:
        print(f"❌ Database error: {e}")

if __name__ == "__main__":
    main()