import pandas as pd
import mysql.connector
from mysql.connector import Error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def train_model2():
    print("="*60)
    print("🧠 TRAINING MODEL 2: BEHAVIORAL ADHD DETECTION")
    print("📊 ONLY ON ADHD-200 DATA TABLE")
    print("="*60)
    
    try:
        from config_mysql import MYSQL_CONFIG
        print("✓ Using config from config_mysql.py")
    except ImportError:
        print("⚠️ config_mysql.py not found! Using default...")
        MYSQL_CONFIG = {
            'host': 'localhost',
            'user': 'root',
            'password': 'chinna12',  # CHANGE THIS
            'database': 'features',
            'port': 3306
        }
    
    try:
        # Connect to MySQL
        connection = mysql.connector.connect(**MYSQL_CONFIG)
        print("✓ Connected to MySQL")
        
        # ============================================================
        # LOAD ONLY ADHD-200 DATA (NOT windows table)
        # ============================================================
        query = "SELECT * FROM adhd200_data"
        df = pd.read_sql_query(query, connection)
        connection.close()
        
        print(f"\n📥 Loaded: {len(df)} rows from 'adhd200_data'")
        print(f"  Columns: {df.columns.tolist()}")
        print(f"  ADHD (label=1): {sum(df['label']==1)}")
        print(f"  Control (label=0): {sum(df['label']==0)}")
        
        # ============================================================
        # DEFINE FEATURES FOR MODEL 2 (BEHAVIORAL DATA ONLY)
        # ============================================================
        feature_cols = [
            'gender',           # 0=Female, 1=Male
            'age',              # Age in years
            'handedness',       # 0=Left, 1=Right
            'adhd_measure',     # ADHD symptom measure
            'adhd_index',       # ADHD index score
            'inattentive',      # Inattention score
            'hyper_impulsive',  # Hyperactivity/Impulsivity score
            'verbal_iq',        # Verbal IQ
            'performance_iq',   # Performance IQ
            'full4_iq'          # Full IQ
        ]
        
        # Check if all columns exist
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            print(f"\n⚠️ Missing columns: {missing_cols}")
            print("   Available columns:", df.columns.tolist())
            return
        
        X = df[feature_cols]
        y = df['label']
        
        print(f"\n📊 Features: {len(feature_cols)} columns")
        print(f"   {feature_cols}")
        
        # ============================================================
        # TRAIN/TEST SPLIT
        # ============================================================
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n📊 Train set: {len(X_train)} rows")
        print(f"  ADHD: {sum(y_train==1)}")
        print(f"  Control: {sum(y_train==0)}")
        print(f"\n📊 Test set: {len(X_test)} rows")
        print(f"  ADHD: {sum(y_test==1)}")
        print(f"  Control: {sum(y_test==0)}")
        
        # ============================================================
        # TRAIN RANDOM FOREST CLASSIFIER
        # ============================================================
        print("\n🤖 Training Random Forest Classifier...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        model.fit(X_train, y_train)
        print("✓ Model trained")
        
        # ============================================================
        # EVALUATE ON TEST SET
        # ============================================================
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n✅ Model 2 Accuracy: {accuracy:.4f}")
        
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Control', 'ADHD']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\n📊 Confusion Matrix:")
        print(f"  True Negatives:  {cm[0,0]}")
        print(f"  False Positives: {cm[0,1]}")
        print(f"  False Negatives: {cm[1,0]}")
        print(f"  True Positives:  {cm[1,1]}")
        
        # ============================================================
        # FEATURE IMPORTANCE
        # ============================================================
        importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\n🏆 Top Features (by importance):")
        print(importance.to_string(index=False))
        
        # ============================================================
        # SAVE MODEL
        # ============================================================
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/model2_behavioral.pkl')
        print(f"\n✅ Model 2 saved to: models/model2_behavioral.pkl")
        
        # Save feature names for inference
        with open('models/model2_features.txt', 'w') as f:
            f.write(','.join(feature_cols))
        print("✅ Feature names saved to: models/model2_features.txt")
        
        return model
        
    except Error as e:
        print(f"❌ Database error: {e}")
        return None

if __name__ == "__main__":
    train_model2()