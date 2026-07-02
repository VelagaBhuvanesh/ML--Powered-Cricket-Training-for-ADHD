import mysql.connector
import pandas as pd
import joblib
from sqlalchemy import create_engine
from config_mysql import MYSQL_CONFIG

def add_column():
    conn = mysql.connector.connect(**MYSQL_CONFIG)
    cursor = conn.cursor()
    
    # Check if column exists
    cursor.execute("SHOW COLUMNS FROM windows LIKE 'adhd_prob'")
    if not cursor.fetchone():
        print("⚠️ Adding 'adhd_prob' column to windows table...")
        cursor.execute("ALTER TABLE windows ADD COLUMN adhd_prob FLOAT")
        conn.commit()
        print("✓ Column added.")
    else:
        print("✓ Column already exists.")
    
    cursor.close()
    conn.close()

def update_predictions():
    print("🔄 Loading model and scaler...")
    model = joblib.load('models/xgboost_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    # Load data via SQLAlchemy to avoid warnings
    engine = create_engine(f"mysql+mysqlconnector://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}@{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}")
    df = pd.read_sql_query("SELECT * FROM windows", engine)
    
    print(f"✓ Loaded {len(df)} windows")
    
    # Prepare features (same as training)
    drop_cols = ['window_id', 'participant_id', 'label', 'window_index', 'created_at', 'adhd_prob']
    existing_drop = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=existing_drop, errors='ignore')
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
    X = X[numeric_cols].fillna(0)
    
    # Scale and predict
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:, 1]
    
    # Update database
    conn = mysql.connector.connect(**MYSQL_CONFIG)
    cursor = conn.cursor()
    
    for idx, row in df.iterrows():
        cursor.execute(
            "UPDATE windows SET adhd_prob = %s WHERE window_id = %s",
            (float(probs[idx]), int(row['window_id']))
        )
    conn.commit()
    cursor.close()
    conn.close()
    
    print(f"✓ Updated adhd_prob for {len(probs)} windows")

if __name__ == "__main__":
    add_column()
    update_predictions()
    print("✅ Done!")