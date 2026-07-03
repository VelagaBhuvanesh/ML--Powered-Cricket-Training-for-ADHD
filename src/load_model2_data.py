import pandas as pd
import mysql.connector
from mysql.connector import Error
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_scaled_data():
    print("="*60)
    print("💾 LOADING SCALED TRAINING DATA INTO MYSQL")
    print("="*60)
    
    # ============================================================
    # 1. FIND THE SCALED TRAINING CSV
    # ============================================================
    possible_paths = [
        '../data/raw/adhd200_balanced_train.csv',
        'data/raw/adhd200_balanced_train.csv',
        'adhd200_balanced_train.csv'
    ]
    file_path = None
    for path in possible_paths:
        if os.path.exists(path):
            file_path = path
            print(f"✅ Found training file at: {path}")
            break
    
    if file_path is None:
        print("❌ adhd200_balanced_train.csv not found!")
        return
    
    # ============================================================
    # 2. LOAD THE CSV
    # ============================================================
    df = pd.read_csv(file_path)
    print(f"\n📥 Loaded: {len(df)} rows")
    print(f"  Columns: {df.columns.tolist()}")
    
    # Ensure label column is named 'label'
    if 'DX' in df.columns:
        df = df.rename(columns={'DX': 'label'})
    elif 'dx' in df.columns:
        df = df.rename(columns={'dx': 'label'})
    
    # Ensure all column names are lowercase
    df.columns = df.columns.str.lower()
    
    # Add a participant_id column
    df['participant_id'] = ['SCALED_' + str(i) for i in range(len(df))]
    
    print(f"\n📊 Preparing to load with columns:")
    print(df.columns.tolist())
    
    # ============================================================
    # 3. CONNECT TO MYSQL
    # ============================================================
    try:
        from config_mysql import MYSQL_CONFIG
        print("\n✓ Using config from config_mysql.py")
    except ImportError:
        print("\n⚠️ config_mysql.py not found! Using default...")
        MYSQL_CONFIG = {
            'host': 'localhost',
            'user': 'root',
            'password': 'chinna12',
            'database': 'features',
            'port': 3306
        }
    
    try:
        connection = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = connection.cursor()
        print("✓ Connected to MySQL")
        
        # ============================================================
        # 4. DROP OLD TABLE AND RECREATE
        # ============================================================
        print("\n🔄 Dropping old table...")
        cursor.execute("DROP TABLE IF EXISTS adhd200_data")
        print("✓ Dropped existing table")
        
        create_table_query = """
        CREATE TABLE adhd200_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            gender FLOAT,
            age FLOAT,
            handedness FLOAT,
            adhd_measure FLOAT,
            adhd_index FLOAT,
            inattentive FLOAT,
            hyper_impulsive FLOAT,
            verbal_iq FLOAT,
            performance_iq FLOAT,
            full4_iq FLOAT,
            label INT,
            participant_id VARCHAR(50)
        )
        """
        cursor.execute(create_table_query)
        print("✓ Table 'adhd200_data' recreated")
        
        # ============================================================
        # 5. INSERT SCALED DATA
        # ============================================================
        # Define column mapping
        feature_cols = ['gender', 'age', 'handedness', 'adhd_measure', 'adhd_index',
                       'inattentive', 'hyper_impulsive', 'verbal_iq', 
                       'performance_iq', 'full4_iq']
        
        insert_query = """
        INSERT INTO adhd200_data 
        (gender, age, handedness, adhd_measure, adhd_index, 
         inattentive, hyper_impulsive, verbal_iq, performance_iq, full4_iq, 
         label, participant_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        data_to_insert = []
        for idx, row in df.iterrows():
            values = (
                float(row['gender']),
                float(row['age']),
                float(row['handedness']),
                float(row['adhd_measure']),
                float(row['adhd_index']),
                float(row['inattentive']),
                float(row['hyper_impulsive']),
                float(row['verbal_iq']),
                float(row['performance_iq']),
                float(row['full4_iq']),
                int(row['label']),
                row['participant_id']
            )
            data_to_insert.append(values)
        
        cursor.executemany(insert_query, data_to_insert)
        connection.commit()
        
        print(f"✓ Loaded {len(data_to_insert)} scaled records into 'adhd200_data'")
        
        # ============================================================
        # 6. VERIFY
        # ============================================================
        cursor.execute("SELECT label, COUNT(*) FROM adhd200_data GROUP BY label")
        results = cursor.fetchall()
        print("\n📊 Final distribution:")
        for label, count in results:
            print(f"  {'ADHD' if label==1 else 'Control'}: {count}")
        
        cursor.execute("SELECT * FROM adhd200_data LIMIT 5")
        print("\n📊 Sample of inserted data:")
        for row in cursor.fetchall():
            print(row)
        
    except Error as e:
        print(f"❌ Database error: {e}")
    
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("✓ MySQL connection closed")

if __name__ == "__main__":
    load_scaled_data()