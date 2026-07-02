import mysql.connector
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import os

# ===== CONFIGURATION =====
PROJECT_ID = "ml-model-501211"
DATASET_ID = "ml"

# MySQL Connection
MYSQL_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'chinna12',  # CHANGE THIS
    'database': 'features'
}

# Service Account Key File
CREDENTIALS_PATH = "keys\credentials.json"  # Must be in project folder

# Tables to transfer
TABLES = ['participants', 'windows', 'feature_columns']
# =========================

def get_bigquery_client():
    """Initialize BigQuery client using service account key."""
    if os.path.exists(CREDENTIALS_PATH):
        print(f"✓ Using credentials from: {CREDENTIALS_PATH}")
        credentials = service_account.Credentials.from_service_account_file(
            CREDENTIALS_PATH
        )
        return bigquery.Client(credentials=credentials, project=PROJECT_ID)
    else:
        print(f"❌ Credentials file '{CREDENTIALS_PATH}' not found!")
        print("   Please download the service account key and save it as credentials.json")
        exit(1)

def transfer_mysql_to_bigquery():
    print("🚀 Transferring MySQL to BigQuery...")
    print(f"   Project: {PROJECT_ID}")
    print(f"   Dataset: {DATASET_ID}")
    print("-" * 50)
    
    # Connect to MySQL
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        print("✓ Connected to MySQL")
    except Exception as e:
        print(f"❌ MySQL connection failed: {e}")
        return
    
    # Connect to BigQuery
    client = get_bigquery_client()
    print("✓ Connected to BigQuery")
    
    # Create dataset if it doesn't exist
    dataset_ref = client.dataset(DATASET_ID)
    try:
        client.get_dataset(dataset_ref)
        print(f"✓ Dataset {DATASET_ID} exists")
    except:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "US"
        client.create_dataset(dataset)
        print(f"✓ Created dataset {DATASET_ID}")
    
    # Transfer each table
    for table_name in TABLES:
        print(f"\n📦 Transferring: {table_name}")
        
        # Read from MySQL
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            print(f"  ✓ Read {len(df)} rows from MySQL")
        except Exception as e:
            print(f"  ❌ Error reading from MySQL: {e}")
            continue
        
        # Upload to BigQuery
        try:
            table_ref = dataset_ref.table(table_name)
            
            job_config = bigquery.LoadJobConfig()
            job_config.autodetect = True
            job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
            
            job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
            job.result()  # Wait for completion
            
            table = client.get_table(table_ref)
            print(f"  ✓ Uploaded {table.num_rows} rows to BigQuery")
        except Exception as e:
            print(f"  ❌ Upload failed: {e}")
    
    conn.close()
    
    print("\n" + "="*50)
    print("✅ Transfer complete!")
    print(f"📍 Location: {PROJECT_ID}.{DATASET_ID}")
    print("="*50)

if __name__ == "__main__":
    transfer_mysql_to_bigquery()