import mysql.connector
import pandas as pd
import numpy as np
from mysql.connector import Error
import warnings
warnings.filterwarnings('ignore')

class MySQLHandler:
    def __init__(self, config: dict):
        self.config = config
        self.connection = None
        self.cursor = None
        self._connect()
        self._fix_schema_if_needed()
        self._initialize_database()
    
    def _connect(self):
        """Establish MySQL connection"""
        try:
            self.connection = mysql.connector.connect(**self.config)
            self.cursor = self.connection.cursor()
            print(f"✓ Connected to MySQL database: {self.config['database']}")
        except Error as e:
            print(f"✗ Error connecting to MySQL: {e}")
            raise
    
    def _fix_schema_if_needed(self):
        """Check and fix participant_id column type if needed"""
        try:
            # Check if participants table exists
            self.cursor.execute("SHOW TABLES LIKE 'participants'")
            if self.cursor.fetchone():
                # Check column type
                self.cursor.execute("DESCRIBE participants")
                columns = self.cursor.fetchall()
                for col in columns:
                    if col[0] == 'participant_id':
                        current_type = col[1]
                        if 'int' in current_type.lower():
                            print("⚠️ participant_id is INTEGER, updating to VARCHAR...")
                            # Drop and recreate tables
                            self.cursor.execute("DROP TABLE IF EXISTS windows")
                            self.cursor.execute("DROP TABLE IF EXISTS participants")
                            self.connection.commit()
                            print("✓ Tables recreated with VARCHAR for participant_id")
                        break
        except Error as e:
            print(f"⚠️ Could not check schema: {e}")
    
    def _initialize_database(self):
        """Create all tables if they don't exist"""
        try:
            # Create participants table with VARCHAR
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS participants (
                    participant_id VARCHAR(20) PRIMARY KEY,
                    age INT,
                    gender VARCHAR(20),
                    bmi FLOAT,
                    sdq_total INT,
                    snap_total INT,
                    label INT,
                    adhd_prob FLOAT DEFAULT NULL
                )
            ''')
            
            # Create windows table with VARCHAR
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS windows (
                    window_id INT AUTO_INCREMENT PRIMARY KEY,
                    participant_id VARCHAR(20),
                    label INT,
                    window_index INT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create feature columns tracking table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS feature_columns (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    column_name VARCHAR(100) UNIQUE,
                    data_type VARCHAR(20) DEFAULT 'FLOAT'
                )
            ''')
            
            # Create pipeline_log table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS pipeline_log (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    run_id VARCHAR(50),
                    step VARCHAR(50),
                    message TEXT,
                    records_affected INT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            self.connection.commit()
            print("✓ Database tables initialized")
            
        except Error as e:
            print(f"✗ Error initializing database: {e}")
            raise
    
    def _convert_to_native_types(self, row):
        """Convert numpy/pandas types to Python native types for MySQL"""
        converted = []
        for val in row:
            if pd.isna(val):
                converted.append(None)
            elif isinstance(val, (np.integer, np.int64)):
                converted.append(int(val))
            elif isinstance(val, (np.floating, np.float64)):
                converted.append(float(val))
            elif isinstance(val, np.bool_):
                converted.append(bool(val))
            else:
                converted.append(val)
        return tuple(converted)
    
    def insert_participants(self, df: pd.DataFrame):
        """Insert participant metadata"""
        try:
            # Define columns to insert
            columns = ['participant_id', 'age', 'gender', 'bmi', 
                       'sdq_total', 'snap_total', 'label']
            
            # Check which columns exist in the dataframe
            available_cols = [c for c in columns if c in df.columns]
            
            # Create a copy with only available columns
            clean_df = df[available_cols].copy()
            
            # Handle NaN values
            clean_df = clean_df.replace([np.inf, -np.inf], np.nan)
            
            # Convert participant_id to string
            if 'participant_id' in clean_df.columns:
                clean_df['participant_id'] = clean_df['participant_id'].astype(str)
            
            # Fill NaN values with 0 for numeric columns, 'Unknown' for string columns
            for col in clean_df.columns:
                if col == 'participant_id':
                    continue  # Don't fill participant_id
                if clean_df[col].dtype in ['float64', 'int64']:
                    clean_df[col] = clean_df[col].fillna(0)
                else:
                    clean_df[col] = clean_df[col].fillna('Unknown')
            
            # Convert to list of tuples with native types
            records = clean_df.to_records(index=False)
            raw_values = [tuple(row) for row in records]
            
            if not raw_values:
                print("  ℹ️ No participant data to insert")
                return
            
            # Convert each row to native Python types
            values = [self._convert_to_native_types(row) for row in raw_values]
            
            # Build insert query with available columns
            placeholders = ', '.join(['%s'] * len(available_cols))
            query = f"""
                INSERT INTO participants ({', '.join(available_cols)}) 
                VALUES ({placeholders})
                ON DUPLICATE KEY UPDATE
            """
            
            # Add update clauses for each column
            update_clauses = []
            for col in available_cols:
                if col != 'participant_id':
                    update_clauses.append(f"{col} = VALUES({col})")
            query += ', '.join(update_clauses)
            
            # Execute in batches
            batch_size = 1000
            for i in range(0, len(values), batch_size):
                batch = values[i:i+batch_size]
                self.cursor.executemany(query, batch)
            
            self.connection.commit()
            print(f"✓ Inserted/Updated {len(df)} participants")
            
        except Error as e:
            print(f"✗ Error inserting participants: {e}")
            self.connection.rollback()
            raise
    
    def insert_windows(self, df: pd.DataFrame):
        """Insert window data with features"""
        if df.empty:
            print("✗ No windows to insert")
            return
        
        try:
            # Convert participant_id to string
            df['participant_id'] = df['participant_id'].astype(str)
            
            # Ensure label is int
            if 'label' in df.columns:
                df['label'] = df['label'].astype(int)
            
            feature_cols = [col for col in df.columns if col not in 
                           ['participant_id', 'label', 'window_index']]
            
            for col in feature_cols:
                self.add_feature_column(col)
            
            columns = ['participant_id', 'label', 'window_index'] + feature_cols
            placeholders = ', '.join(['%s'] * len(columns))
            query = f"INSERT INTO windows ({', '.join(columns)}) VALUES ({placeholders})"
            
            # Convert to list of tuples
            raw_values = df[columns].values.tolist()
            
            # Convert each row to native Python types
            values = [self._convert_to_native_types(row) for row in raw_values]
            
            # Insert in batches
            batch_size = 5000
            total_inserted = 0
            
            for i in range(0, len(values), batch_size):
                batch = values[i:i+batch_size]
                self.cursor.executemany(query, batch)
                total_inserted += len(batch)
                print(f"  Inserted {total_inserted:,} windows...", end='\r')
            
            self.connection.commit()
            print(f"\n✓ Inserted {len(df)} windows")
            
        except Error as e:
            print(f"✗ Error inserting windows: {e}")
            self.connection.rollback()
            raise
    
    def add_feature_column(self, column_name: str):
        """Dynamically add a feature column to windows table"""
        try:
            self.cursor.execute("""
                SELECT COLUMN_NAME 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_NAME = 'windows' AND COLUMN_NAME = %s
            """, (column_name,))
            
            if not self.cursor.fetchone():
                self.cursor.execute(f"ALTER TABLE windows ADD COLUMN {column_name} FLOAT")
                self.cursor.execute("""
                    INSERT IGNORE INTO feature_columns (column_name) 
                    VALUES (%s)
                """, (column_name,))
                self.connection.commit()
                return True
            return False
            
        except Error as e:
            print(f"✗ Error adding column {column_name}: {e}")
            return False
    
    def log_pipeline_step(self, run_id: str, step: str, message: str, records: int = 0):
        """Log pipeline execution steps"""
        try:
            query = """
                INSERT INTO pipeline_log (run_id, step, message, records_affected)
                VALUES (%s, %s, %s, %s)
            """
            self.cursor.execute(query, (run_id, step, message, records))
            self.connection.commit()
        except Error as e:
            print(f"Warning: Could not log step: {e}")
    
    def get_summary(self) -> dict:
        """Get dataset summary"""
        try:
            self.cursor.execute("SELECT COUNT(*) FROM windows")
            total_windows = self.cursor.fetchone()[0]
            
            self.cursor.execute("SELECT label, COUNT(*) FROM windows GROUP BY label")
            label_counts = {row[0]: row[1] for row in self.cursor.fetchall()}
            
            self.cursor.execute("SELECT COUNT(*) FROM participants")
            total_participants = self.cursor.fetchone()[0]
            
            self.cursor.execute("SELECT COUNT(*) FROM feature_columns")
            feature_count = self.cursor.fetchone()[0]
            
            return {
                'total_windows': total_windows,
                'label_counts': label_counts,
                'total_participants': total_participants,
                'feature_count': feature_count
            }
            
        except Error as e:
            print(f"✗ Error getting summary: {e}")
            return {}
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
            print("✓ Database connection closed")