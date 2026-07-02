import pandas as pd
import numpy as np
from mysql_handler import MySQLHandler
from config_mysql import MYSQL_CONFIG, SNAP_COLUMNS

class Loader:
    def __init__(self):
        self.db = MySQLHandler(MYSQL_CONFIG)
    
    def calculate_snap_total(self, row):
        """Calculate SNAP-IV total score from a1-a26 columns"""
        total = 0
        valid_count = 0
        for col in SNAP_COLUMNS:
            if col in row and pd.notna(row[col]):
                total += row[col]
                valid_count += 1
        if valid_count == 26:
            return total
        elif valid_count > 0:
            return (total / valid_count) * 26
        else:
            return np.nan
    
    def calculate_sdq_total(self, row):
        """Calculate SDQ total score from SDQ1-SDQ25 columns"""
        sdq_cols = [f'SDQ{i}' for i in range(1, 26)]
        total = 0
        valid_count = 0
        for col in sdq_cols:
            if col in row and pd.notna(row[col]):
                total += row[col]
                valid_count += 1
        if valid_count == 25:
            return total
        elif valid_count > 0:
            return (total / valid_count) * 25
        else:
            return np.nan
    
    def load(self, metadata, features):
        """Load data into MySQL database"""
        print("\n" + "="*60)
        print("💾 LOAD: Storing Data in MySQL")
        print("="*60)
        
        # Prepare participant metadata with correct column names
        participants_df = metadata.copy()
        participants_df = participants_df.rename(columns={
            'ID': 'participant_id',
            'SEX': 'gender',        # Rename SEX to gender
            'BMI': 'bmi'            # Rename BMI to bmi (lowercase)
        })
        
        # Add age column if not present
        if 'age' not in participants_df.columns:
            # If age is not available, set to None
            participants_df['age'] = np.nan
            print("  ℹ️ 'age' column not found in metadata. Setting to NULL.")
        
        # Add SDQ total
        if 'sdq_total' not in participants_df.columns:
            participants_df['sdq_total'] = participants_df.apply(self.calculate_sdq_total, axis=1)
            print(f"  ✓ Calculated SDQ total scores")
        
        # Add SNAP total
        if 'snap_total' not in participants_df.columns:
            participants_df['snap_total'] = participants_df.apply(self.calculate_snap_total, axis=1)
            print(f"  ✓ Calculated SNAP total scores")
        
        # Add label based on SNAP total
        participants_df['label'] = participants_df['snap_total'].apply(
            lambda x: 1 if x >= 24 else 0
        )
        
        print(f"\n  Participant Summary:")
        print(f"  - Total participants: {len(participants_df)}")
        print(f"  - ADHD (SNAP >= 24): {participants_df['label'].sum()}")
        print(f"  - Control (SNAP < 24): {len(participants_df) - participants_df['label'].sum()}")
        print(f"  - SNAP Scores: Min={participants_df['snap_total'].min():.1f}, "
              f"Max={participants_df['snap_total'].max():.1f}, "
              f"Mean={participants_df['snap_total'].mean():.1f}")
        
        # Insert participants
        print("\n  Inserting participant metadata...")
        self.db.insert_participants(participants_df)
        
        # Insert windows
        print("\n  Inserting windows with features...")
        if not features.empty:
            self.db.insert_windows(features)
        
        # Get summary
        summary = self.db.get_summary()
        print("\n  Database Summary:")
        print(f"  - Total windows: {summary['total_windows']:,}")
        print(f"  - ADHD windows: {summary['label_counts'].get(1, 0):,}")
        print(f"  - Control windows: {summary['label_counts'].get(0, 0):,}")
        print(f"  - Total participants: {summary['total_participants']}")
        print(f"  - Features per window: {summary['feature_count']}")
        
        return summary
    
    def close(self):
        self.db.close()