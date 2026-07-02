import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from feature_extraction import FeatureExtractor
from config_mysql import ADHD_THRESHOLD, SNAP_COLUMNS, DATA_RAW_DIR

class Transformer:
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.combined_features = None
        self.stats = {
            'participants_processed': 0,
            'participants_skipped': 0,
            'windows_generated': 0,
            'skipped_ids': []
        }
    
    def find_sensor_columns(self, df):
        """
        Find accelerometer columns in the dataframe.
        Returns dict with {'accx': 'actual_column_name', ...}
        """
        # Common patterns for accelerometer columns
        patterns = {
            'accx': ['accx', 'acc_x', 'AccX', 'Acc_X', 'x', 'X', 'accel_x', 'AccelX'],
            'accy': ['accy', 'acc_y', 'AccY', 'Acc_Y', 'y', 'Y', 'accel_y', 'AccelY'],
            'accz': ['accz', 'acc_z', 'AccZ', 'Acc_Z', 'z', 'Z', 'accel_z', 'AccelZ']
        }
        
        found = {}
        
        # First, try exact matches
        for target, possible_names in patterns.items():
            for col in df.columns:
                if col.lower() in [p.lower() for p in possible_names]:
                    found[target] = col
                    break
        
        # If we have all 3, return them
        if len(found) == 3:
            return found
        
        # If not, try to find by position (common order: x, y, z)
        # Look for numeric columns that could be accelerometer data
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude common non-accelerometer columns
        exclude = ['timestamp', 'time', 'hr', 'heart_rate', 'battery', 'confidence', 
                   'mag_x', 'mag_y', 'mag_z', 'gyro_x', 'gyro_y', 'gyro_z']
        potential_cols = [c for c in numeric_cols if c.lower() not in exclude]
        
        # Check if we have 3 potential columns
        if len(potential_cols) >= 3:
            # Take first 3 as accx, accy, accz
            found = {
                'accx': potential_cols[0],
                'accy': potential_cols[1],
                'accz': potential_cols[2]
            }
            return found
        
        # If still nothing, check for 'acc' in column names
        acc_cols = [c for c in df.columns if 'acc' in c.lower()]
        if len(acc_cols) >= 3:
            found = {
                'accx': acc_cols[0],
                'accy': acc_cols[1],
                'accz': acc_cols[2]
            }
            return found
        
        return None
    
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
    
    def transform(self, metadata, id_mapping):
        """Extract features from raw data"""
        print("\n" + "="*60)
        print("🔄 TRANSFORM: Extracting Features")
        print("="*60)
        
        # Calculate SNAP total score for each participant
        metadata['SNAP_total'] = metadata.apply(self.calculate_snap_total, axis=1)
        
        # Add labels based on SNAP total
        metadata['label'] = metadata['SNAP_total'].apply(
            lambda x: 1 if x >= ADHD_THRESHOLD else 0
        )
        
        print(f"✓ Calculated SNAP total scores")
        print(f"\n  SNAP Score Statistics:")
        print(f"  - Min: {metadata['SNAP_total'].min():.1f}")
        print(f"  - Max: {metadata['SNAP_total'].max():.1f}")
        print(f"  - Mean: {metadata['SNAP_total'].mean():.1f}")
        print(f"  - Median: {metadata['SNAP_total'].median():.1f}")
        
        print(f"\n  Label distribution:")
        print(f"  - ADHD (score >= {ADHD_THRESHOLD}): {metadata['label'].sum()} participants")
        print(f"  - Control (score < {ADHD_THRESHOLD}): {len(metadata) - metadata['label'].sum()} participants")
        
        # Get participant IDs that have both metadata AND data files
        metadata_ids = set(metadata['ID'].tolist())
        data_ids = set(id_mapping['ID'].tolist())
        valid_ids = metadata_ids.intersection(data_ids)
        
        print(f"\nProcessing {len(valid_ids)} participants with both metadata and data...")
        
        if not valid_ids:
            print("✗ No matching IDs between metadata and data files!")
            return False
        
        all_features = []
        
        # Create a mapping for fast lookup
        id_to_file = dict(zip(id_mapping['ID'], id_mapping['File_name']))
        
        for pid in tqdm(sorted(valid_ids), desc="Processing participants"):
            try:
                file_name = id_to_file.get(pid)
                if not file_name:
                    self.stats['participants_skipped'] += 1
                    self.stats['skipped_ids'].append(pid)
                    continue
                
                file_path = os.path.join(DATA_RAW_DIR, file_name)
                
                if not os.path.exists(file_path):
                    self.stats['participants_skipped'] += 1
                    self.stats['skipped_ids'].append(pid)
                    continue
                
                # Load sensor data
                sensor_data = pd.read_csv(file_path)
                
                # Find accelerometer columns
                col_map = self.find_sensor_columns(sensor_data)
                
                if col_map is None:
                    print(f"  ⚠️ Participant {pid}: Could not find accelerometer columns")
                    print(f"     Available columns: {sensor_data.columns.tolist()}")
                    self.stats['participants_skipped'] += 1
                    self.stats['skipped_ids'].append(pid)
                    continue
                
                # Rename columns to standard names
                sensor_data = sensor_data.rename(columns={
                    col_map['accx']: 'accx',
                    col_map['accy']: 'accy',
                    col_map['accz']: 'accz'
                })
                
                # Get label from metadata
                label = metadata[metadata['ID'] == pid]['label'].iloc[0]
                
                # Extract features
                participant_features = self.feature_extractor.process_participant(
                    sensor_data, label, pid
                )
                
                if not participant_features.empty:
                    all_features.append(participant_features)
                    self.stats['participants_processed'] += 1
                else:
                    self.stats['participants_skipped'] += 1
                    self.stats['skipped_ids'].append(pid)
                    
            except Exception as e:
                print(f"  ✗ Error processing participant {pid}: {str(e)}")
                self.stats['participants_skipped'] += 1
                self.stats['skipped_ids'].append(pid)
                continue
        
        # Combine all features
        if all_features:
            self.combined_features = pd.concat(all_features, ignore_index=True)
            self.stats['windows_generated'] = len(self.combined_features)
            
            print(f"\n✓ Processed: {self.stats['participants_processed']} participants")
            print(f"✓ Skipped: {self.stats['participants_skipped']} participants")
            print(f"✓ Generated: {self.stats['windows_generated']:,} windows")
            
            # Show feature preview
            feature_cols = [c for c in self.combined_features.columns 
                           if c not in ['participant_id', 'label', 'window_index']]
            print(f"✓ Features per window: {len(feature_cols)}")
            
            return True
        else:
            print("\n✗ No features extracted!")
            return False
    
    def get_features(self):
        return self.combined_features
    
    def get_stats(self):
        return self.stats