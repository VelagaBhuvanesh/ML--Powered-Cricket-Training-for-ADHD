import os
import re
import pandas as pd
import glob
from config_mysql import DATA_RAW_DIR

class Extractor:
    def __init__(self):
        self.metadata = None
        self.id_mapping = None
        self.stats = {'participants_total': 0}
    
    def extract_participant_id_from_filename(self, filename: str) -> str:
        """
        Extract participant ID from filename.
        
        Handles formats like:
        - H1_F.csv → H1
        - W4_F.csv → W4
        - B26_F.csv → B26
        - 0001_F.csv → 1
        """
        # Remove extension
        name = filename.replace('.csv', '')
        
        # Pattern: letters followed by digits followed by _F
        match = re.search(r'([A-Za-z]+)(\d+)_F', name)
        if match:
            return match.group(1) + match.group(2)  # e.g., H1, W4, B26
        
        # Alternative: just digits
        match = re.search(r'(\d+)_F', name)
        if match:
            return match.group(1)
        
        # If no ID found, return None
        return None
    
    def extract(self):
        """Load all raw data"""
        print("\n" + "="*60)
        print("📥 EXTRACT: Loading Raw Data")
        print("="*60)
        
        # Load metadata
        metadata_path = os.path.join(DATA_RAW_DIR, "Demographic and mental health data.csv")
        if not os.path.exists(metadata_path):
            print(f"✗ Metadata file not found: {metadata_path}")
            return False
        
        self.metadata = pd.read_csv(metadata_path)
        print(f"✓ Loaded metadata: {len(self.metadata)} participants")
        print(f"✓ Columns: {self.metadata.columns.tolist()[:10]}...")
        
        # Get all F.csv files
        f_files = glob.glob(os.path.join(DATA_RAW_DIR, "*_F.csv"))
        f_files.sort()
        
        if not f_files:
            print("✗ No _F.csv files found in data/raw/")
            return False
        
        print(f"✓ Found {len(f_files)} _F.csv files")
        
        # Show sample files
        print("\n  Sample files found:")
        for f in f_files[:10]:
            print(f"    - {os.path.basename(f)}")
        if len(f_files) > 10:
            print(f"    ... and {len(f_files) - 10} more")
        
        # Create ID mapping from files
        mapping_data = []
        for file_path in f_files:
            file_name = os.path.basename(file_path)
            participant_id = self.extract_participant_id_from_filename(file_name)
            
            if participant_id is not None:
                mapping_data.append({
                    'ID': participant_id,
                    'File_name': file_name
                })
            else:
                print(f"  ⚠️ Could not extract ID from: {file_name}")
        
        if not mapping_data:
            print("✗ No IDs could be extracted from filenames")
            return False
        
        self.id_mapping = pd.DataFrame(mapping_data)
        
        # Sort by ID
        self.id_mapping = self.id_mapping.sort_values('ID').reset_index(drop=True)
        
        print(f"✓ Created ID mapping: {len(self.id_mapping)} participants")
        
        # Save mapping for future use
        mapping_path = os.path.join(DATA_RAW_DIR, "ID_activity_file.csv")
        self.id_mapping.to_csv(mapping_path, index=False)
        print(f"✓ Saved mapping to: {mapping_path}")
        
        self.stats['participants_total'] = len(self.metadata)
        
        # Preview mapping
        print("\n  ID Mapping preview:")
        print(self.id_mapping.head(10).to_string())
        
        # Check which participants have data
        metadata_ids = set(self.metadata['ID'].tolist())
        data_ids = set(self.id_mapping['ID'].tolist())
        missing_data = metadata_ids - data_ids
        extra_data = data_ids - metadata_ids
        
        print(f"\n  Participants in metadata: {len(metadata_ids)}")
        print(f"  Participants with data files: {len(data_ids)}")
        if missing_data:
            print(f"  ⚠️ Participants missing data files: {sorted(missing_data)[:10]}...")
        if extra_data:
            print(f"  ℹ️ Extra data files with no metadata: {sorted(extra_data)[:10]}...")
        
        return True
    
    def get_metadata(self):
        return self.metadata
    
    def get_id_mapping(self):
        return self.id_mapping
    
    def get_stats(self):
        return self.stats


# Quick test function
def test_extractor():
    """Test the extractor with sample files"""
    print("Testing Extractor...")
    
    test_files = [
        "H1_F.csv", "W4_F.csv", "Z5_F.csv", "B26_F.csv", "C10_F.csv"
    ]
    
    extractor = Extractor()
    
    print("\nFilename parsing test:")
    for f in test_files:
        pid = extractor.extract_participant_id_from_filename(f)
        print(f"  {f} → ID: {pid}")
    
    print("\n✓ Extractor test complete")


if __name__ == "__main__":
    test_extractor()