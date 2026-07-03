import pandas as pd
import numpy as np
import os

# ============================================================
# AUTO-DETECT FILE FUNCTION
# ============================================================
def find_adhd200_file():
    """
    Search for the ADHD-200 phenotypic file in multiple locations.
    Returns the file path if found, else None.
    """
    possible_locations = [
        # Current directory (where script is run)
        'allSubs_testSet_phenotypic_dx.csv',
        # Project root (one level up from src/)
        '../allSubs_testSet_phenotypic_dx.csv',
        # data/raw/ from project root
        'data/raw/allSubs_testSet_phenotypic_dx.csv',
        # data/raw/ from src/ folder
        '../data/raw/allSubs_testSet_phenotypic_dx.csv',
        # Absolute path (hardcoded example - can be removed)
        r'C:\Users\Velaga Bhuvanesh\Downloads\Cricket ADHD\ML--Powered-Cricket-Training-for-ADHD\allSubs_testSet_phenotypic_dx.csv',
        r'C:\Users\Velaga Bhuvanesh\Downloads\Cricket ADHD\ML--Powered-Cricket-Training-for-ADHD\data\raw\allSubs_testSet_phenotypic_dx.csv',
        # Also check if the file is in the same folder as this script
        os.path.join(os.path.dirname(__file__), 'allSubs_testSet_phenotypic_dx.csv'),
        os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'allSubs_testSet_phenotypic_dx.csv'),
    ]
    
    for loc in possible_locations:
        if os.path.exists(loc):
            print(f"✅ Found file at: {loc}")
            return loc
    
    print("❌ File not found in any searched location!")
    print("\n📁 Current working directory:", os.getcwd())
    print("\n📄 Files in current directory:")
    for f in os.listdir('.'):
        print(f"  - {f}")
    print("\n🔍 Please place the file in one of these locations:")
    print("  1. In the current folder (where you run the script)")
    print("  2. In the project root folder (one level up from src/)")
    print("  3. In data/raw/ folder")
    print("  4. Or update the possible_locations list with your path.")
    return None

# ============================================================
# CLEANING FUNCTION
# ============================================================
def clean_adhd200_data(file_path):
    """
    Clean and prepare ADHD-200 data for Model 2.
    """
    print("="*60)
    print("🧹 CLEANING ADHD-200 DATA")
    print("="*60)
    
    # Load data
    df = pd.read_csv(file_path)
    print(f"\n📥 Loaded: {len(df)} rows, {len(df.columns)} columns")
    
    # ============================================================
    # 1. FIX COLUMN NAMES
    # ============================================================
    print("\n🔧 Fixing column names...")
    df.columns = df.columns.str.strip().str.replace(' ', '_')
    df.columns = df.columns.str.replace('/', '_')
    print(f"  New columns: {df.columns.tolist()}")
    
    # ============================================================
    # 2. HANDLE DIAGNOSIS (DX)
    # ============================================================
    print("\n🎯 Handling Diagnosis (DX)...")
    
    # Display current DX distribution
    print(f"  Current DX distribution:")
    print(df['DX'].value_counts())
    
    # Convert DX to numeric
    # Map: '0' → 0, '1' → 1, others → NaN
    def clean_dx(val):
        try:
            val_int = int(val)
            if val_int in [0, 1]:
                return val_int
            else:
                return np.nan
        except:
            return np.nan
    
    df['DX_clean'] = df['DX'].apply(clean_dx)
    
    # Drop rows with invalid DX
    df = df.dropna(subset=['DX_clean'])
    df['DX'] = df['DX_clean'].astype(int)
    df = df.drop('DX_clean', axis=1)
    
    print(f"  Cleaned DX distribution:")
    print(df['DX'].value_counts())
    
    # ============================================================
    # 3. HANDLE MISSING VALUES
    # ============================================================
    print("\n📊 Handling missing values...")
    
    # Columns to keep for Model 2
    feature_columns = [
        'Gender', 'Age', 'Handedness', 
        'ADHD_Measure', 'ADHD_Index', 
        'Inattentive', 'Hyper_Impulsive',
        'Verbal_IQ', 'Performance_IQ', 'Full4_IQ'
    ]
    
    # Keep only available columns
    available_cols = [col for col in feature_columns if col in df.columns]
    print(f"  Keeping columns: {available_cols}")
    
    df_features = df[available_cols + ['DX']].copy()
    
    # Clean Handedness
    def clean_handedness(val):
        if pd.isna(val):
            return 0
        if isinstance(val, str):
            if val in ['R', '1', '1.0']:
                return 1
            elif val in ['L', '0', '0.0']:
                return 0
        try:
            val_num = float(val)
            if val_num >= 0.5:
                return 1
            else:
                return 0
        except:
            return 0
    
    df_features['Handedness'] = df_features['Handedness'].apply(clean_handedness)
    
    # Convert numeric columns
    numeric_cols = ['ADHD_Measure', 'ADHD_Index', 'Inattentive', 'Hyper_Impulsive',
                   'Verbal_IQ', 'Performance_IQ', 'Full4_IQ']
    
    for col in numeric_cols:
        if col in df_features.columns:
            df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
    
    # Fill missing values with median
    print(f"\n  Missing values before filling:")
    print(df_features.isnull().sum())
    
    for col in numeric_cols:
        if col in df_features.columns:
            median_val = df_features[col].median()
            df_features[col] = df_features[col].fillna(median_val)
            print(f"  Filled {col} with median: {median_val:.2f}")
    
    # Fill Gender (most are 0/1, fill with mode)
    if 'Gender' in df_features.columns:
        mode_gender = df_features['Gender'].mode()[0]
        df_features['Gender'] = df_features['Gender'].fillna(mode_gender)
    
    # Fill Age (fill with mean)
    if 'Age' in df_features.columns:
        mean_age = df_features['Age'].mean()
        df_features['Age'] = df_features['Age'].fillna(mean_age)
    
    # ============================================================
    # 4. FINAL CLEAN
    # ============================================================
    print(f"\n✅ Final data shape: {df_features.shape}")
    print(f"  ADHD (DX=1): {sum(df_features['DX']==1)}")
    print(f"  Control (DX=0): {sum(df_features['DX']==0)}")
    
    # Check for any remaining missing values
    print(f"\n  Remaining missing values:")
    print(df_features.isnull().sum())
    
    # ============================================================
    # 5. SAVE CLEANED DATA
    # ============================================================
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    df_features.to_csv('data/adhd200_cleaned.csv', index=False)
    print(f"\n✅ Saved to 'data/adhd200_cleaned.csv'")
    
    return df_features

# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    # Auto-detect the file
    file_path = find_adhd200_file()
    
    if file_path:
        cleaned_df = clean_adhd200_data(file_path)
        
        print("\n📊 Cleaned Data Preview:")
        print(cleaned_df.head())
        print("\n📊 Summary Statistics:")
        print(cleaned_df.describe())
    else:
        print("\n❌ Cannot proceed without the file.")