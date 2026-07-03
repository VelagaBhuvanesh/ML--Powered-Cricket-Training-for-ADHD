import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def balance_adhd200_data():
    print("="*60)
    print("⚖️ BALANCING ADHD-200 DATA")
    print("="*60)
    
    # Load the cleaned data (try both locations)
    import os
    possible_paths = [
        'data/adhd200_cleaned.csv',
        '../data/adhd200_cleaned.csv',
        'src/data/adhd200_cleaned.csv'
    ]
    
    file_path = None
    for path in possible_paths:
        if os.path.exists(path):
            file_path = path
            print(f"✅ Found file at: {path}")
            break
    
    if file_path is None:
        print("❌ File not found!")
        return
    
    df = pd.read_csv(file_path)
    print(f"\n📥 Loaded: {len(df)} rows")
    
    # Separate features and target
    X = df.drop('DX', axis=1)
    y = df['DX']
    
    print(f"\n📊 Before Balancing:")
    print(f"  ADHD (1): {sum(y==1)}")
    print(f"  Control (0): {sum(y==0)}")
    print(f"  Total: {len(y)}")
    
    # Split first (CRITICAL for validation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    print(f"\n📊 After SMOTE:")
    print(f"  ADHD (1): {sum(y_train_balanced)}")
    print(f"  Control (0): {len(y_train_balanced) - sum(y_train_balanced)}")
    print(f"  Total: {len(y_train_balanced)}")
    
    # Create balanced DataFrame
    feature_names = X.columns.tolist()
    X_train_balanced_df = pd.DataFrame(X_train_balanced, columns=feature_names)
    X_train_balanced_df['DX'] = y_train_balanced
    
    # Save balanced data
    balanced_path = os.path.join(os.path.dirname(file_path), 'adhd200_balanced_train.csv')
    X_train_balanced_df.to_csv(balanced_path, index=False)
    
    # Also save test data (unbalanced, for final evaluation)
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    X_test_df['DX'] = y_test.values
    test_path = os.path.join(os.path.dirname(file_path), 'adhd200_test.csv')
    X_test_df.to_csv(test_path, index=False)
    
    print(f"\n✅ Saved:")
    print(f"  - {balanced_path} ({len(X_train_balanced_df)} rows)")
    print(f"  - {test_path} ({len(X_test_df)} rows)")
    
    return X_train_balanced_df, X_test_df, scaler

if __name__ == "__main__":
    balance_adhd200_data()