import pandas as pd
import os

# Path to your metadata file
metadata_path = "data/raw/Demographic and mental health data.csv"

# Load and inspect
df = pd.read_csv(metadata_path)

print("📊 Metadata File Columns:")
print("=" * 50)
print(df.columns.tolist())
print("\n" + "=" * 50)
print("\n📋 First 5 rows:")
print(df.head())

print("\n📊 Data Types:")
print(df.dtypes)