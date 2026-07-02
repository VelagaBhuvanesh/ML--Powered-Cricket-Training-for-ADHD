import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

# MySQL Connection Configuration
MYSQL_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'chinna12',  # CHANGE THIS
    'database': 'features',
    'port': 3306
}

# Sampling Parameters
SAMPLE_RATE = 100  # Hz
WINDOW_SIZE = 5    # seconds
WINDOW_STRIDE = 2  # seconds (50% overlap)
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SIZE  # 500 samples per window

# Sensor Channels
SENSOR_CHANNELS = ['accx', 'accy', 'accz']

# ADHD Threshold (SNAP-IV score >= 24 indicates ADHD)
ADHD_THRESHOLD = 24

# SNAP-IV columns
SNAP_COLUMNS = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10',
                'a11', 'a12', 'a13', 'a14', 'a15', 'a16', 'a17', 'a18', 'a19', 'a20',
                'a21', 'a22', 'a23', 'a24', 'a25', 'a26']

# Batch size for inserting data
BATCH_SIZE = 50000