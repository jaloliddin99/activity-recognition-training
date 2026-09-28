from pathlib import Path


ROOT = Path(__file__).resolve().parent
RAW_DATA_PATH = ROOT / "data/raw/StresSense.csv"
PROCESSED_DIR = ROOT / "data/processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

SAMPLING_RATE = 50
DEFAULT_WINDOW_SIZE = 250
DEFAULT_OVERLAP = 0.8

AXIS_COLUMNS = {
    "accelerometer": ["Ax", "Ay", "Az"],
    "gyroscope": ["Gx", "Gy", "Gz"],
    "magnetometer": ["Mx", "My", "Mz"],
}

SENSOR_PROFILES = {
    "ag": ["accelerometer", "gyroscope"],
    "agm": ["accelerometer", "gyroscope", "magnetometer"],
}

# Column names are resolved case-insensitively and punctuation is ignored.
COLUMN_ALIASES = {
    "timestamp": ["timestamp", "time", "datetime"],
    "user": ["user", "userid", "subject", "subjectid", "participant"],
    "activity": ["activity", "label", "class"],
    "Ax": ["ax", "accx", "accelerometerx"],
    "Ay": ["ay", "accy", "accelerometery"],
    "Az": ["az", "accz", "accelerometerz"],
    "Gx": ["gx", "gyrox", "gyroscopex"],
    "Gy": ["gy", "gyroy", "gyroscopey"],
    "Gz": ["gz", "gyroz", "gyroscopez"],
    "Mx": ["mx", "magx", "magnetometerx"],
    "My": ["my", "magy", "magnetometery"],
    "Mz": ["mz", "magz", "magnetometerz"],
}
