"""IncidentLens — centralized configuration and paths."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_TEST_DIR = PROJECT_ROOT / "data" / "test"

PRIMARY_DATASET_PATH = DATA_RAW_DIR / "synthetic_it_support_tickets.csv"
