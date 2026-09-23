"""
IncidentLens — data loading module.

Loads the primary incident ticket dataset with validation on file
existence, encoding, schema, and content integrity. Fails loudly and
clearly rather than letting a bad file silently propagate downstream.
"""
from pathlib import Path
import logging

import pandas as pd

from src.config import PRIMARY_DATASET_PATH

logger = logging.getLogger(__name__)

# Columns IncidentLens actually depends on. If any is missing, loading
# fails now instead of breaking mysteriously in a later stage.
REQUIRED_COLUMNS = [
    "ticket_id",
    "created_at",
    "issue_type",
    "priority",
    "status",
    "initial_message",
    "resolution_summary",
]


class DataLoadError(Exception):
    """Raised when the incident dataset cannot be loaded or is invalid."""


def load_tickets(path) -> pd.DataFrame:
    """
    Load the primary incident ticket dataset from a CSV file.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        The loaded, validated (but not yet cleaned) ticket data.

    Raises
    ------
    DataLoadError
        If the file is missing, not a file, empty, malformed, has a
        bad encoding, has zero rows, or is missing required columns.
    """
    path = Path(path)

    if not path.exists():
        raise DataLoadError(f"Dataset file not found: {path}")

    if not path.is_file():
        raise DataLoadError(f"Path is not a file: {path}")

    if path.stat().st_size == 0:
        raise DataLoadError(f"Dataset file is empty: {path}")

    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError as e:
        raise DataLoadError(f"Dataset file is not valid UTF-8: {path}") from e
    except pd.errors.EmptyDataError as e:
        raise DataLoadError(f"Dataset file has no columns/data: {path}") from e
    except pd.errors.ParserError as e:
        raise DataLoadError(f"Dataset file is malformed CSV: {path}") from e

    if df.empty:
        raise DataLoadError(f"Dataset loaded but contains zero rows: {path}")

    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise DataLoadError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    logger.info(
        "Loaded %d rows, %d columns from %s", len(df), len(df.columns), path
    )
    return df


def load_primary_dataset() -> pd.DataFrame:
    """Convenience wrapper: load the primary dataset from its default path."""
    return load_tickets(PRIMARY_DATASET_PATH)
