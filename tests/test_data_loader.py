"""Unit tests for src.data_loader."""
import pandas as pd
import pytest

from src.data_loader import DataLoadError, load_tickets, REQUIRED_COLUMNS

VALID_HEADER = ",".join(REQUIRED_COLUMNS)
VALID_ROW = (
    "TCKT_000001,2024-01-31T05:14:27,account_access,low,resolved,"
    "Cannot log in.,Reset credentials."
)


def write(tmp_path, name, content, encoding="utf-8"):
    p = tmp_path / name
    p.write_bytes(content.encode(encoding) if isinstance(content, str) else content)
    return p


def test_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises(DataLoadError, match="not found"):
        load_tickets(missing)


def test_path_is_directory(tmp_path):
    with pytest.raises(DataLoadError, match="not a file"):
        load_tickets(tmp_path)


def test_empty_file(tmp_path):
    path = write(tmp_path, "empty.csv", "")
    with pytest.raises(DataLoadError, match="empty"):
        load_tickets(path)


def test_header_only_no_rows(tmp_path):
    path = write(tmp_path, "header_only.csv", VALID_HEADER + "\n")
    with pytest.raises(DataLoadError, match="zero rows"):
        load_tickets(path)


def test_missing_required_columns(tmp_path):
    path = write(
        tmp_path,
        "missing_cols.csv",
        "ticket_id,created_at\nTCKT_1,2024-01-01\n",
    )
    with pytest.raises(DataLoadError, match="missing required columns"):
        load_tickets(path)


def test_malformed_csv_unterminated_quote(tmp_path):
    # An unterminated quoted field is a genuine tokenization failure —
    # the C parser cannot recover from an unclosed quote before EOF.
    content = (
        VALID_HEADER + "\n"
        'TCKT_000002,2024-01-31T05:14:27,bug,low,resolved,'
        '"This message quote is never closed,fixed it\n'
    )
    path = write(tmp_path, "malformed.csv", content)
    with pytest.raises(DataLoadError):
        load_tickets(path)


def test_bad_encoding(tmp_path):
    path = tmp_path / "bad_encoding.csv"
    path.write_bytes((VALID_HEADER + "\n").encode("utf-8") + b"\xff\xfe\x00\x01")
    with pytest.raises(DataLoadError):
        load_tickets(path)


def test_valid_load(tmp_path):
    content = VALID_HEADER + "\n" + VALID_ROW + "\n"
    path = write(tmp_path, "valid.csv", content)
    df = load_tickets(path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert set(REQUIRED_COLUMNS).issubset(df.columns)
