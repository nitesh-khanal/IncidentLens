"""Unit tests for src.preprocessing."""
import pandas as pd

from src.preprocessing import clean_tickets


def make_df(rows):
    return pd.DataFrame(rows)


def test_strips_whitespace():
    df = make_df([{
        "ticket_id": " TCKT_1 ", "created_at": "2024-01-01T00:00:00",
        "issue_type": " Bug ", "priority": "low", "status": "resolved",
        "initial_message": "  hello  ", "resolution_summary": "fixed",
        "region": "EU",
    }])
    cleaned = clean_tickets(df)
    assert cleaned.loc[0, "ticket_id"] == "TCKT_1"
    assert cleaned.loc[0, "initial_message"] == "hello"


def test_normalizes_categories_lowercase():
    df = make_df([{
        "ticket_id": "TCKT_1", "created_at": "2024-01-01T00:00:00",
        "issue_type": "Bug", "priority": "LOW", "status": "Resolved",
        "initial_message": "msg", "resolution_summary": "fixed",
        "region": "EU",
    }])
    cleaned = clean_tickets(df)
    assert cleaned.loc[0, "issue_type"] == "bug"
    assert cleaned.loc[0, "priority"] == "low"
    assert cleaned.loc[0, "status"] == "resolved"


def test_parses_timestamps_and_flags_invalid():
    df = make_df([
        {"ticket_id": "TCKT_1", "created_at": "2024-01-01T00:00:00",
         "issue_type": "bug", "priority": "low", "status": "resolved",
         "initial_message": "msg", "resolution_summary": "fixed", "region": "EU"},
        {"ticket_id": "TCKT_2", "created_at": "not-a-date",
         "issue_type": "bug", "priority": "low", "status": "resolved",
         "initial_message": "msg", "resolution_summary": "fixed", "region": "EU"},
    ])
    cleaned = clean_tickets(df)
    assert pd.notna(cleaned.loc[0, "created_at"])
    assert pd.isna(cleaned.loc[1, "created_at"])


def test_missing_region_filled_unknown():
    df = make_df([{
        "ticket_id": "TCKT_1", "created_at": "2024-01-01T00:00:00",
        "issue_type": "bug", "priority": "low", "status": "resolved",
        "initial_message": "msg", "resolution_summary": "fixed",
        "region": None,
    }])
    cleaned = clean_tickets(df)
    assert cleaned.loc[0, "region"] == "unknown"


def test_missing_resolution_not_fabricated():
    df = make_df([{
        "ticket_id": "TCKT_1", "created_at": "2024-01-01T00:00:00",
        "issue_type": "bug", "priority": "low", "status": "in_progress",
        "initial_message": "msg", "resolution_summary": None,
        "region": "EU",
    }])
    cleaned = clean_tickets(df)
    assert cleaned.loc[0, "has_resolution"] == False
    assert pd.isna(cleaned.loc[0, "resolution_summary"])


def test_empty_description_flagged():
    df = make_df([{
        "ticket_id": "TCKT_1", "created_at": "2024-01-01T00:00:00",
        "issue_type": "bug", "priority": "low", "status": "resolved",
        "initial_message": "   ", "resolution_summary": "fixed", "region": "EU",
    }])
    cleaned = clean_tickets(df)
    assert cleaned.loc[0, "has_valid_message"] == False


def test_duplicate_ticket_ids_removed():
    df = make_df([
        {"ticket_id": "TCKT_1", "created_at": "2024-01-01T00:00:00",
         "issue_type": "bug", "priority": "low", "status": "resolved",
         "initial_message": "msg", "resolution_summary": "fixed", "region": "EU"},
        {"ticket_id": "TCKT_1", "created_at": "2024-01-02T00:00:00",
         "issue_type": "bug", "priority": "low", "status": "resolved",
         "initial_message": "msg2", "resolution_summary": "fixed2", "region": "EU"},
    ])
    cleaned = clean_tickets(df)
    assert len(cleaned) == 1
