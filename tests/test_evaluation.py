"""Unit tests for src.evaluation."""
from src.evaluation import hit_at_k, precision_at_k


def test_hit_at_k_true_when_present():
    assert hit_at_k(["bug", "performance", "billing_problem"], "performance", 3) is True


def test_hit_at_k_false_when_absent():
    assert hit_at_k(["bug", "billing_problem"], "performance", 2) is False


def test_hit_at_k_respects_k_boundary():
    assert hit_at_k(["bug", "billing_problem", "performance"], "performance", 2) is False
    assert hit_at_k(["bug", "billing_problem", "performance"], "performance", 3) is True


def test_hit_at_k_none_when_no_expected_category():
    assert hit_at_k(["bug", "performance"], None, 3) is None


def test_precision_at_k_all_relevant():
    assert precision_at_k(["performance", "performance"], "performance", 2) == 1.0


def test_precision_at_k_none_relevant():
    assert precision_at_k(["bug", "billing_problem"], "performance", 2) == 0.0


def test_precision_at_k_partial():
    assert precision_at_k(["performance", "bug"], "performance", 2) == 0.5


def test_precision_at_k_empty_results():
    assert precision_at_k([], "performance", 5) == 0.0


def test_precision_at_k_none_when_no_expected_category():
    assert precision_at_k(["bug"], None, 3) is None
