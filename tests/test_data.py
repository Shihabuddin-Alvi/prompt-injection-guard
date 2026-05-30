"""Data pipeline sanity checks."""

import duckdb
import pytest

DB_PATH = "data/unified.duckdb"


@pytest.fixture
def con():
    return duckdb.connect(DB_PATH)


def test_unified_table_exists(con):
    tables = con.execute("SHOW TABLES").fetchdf()["name"].tolist()
    assert "unified" in tables


def test_unified_schema(con):
    cols = con.execute("DESCRIBE unified").fetchdf()["column_name"].tolist()
    for col in ["id", "input_text", "label", "attack_category", "source", "split"]:
        assert col in cols


def test_no_null_input_text(con):
    nulls = con.execute(
        "SELECT COUNT(*) FROM unified WHERE input_text IS NULL OR input_text = ''"
    ).fetchone()[0]
    assert nulls == 0


def test_labels_are_binary(con):
    bad = con.execute(
        "SELECT COUNT(*) FROM unified WHERE label NOT IN (0, 1)"
    ).fetchone()[0]
    assert bad == 0


def test_no_duplicate_ids(con):
    total = con.execute("SELECT COUNT(*) FROM unified").fetchone()[0]
    unique = con.execute("SELECT COUNT(DISTINCT id) FROM unified").fetchone()[0]
    assert total == unique


def test_minimum_row_count(con):
    total = con.execute("SELECT COUNT(*) FROM unified").fetchone()[0]
    assert total >= 10000


def test_split_tables_exist(con):
    tables = con.execute("SHOW TABLES").fetchdf()["name"].tolist()
    for t in ["split_train", "split_val", "split_test"]:
        assert t in tables


def test_label_distribution_not_degenerate(con):
    for label in [0, 1]:
        count = con.execute(
            f"SELECT COUNT(*) FROM unified WHERE label = {label}"
        ).fetchone()[0]
        assert count > 100, f"Label {label} has too few examples: {count}"


def test_no_conflicting_labels(con):
    conflicts = con.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT input_text
            FROM unified
            GROUP BY input_text
            HAVING COUNT(DISTINCT label) > 1
        )
    """
    ).fetchone()[0]
    assert conflicts == 0
