"""Tests del módulo src.data.load."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.load import TARGET, load_raw, validate


def test_load_raw_returns_dataframe():
    df = load_raw()
    assert isinstance(df, pd.DataFrame)
    assert TARGET in df.columns
    assert len(df) >= 500


def test_validate_accepts_clean_dataset():
    df = load_raw()
    validate(df)  # No debe lanzar


def test_validate_rejects_missing_target():
    df = load_raw().drop(columns=[TARGET])
    with pytest.raises(AssertionError):
        validate(df)
