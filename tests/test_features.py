"""Tests del módulo src.features.build."""

from __future__ import annotations

from src.data.load import load_raw
from src.features.build import build_features, make_reference_and_current


def test_build_features_preserves_rows():
    df = load_raw()
    out = build_features(df)
    assert len(out) == len(df)


def test_reference_current_are_disjoint():
    df = build_features(load_raw())
    ref, cur = make_reference_and_current(df)
    # Sin solapamiento de tamaños y mismas columnas
    assert len(ref) + len(cur) == len(df)
    assert list(ref.columns) == list(cur.columns)


def test_drift_is_actually_injected():
    df = build_features(load_raw())
    ref, cur = make_reference_and_current(df)
    # La feature 'mean radius' debe tener media mayor en current.
    assert cur["mean radius"].mean() > ref["mean radius"].mean()
    # Y la diferencia debe ser claramente perceptible (>5%).
    rel = (cur["mean radius"].mean() - ref["mean radius"].mean()) / ref["mean radius"].mean()
    assert rel > 0.05
