import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]

GRID_CASES = [
    ("e1_nosd.csv", ["20261110_001823"]),
    ("e2_specbench.csv", [
        "20261110_005501",
        "20261110_064430",
        "20261110_171854",
    ]),
    ("e2_ctx512.csv", [
        "20261110_021734",
        "20261110_001823",
    ]),
    ("e2_ctx2k.csv", [
        "20261110_042842",
        "20261110_001823",
    ]),
    ("e2_dynamic_controls.csv", ["20261110_180529"]),
]


@pytest.mark.parametrize("filename,runs", GRID_CASES)
def test_reproduce_grid(filename, runs, tmp_path):
    output = tmp_path / filename

    raw_files = [
        ROOT / "results" / "raw" / "e2" / run / "rows.jsonl"
        for run in runs
    ]

    command = [
        sys.executable,
        str(ROOT / "scripts" / "summarize_grid.py"),
        *[str(path) for path in raw_files],
        "--out",
        str(output),
    ]

    subprocess.run(command, check=True, cwd=ROOT, capture_output=True, text=True)

    actual = pd.read_csv(output)
    expected = pd.read_csv(ROOT / "results" / "summary" / filename)

    keys = ["setting", "context", "concurrency"]

    assert set(actual.columns) == set(expected.columns)

    actual = actual.set_index(keys).sort_index()
    expected = expected.set_index(keys).sort_index()

    assert actual.index.is_unique
    assert expected.index.is_unique
    pd.testing.assert_index_equal(actual.index, expected.index)

    for column in expected.columns:
        if pd.api.types.is_numeric_dtype(expected[column]):
            assert np.isclose(
                actual[column].to_numpy(),
                expected[column].to_numpy(),
                rtol=1e-9,
                equal_nan=True,
            ).all(), f"Mismatch in {filename}: {column}"
        else:
            pd.testing.assert_series_equal(
                actual[column],
                expected[column],
                check_names=False,
            )
