from pathlib import Path
import shutil

import pandas as pd
import pytest

from verify_results import verify

ROOT = Path(__file__).resolve().parents[1]


def test_included_data_reproduces_descriptive_outputs():
    result = verify(ROOT)
    assert result["verified"]
    assert result["respondents"] == 1101
    assert result["daily_or_more"] == 905


def test_changed_recorded_result_is_detected(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "outputs").mkdir()
    shutil.copy(ROOT / "data/practical_data_survey_2026.parquet", tmp_path / "data")
    for filename in ("table1_practical_sample_composition.csv", "spearman_usage_vs_maturity.csv"):
        shutil.copy(ROOT / "outputs" / filename, tmp_path / "outputs")
    path = tmp_path / "outputs/spearman_usage_vs_maturity.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "spearman_rho"] = 0.99
    frame.to_csv(path, index=False)
    with pytest.raises(AssertionError):
        verify(tmp_path)
