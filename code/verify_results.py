"""Recompute the included survey's descriptive outputs without downloading data."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_stats import ordinal_spearman
from run_all import build_practical, dist_table


def verify(repo_root):
    root = Path(repo_root)
    data = build_practical(root / "data/practical_data_survey_2026.parquet")
    columns = {
        "role": "Role", "org_size": "Organization size", "industry": "Industry",
        "region": "Region", "ai_usage_frequency": "AI usage frequency", "ai_adoption": "AI adoption maturity",
    }
    expected = pd.concat([dist_table(data[column].astype(str), label) for column, label in columns.items()], ignore_index=True)
    recorded = pd.read_csv(root / "outputs/table1_practical_sample_composition.csv")
    sort = lambda frame: frame.sort_values(["variable", "category"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(sort(expected), sort(recorded), check_dtype=False, atol=1e-10, rtol=1e-10)
    rho, pvalue = ordinal_spearman(data["ai_usage_frequency"], data["ai_adoption"])
    correlation = pd.read_csv(root / "outputs/spearman_usage_vs_maturity.csv").iloc[0]
    np.testing.assert_allclose([rho, pvalue], correlation[["spearman_rho", "p_value"]], rtol=1e-10, atol=0)
    return {"verified": True, "respondents": len(data), "composition_rows": len(expected),
            "daily_or_more": int(data["daily_plus"].sum()), "spearman_rho": rho, "p_value": pvalue}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(verify(args.repo_root), indent=2))
