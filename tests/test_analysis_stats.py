import math

import pandas as pd
import pytest
from scipy.stats import spearmanr

from analysis_stats import ordinal_spearman


def ordinal(values, index=None, ordered=True):
    return pd.Series(pd.Categorical(values, categories=["low", "mid", "high"], ordered=ordered), index=index)


def test_missing_responses_are_not_lowest_rank():
    left = ordinal(["low", "mid", "high", None, "high"])
    right = ordinal(["high", "low", "mid", "high", None])
    expected = spearmanr([0, 1, 2], [2, 0, 1])
    assert ordinal_spearman(left, right) == pytest.approx(expected)
    assert ordinal_spearman(left, right)[0] != pytest.approx(spearmanr(left.cat.codes, right.cat.codes).statistic)


def test_complete_responses_keep_existing_results():
    left = ordinal(["low", "mid", "high", "mid"])
    right = ordinal(["mid", "high", "low", "mid"])
    assert ordinal_spearman(left, right) == pytest.approx(spearmanr(left.cat.codes, right.cat.codes))


def test_pairs_align_by_respondent_index():
    left = ordinal(["low", "mid", "high"], index=["a", "b", "c"])
    right = ordinal(["high", "low", "mid"], index=["c", "a", "b"])
    assert ordinal_spearman(left, right)[0] == pytest.approx(1.0)


@pytest.mark.parametrize("left,right", [
    ([None, None], ["low", "high"]),
    (["low", None], ["high", "low"]),
    (["low", "low"], ["low", "high"]),
    ([], []),
])
def test_insufficient_or_constant_responses_have_undefined_correlation(left, right):
    assert all(math.isnan(value) for value in ordinal_spearman(ordinal(left), ordinal(right)))


def test_unordered_categories_are_rejected():
    with pytest.raises(ValueError, match="ordered categorical"):
        ordinal_spearman(ordinal(["low"], ordered=False), ordinal(["high"]))
