"""Small statistical operations used by the survey analysis."""

import pandas as pd
from scipy.stats import spearmanr


def ordinal_spearman(left, right):
    """Return Spearman rho/p for aligned, observed ordinal response pairs.

    Both inputs must be ordered categorical Series. Missing categories are
    removed before encoding: pandas uses -1 for missing values, which must not
    be interpreted as a response below the lowest category.
    """
    for values in (left, right):
        if not isinstance(values.dtype, pd.CategoricalDtype) or not values.cat.ordered:
            raise ValueError("ordinal_spearman requires ordered categorical Series")
    paired = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
    if len(paired) < 2 or any(paired[column].nunique() < 2 for column in paired):
        return float("nan"), float("nan")
    result = spearmanr(paired["left"].cat.codes, paired["right"].cat.codes)
    return result.statistic, result.pvalue
