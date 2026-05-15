#!/usr/bin/env python3
"""Reproduce analyses and figures for the revised CADE 2026 paper.

This script:
  1) Loads Practical Data survey parquet.
  2) Loads Stack Overflow 2025 survey CSV (selected columns only).
  3) Produces CSV outputs (sample composition and key regression terms).
  4) Generates all figures used in the paper (PNG + PDF).

Example:
  python code/run_all.py \
    --practical ../practical_data_survey_2026.parquet \
    --stackoverflow ../stackoverflow_2025_survey.csv \
    --outdir ..

Notes:
  - The Practical Data file is parquet and requires pyarrow.
  - The Stack Overflow CSV is large; we read only a small set of columns.
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.stats import spearmanr
import statsmodels.formula.api as smf


ADOPTION_ORDER = [
    "No meaningful adoption yet",
    "Experimenting",
    "Using AI for tactical tasks",
    "Building internal AI platforms",
    "AI embedded in most workflows",
]
FREQ_ORDER = ["Never", "Rarely", "Weekly", "Daily", "Multiple times per day"]


def clean_modeling(x: str) -> str:
    """Normalize free-text modeling approaches into a small taxonomy."""
    if pd.isna(x):
        return "Other"
    s = str(x).strip().lower()

    # Mixed/hybrid first so we don't mislabel "Kimball + ..." as pure Kimball.
    if s.startswith("mixed") or "mix of" in s or "depends" in s:
        return "Mixed"

    if "ad-hoc" in s or "ad hoc" in s or "tables added" in s:
        return "Ad-hoc"

    if "one big table" in s or re.search(r"\bobt\b", s) or "1 table per report" in s:
        if any(k in s for k in ["kimball", "data vault", "datavault", "semantic", "canonical"]):
            return "Mixed"
        return "One Big Table"

    if "data vault" in s or "datavault" in s or re.search(r"\bdv2?(?:\.0)?\b", s):
        if any(k in s for k in ["kimball", "obt"]):
            return "Mixed"
        return "Data Vault"

    if "kimball" in s or "dimensional" in s:
        if any(k in s for k in ["mixed", "mix", "obt", "data vault", "datavault"]):
            return "Mixed"
        return "Kimball"

    if "event" in s:
        if any(k in s for k in ["canonical", "kimball"]):
            return "Mixed"
        return "Event-driven"

    if "canonical" in s or "semantic" in s or "knowledge graph" in s:
        return "Canonical/Semantic"

    return "Other"


def build_practical(practical_path: str) -> pd.DataFrame:
    df = pd.read_parquet(practical_path)

    df["ai_adoption"] = pd.Categorical(df["ai_adoption"], categories=ADOPTION_ORDER, ordered=True)
    df["ai_usage_frequency"] = pd.Categorical(df["ai_usage_frequency"], categories=FREQ_ORDER, ordered=True)

    df["daily_plus"] = df["ai_usage_frequency"].isin(["Daily", "Multiple times per day"])
    df["org_beyond_experimentation"] = df["ai_adoption"].isin(
        ["Building internal AI platforms", "AI embedded in most workflows"]
    )
    df["shadow_ai"] = df["daily_plus"] & df["ai_adoption"].isin(["No meaningful adoption yet", "Experimenting"])

    df["modeling_cat"] = df["modeling_approach"].apply(clean_modeling)

    # Modeling pain points
    pain_tokens = {
        "pain_ownership": "Lack of clear ownership",
        "pain_speed": "Pressure to “move fast”",
        "pain_maintain": "Hard to maintain over time",
        "pain_tools": "Tools don’t support good modeling",
        "pain_ai_schema": "AI tools produce inconsistent schemas",
        "pain_none": "None / modeling is going well",
    }
    for col, tok in pain_tokens.items():
        df[col] = df["modeling_pain_points"].fillna("").str.contains(re.escape(tok))

    # Bottleneck type: strict mapping; any free-text goes to Other
    main_bottlenecks = {
        "Legacy / technical debt": "Technical",
        "Data quality": "Technical",
        "Compute costs": "Technical",
        "Tool complexity": "Technical",
        "Lack of leadership direction": "Organizational",
        "Poor requirements / upstream issues": "Organizational",
        "Talent / hiring": "Organizational",
        "Other": "Other",
    }
    options = set(main_bottlenecks.keys())
    df["bottleneck_cat"] = df["biggest_bottleneck"].apply(lambda x: x if x in options else "Other")
    df["bottleneck_type"] = df["bottleneck_cat"].map(main_bottlenecks).fillna("Other")

    return df


def build_stackoverflow(so_path: str) -> pd.DataFrame:
    usecols = [
        "ResponseId",
        "MainBranch",
        "DevType",
        "OrgSize",
        "Industry",
        "AISelect",
        "AIAcc",
        "AISent",
        "AIAgents",
        "AIAgentChange",
        "AIFrustration",
        "AIComplex",
    ]
    df = pd.read_csv(so_path, usecols=usecols, low_memory=False)

    # Convenience buckets
    df["usage_bucket"] = df["AISelect"].map(
        {
            "Yes, I use AI tools daily": "Daily",
            "Yes, I use AI tools weekly": "Weekly",
            "Yes, I use AI tools monthly or infrequently": "Monthly/infrequent",
            "No, but I plan to soon": "Plan to soon",
            "No, and I don't plan to": "Don't plan",
        }
    )

    df["trust_class"] = df["AIAcc"].map(
        {
            "Highly trust": "Trust",
            "Somewhat trust": "Trust",
            "Neither trust nor distrust": "Neutral",
            "Somewhat distrust": "Distrust",
            "Highly distrust": "Distrust",
        }
    )

    def agent_bucket(x: str) -> str:
        if pd.isna(x):
            return "Missing"
        if str(x).startswith("Yes, I use AI agents at work daily"):
            return "Agents daily"
        if str(x).startswith("Yes, I use AI agents at work weekly"):
            return "Agents weekly"
        if str(x).startswith("Yes, I use AI agents at work monthly"):
            return "Agents monthly+"
        if str(x).startswith("No, I use AI exclusively in copilot"):
            return "Copilot only"
        if str(x).startswith("No, but I plan"):
            return "Plan to use agents"
        if str(x).startswith("No, and I don't plan"):
            return "Don't plan"
        return "Other"

    df["agent_bucket"] = df["AIAgents"].apply(agent_bucket)

    # Frustration flags (multi-select separated by semicolons)
    frust = df["AIFrustration"].fillna("")
    df["frust_almost_right"] = frust.str.contains(re.escape("AI solutions that are almost right, but not quite"))
    df["frust_debug_time"] = frust.str.contains(re.escape("Debugging AI-generated code is more time-consuming"))
    df["frust_hard_understand"] = frust.str.contains(re.escape("It’s hard to understand how or why the code works"))

    return df


def dist_table(series: pd.Series, variable: str) -> pd.DataFrame:
    c = series.value_counts(dropna=False)
    df = pd.DataFrame({"variable": variable, "category": c.index.astype(str), "n": c.values})
    df["pct"] = (df["n"] / df["n"].sum() * 100).round(1)
    return df


def ensure_dirs(outdir: str) -> tuple[str, str]:
    analysis_dir = os.path.join(outdir, "analysis_outputs")
    figures_dir = os.path.join(outdir, "figures")
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    return analysis_dir, figures_dir


def make_figures(practical: pd.DataFrame, so: pd.DataFrame, figures_dir: str) -> None:
    # Figure 1: Practical Data usage + adoption distributions
    freq_counts = practical["ai_usage_frequency"].value_counts().reindex(FREQ_ORDER)
    freq_pct = freq_counts / freq_counts.sum() * 100

    adopt_counts = practical["ai_adoption"].value_counts().reindex(ADOPTION_ORDER)
    adopt_pct = adopt_counts / adopt_counts.sum() * 100

    plt.close("all")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), dpi=200)

    ax = axes[0]
    y = list(range(len(freq_counts)))
    ax.barh(y, freq_counts.values)
    ax.set_yticks(y, labels=freq_counts.index)
    ax.invert_yaxis()
    ax.set_xlabel("Respondents")
    ax.set_title("(a) AI tool usage frequency")
    for i, (v, p) in enumerate(zip(freq_counts.values, freq_pct.values)):
        ax.text(v + max(freq_counts.values) * 0.01, i, f"{p:.0f}%", va="center")

    ax = axes[1]
    y = list(range(len(adopt_counts)))
    ax.barh(y, adopt_counts.values)
    ax.set_yticks(y, labels=["No adoption", "Experimenting", "Tactical", "Building\nplatforms", "Embedded"])
    ax.invert_yaxis()
    ax.set_xlabel("Respondents")
    ax.set_title("(b) Organizational AI adoption maturity")
    for i, (v, p) in enumerate(zip(adopt_counts.values, adopt_pct.values)):
        ax.text(v + max(adopt_counts.values) * 0.01, i, f"{p:.0f}%", va="center")

    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "fig1_usage_and_adoption.png"), bbox_inches="tight")
    fig.savefig(os.path.join(figures_dir, "fig1_usage_and_adoption.pdf"), bbox_inches="tight")

    # Figure 2: Shadow AI segmentation
    daily = practical[practical["daily_plus"]].copy()
    nondaily = practical[~practical["daily_plus"]].copy()

    daily_dist = daily.groupby("ai_adoption", observed=True).size().reindex(ADOPTION_ORDER).fillna(0)
    daily_pct = daily_dist / daily_dist.sum() * 100

    nondaily_dist = nondaily.groupby("ai_adoption", observed=True).size().reindex(ADOPTION_ORDER).fillna(0)
    nondaily_pct = nondaily_dist / nondaily_dist.sum() * 100

    plt.close("all")
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=200)

    groups = ["Daily+ users", "Weekly or less"]
    data = np.vstack([daily_pct.values, nondaily_pct.values])

    left = np.zeros(len(groups))
    colors = ["#d73027", "#fc8d59", "#91bfdb", "#4575b4", "#313695"]
    labels = ["No adoption", "Experimenting", "Tactical", "Platforms", "Embedded"]

    for i in range(len(ADOPTION_ORDER)):
        ax.barh(groups, data[:, i], left=left, color=colors[i], label=labels[i])
        left += data[:, i]

    ax.set_xlim(0, 100)
    ax.set_xlabel("Share within group (%)")
    ax.set_title("Figure 2. Where the usage-adoption gap sits: adoption stage by individual AI usage")
    ax.xaxis.set_major_formatter(PercentFormatter())
    ax.legend(ncol=3, bbox_to_anchor=(0.5, -0.18), loc="upper center", frameon=False)

    shadow_share = float(daily_pct.loc["No meaningful adoption yet"] + daily_pct.loc["Experimenting"])
    ax.text(shadow_share / 2, 0, f"Shadow AI\n{shadow_share:.1f}%", va="center", ha="center", color="white", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "fig2_shadow_ai_segmentation.png"), bbox_inches="tight")
    fig.savefig(os.path.join(figures_dir, "fig2_shadow_ai_segmentation.pdf"), bbox_inches="tight")

    # Figure 3: Bottleneck inversion
    counts = practical.groupby(["ai_adoption", "bottleneck_type"], observed=True).size().reset_index(name="n")
    counts["pct"] = counts.groupby("ai_adoption")["n"].transform(lambda x: x / x.sum() * 100)
    pivot = counts.pivot(index="ai_adoption", columns="bottleneck_type", values="pct").fillna(0).reindex(ADOPTION_ORDER)

    plt.close("all")
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=200)

    x = np.arange(len(ADOPTION_ORDER))
    bottom = np.zeros(len(ADOPTION_ORDER))
    palette = {"Organizational": "#d7191c", "Technical": "#2c7bb6", "Other": "#bdbdbd"}

    for col in ["Organizational", "Technical", "Other"]:
        ax.bar(x, pivot[col].values, bottom=bottom, label=col, color=palette[col])
        bottom += pivot[col].values

    ax.set_xticks(x, labels=["No adopt", "Experim.", "Tactical", "Platforms", "Embedded"])
    ax.set_ylabel("Share of respondents (%)")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_title("Figure 3. Bottleneck inversion across maturity stages (Practical Data, n=1,101)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)

    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "fig3_bottleneck_inversion.png"), bbox_inches="tight")
    fig.savefig(os.path.join(figures_dir, "fig3_bottleneck_inversion.pdf"), bbox_inches="tight")

    # Figure 4: Modeling mix by maturity
    tmp = practical.groupby(["ai_adoption", "modeling_cat"], observed=True).size().reset_index(name="n")
    tmp["pct"] = tmp.groupby("ai_adoption")["n"].transform(lambda x: x / x.sum() * 100)
    model_pivot = tmp.pivot(index="ai_adoption", columns="modeling_cat", values="pct").fillna(0).reindex(ADOPTION_ORDER)

    cats = ["Ad-hoc", "Kimball", "Mixed", "Canonical/Semantic", "Data Vault", "Event-driven", "One Big Table", "Other"]
    cats = [c for c in cats if c in model_pivot.columns]

    colors = {
        "Ad-hoc": "#d7191c",
        "Kimball": "#fdae61",
        "Mixed": "#2c7bb6",
        "Canonical/Semantic": "#abd9e9",
        "Data Vault": "#ffffbf",
        "Event-driven": "#a6d96a",
        "One Big Table": "#1a9641",
        "Other": "#bdbdbd",
    }

    plt.close("all")
    fig, ax = plt.subplots(figsize=(11, 5), dpi=200)
    x = np.arange(len(ADOPTION_ORDER))
    bottom = np.zeros(len(ADOPTION_ORDER))

    for c in cats:
        ax.bar(x, model_pivot[c].values, bottom=bottom, label=c, color=colors.get(c))
        bottom += model_pivot[c].values

    ax.set_xticks(x, labels=["No adopt", "Experim.", "Tactical", "Platforms", "Embedded"])
    ax.set_ylabel("Share (%)")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_title("Figure 4. Data modeling approach mix by AI adoption maturity (Practical Data, n=1,101)")
    ax.legend(ncol=4, bbox_to_anchor=(0.5, -0.18), loc="upper center", frameon=False)

    # annotate ad-hoc endpoints
    if "Ad-hoc" in model_pivot.columns:
        ax.text(0, model_pivot.iloc[0]["Ad-hoc"] / 2, f"Ad-hoc\n{model_pivot.iloc[0]['Ad-hoc']:.1f}%", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
        ax.text(4, model_pivot.iloc[4]["Ad-hoc"] / 2, f"Ad-hoc\n{model_pivot.iloc[4]['Ad-hoc']:.1f}%", ha="center", va="center", color="white", fontsize=8, fontweight="bold")

    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "fig4_modeling_mix.png"), bbox_inches="tight")
    fig.savefig(os.path.join(figures_dir, "fig4_modeling_mix.pdf"), bbox_inches="tight")

    # Figure 5: Shadow AI predictors (AME forest)
    daily_plus = practical[practical["daily_plus"]].copy()
    daily_plus["shadow_ai_int"] = daily_plus["shadow_ai"].astype(int)

    daily_plus["modeling_cat"] = pd.Categorical(
        daily_plus["modeling_cat"],
        categories=["Mixed", "Ad-hoc", "Kimball", "Canonical/Semantic", "Data Vault", "Event-driven", "One Big Table", "Other"],
    )
    daily_plus["bottleneck_type"] = pd.Categorical(daily_plus["bottleneck_type"], categories=["Technical", "Organizational", "Other"])

    # Group roles to avoid quasi-complete separation from singleton categories
    role_map = {
        "Data Engineer": "Data Engineer",
        "Manager / Director / VP": "Manager/Director/VP",
        "Analytics Engineer": "Analytics Engineer",
        "Data Architect": "Data Architect",
        "Software Engineer working on data": "Software Engineer",
        "Platform Engineer": "Platform Engineer",
        "ML Engineer / MLOps": "ML/AI Engineer",
        "AI Engineer": "ML/AI Engineer",
    }
    daily_plus["role_grouped"] = daily_plus["role"].map(role_map).fillna("Other")
    daily_plus["role_grouped"] = pd.Categorical(
        daily_plus["role_grouped"],
        categories=["Data Engineer", "Manager/Director/VP", "Analytics Engineer", "Data Architect",
                     "Software Engineer", "Platform Engineer", "ML/AI Engineer", "Other"],
    )

    for c in ["org_size", "industry", "team_growth_2026"]:
        daily_plus[c] = daily_plus[c].astype("category")

    formula = (
        "shadow_ai_int ~ C(modeling_cat, Treatment(reference='Mixed')) "
        "+ pain_ownership + pain_speed + pain_maintain + pain_tools "
        "+ C(bottleneck_type, Treatment(reference='Technical')) "
        "+ C(org_size) + C(industry) + C(role_grouped, Treatment(reference='Data Engineer')) + C(team_growth_2026)"
    )

    m = smf.logit(formula, data=daily_plus).fit(method="newton", maxiter=500, disp=False)
    me = m.get_margeff(at="overall", method="dydx").summary_frame()

    terms = [
        ("pain_ownership[T.True]", "Pain: unclear ownership (yes vs no)"),
        ("C(modeling_cat, Treatment(reference='Mixed'))[T.Ad-hoc]", "Modeling: ad-hoc (vs mixed)"),
        ("C(org_size)[T.10,000+]", "Org size: 10,000+ (vs 1,000-10,000)"),
        ("C(bottleneck_type, Treatment(reference='Technical'))[T.Organizational]", "Biggest bottleneck: organizational (vs technical)"),
    ]

    rows = []
    for key, label in terms:
        r = me.loc[key]
        rows.append((label, r["dy/dx"] * 100, r["Conf. Int. Low"] * 100, r["Cont. Int. Hi."] * 100, r["Pr(>|z|)"]))

    plot_df = pd.DataFrame(rows, columns=["term", "pp", "low", "high", "p"]).iloc[::-1]

    plt.close("all")
    fig, ax = plt.subplots(figsize=(5.5, 3.0), dpi=300)
    y = np.arange(len(plot_df))
    ax.hlines(y, plot_df["low"], plot_df["high"], color="black", linewidth=1.5)
    ax.plot(plot_df["pp"], y, "ko", markersize=5)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)

    short_labels = [t.replace(" (yes vs no)", "").replace(" (vs mixed)", "").replace(" (vs 1,000-10,000)", "").replace(" (vs technical)", "") for t in plot_df["term"]]
    ax.set_yticks(y, labels=short_labels, fontsize=7)
    ax.set_xlabel("AME on P(Shadow AI), pp", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)

    # Place AME + significance as compact text to the right of the plot area
    for i, (pp, p, lo, hi) in enumerate(zip(plot_df["pp"], plot_df["p"], plot_df["low"], plot_df["high"])):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        ax.annotate(f"{pp:+.1f}{sig}", xy=(hi + 1.0 if pp >= 0 else lo - 1.0, i),
                    fontsize=6.5, va="center", ha="left" if pp >= 0 else "right")

    ax.set_xlim(ax.get_xlim()[0] - 3, ax.get_xlim()[1] + 5)
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "fig5_shadow_ai_predictors.png"), bbox_inches="tight")
    fig.savefig(os.path.join(figures_dir, "fig5_shadow_ai_predictors.pdf"), bbox_inches="tight")

    # Figure 6: Stack Overflow trust mechanism (agents + debugging)
    so_trust = so[~so["AIAcc"].isna()].copy()
    so_trust = so_trust[so_trust["usage_bucket"] == "Daily"].copy()

    agent_order = ["Agents daily", "Agents weekly", "Agents monthly+", "Copilot only", "Plan to use agents", "Don't plan"]
    tab = so_trust[so_trust["agent_bucket"].isin(agent_order)].groupby(["agent_bucket", "trust_class"]).size().reset_index(name="n")
    tab["pct"] = tab.groupby("agent_bucket")["n"].transform(lambda x: x / x.sum() * 100)
    pivot = tab.pivot(index="agent_bucket", columns="trust_class", values="pct").fillna(0).reindex(agent_order)

    plt.close("all")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=200)

    ax = axes[0]
    x = np.arange(len(agent_order))
    bottom = np.zeros(len(agent_order))
    palette = {"Trust": "#2c7bb6", "Neutral": "#bdbdbd", "Distrust": "#d7191c"}
    for cls in ["Distrust", "Neutral", "Trust"]:
        ax.bar(x, pivot[cls].values, bottom=bottom, label=cls, color=palette[cls])
        bottom += pivot[cls].values

    ax.set_xticks(x, labels=["Agents\nDaily", "Agents\nWeekly", "Agents\nMonthly+", "Copilot\nonly", "Plan\nsoon", "Don't\nplan"])
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_ylabel("Share (%)")
    ax.set_title("(a) Trust distribution among daily AI tool users\nby AI-agent usage (Stack Overflow 2025)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)

    ax = axes[1]
    sub = so_trust[so_trust["trust_class"] != "Neutral"].copy()
    sub["distrust"] = sub["trust_class"].eq("Distrust")
    rates = (sub.groupby("frust_debug_time")["distrust"].mean() * 100).to_dict()

    labels = ["No debugging overhead flag", "Debugging overhead flag"]
    vals = [rates.get(False, np.nan), rates.get(True, np.nan)]
    ax.bar(labels, vals)
    ax.set_ylabel("Distrust rate (%)\n(excluding neutral)")
    ax.set_ylim(0, 100)
    ax.set_title("(b) Debugging overhead and distrust\namong daily AI tool users")
    for i, v in enumerate(vals):
        ax.text(i, v + 2, f"{v:.1f}%", ha="center")

    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "fig6_so_trust_mechanism.png"), bbox_inches="tight")
    fig.savefig(os.path.join(figures_dir, "fig6_so_trust_mechanism.pdf"), bbox_inches="tight")

    # Figure 7: dual gaps
    daily_plus_pct = float(practical["daily_plus"].mean() * 100)
    beyond_exp_pct = float(practical["org_beyond_experimentation"].mean() * 100)

    so_nonnull = so[~so["AISelect"].isna()]
    weekly_plus = float(so_nonnull["AISelect"].isin(["Yes, I use AI tools daily", "Yes, I use AI tools weekly"]).mean() * 100)

    so_acc = so[~so["AIAcc"].isna()]
    trust_pct = float(so_acc["AIAcc"].isin(["Highly trust", "Somewhat trust"]).mean() * 100)

    plt.close("all")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=200)

    ax = axes[0]
    vals = [daily_plus_pct, beyond_exp_pct]
    labels = ["Daily+ AI use", "Org: platforms\nor embedded"]
    ax.bar(labels, vals, color=["#2c7bb6", "#d7191c"])
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_title("(a) Practical Data survey (n=1,101)")
    for i, v in enumerate(vals):
        ax.text(i, v + 2, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")

    ax = axes[1]
    vals = [weekly_plus, trust_pct]
    labels = ["Weekly+ AI use", "Trust AI\naccuracy"]
    ax.bar(labels, vals, color=["#2c7bb6", "#d7191c"])
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PercentFormatter())
    n_aiselect = int(so["AISelect"].notna().sum())
    n_aiacc = int(so["AIAcc"].notna().sum())
    ax.set_title(f"(b) Stack Overflow survey\n(usage n={n_aiselect:,}; trust n={n_aiacc:,})")
    for i, v in enumerate(vals):
        ax.text(i, v + 2, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")

    fig.suptitle("Figure 7. Two adoption gaps: usage vs maturity (Practical Data) and usage vs trust (Stack Overflow)", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "fig7_dual_survey_gaps.png"), bbox_inches="tight")
    fig.savefig(os.path.join(figures_dir, "fig7_dual_survey_gaps.pdf"), bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--practical", required=True, help="Path to practical_data_survey_2026.parquet")
    parser.add_argument("--stackoverflow", required=True, help="Path to stackoverflow_2025_survey.csv")
    parser.add_argument("--outdir", required=True, help="Output directory (project root)")
    args = parser.parse_args()

    analysis_dir, figures_dir = ensure_dirs(args.outdir)

    practical = build_practical(args.practical)
    so = build_stackoverflow(args.stackoverflow)

    # Sample composition export (Practical Data)
    t1 = pd.concat(
        [
            dist_table(practical["role"], "Role"),
            dist_table(practical["org_size"], "Organization size"),
            dist_table(practical["industry"], "Industry"),
            dist_table(practical["region"], "Region"),
            dist_table(practical["ai_usage_frequency"].astype(str), "AI usage frequency"),
            dist_table(practical["ai_adoption"].astype(str), "AI adoption maturity"),
        ],
        ignore_index=True,
    )
    t1.to_csv(os.path.join(analysis_dir, "table1_practical_sample_composition.csv"), index=False)

    # Practical Data Shadow AI model (key terms only)
    daily_plus = practical[practical["daily_plus"]].copy()
    daily_plus["shadow_ai_int"] = daily_plus["shadow_ai"].astype(int)
    daily_plus["modeling_cat"] = pd.Categorical(
        daily_plus["modeling_cat"],
        categories=["Mixed", "Ad-hoc", "Kimball", "Canonical/Semantic", "Data Vault", "Event-driven", "One Big Table", "Other"],
    )
    daily_plus["bottleneck_type"] = pd.Categorical(daily_plus["bottleneck_type"], categories=["Technical", "Organizational", "Other"])
    # Group roles to avoid quasi-complete separation
    role_map = {
        "Data Engineer": "Data Engineer",
        "Manager / Director / VP": "Manager/Director/VP",
        "Analytics Engineer": "Analytics Engineer",
        "Data Architect": "Data Architect",
        "Software Engineer working on data": "Software Engineer",
        "Platform Engineer": "Platform Engineer",
        "ML Engineer / MLOps": "ML/AI Engineer",
        "AI Engineer": "ML/AI Engineer",
    }
    daily_plus["role_grouped"] = daily_plus["role"].map(role_map).fillna("Other")
    daily_plus["role_grouped"] = pd.Categorical(
        daily_plus["role_grouped"],
        categories=["Data Engineer", "Manager/Director/VP", "Analytics Engineer", "Data Architect",
                     "Software Engineer", "Platform Engineer", "ML/AI Engineer", "Other"],
    )

    for c in ["org_size", "industry", "team_growth_2026"]:
        daily_plus[c] = daily_plus[c].astype("category")

    formula = (
        "shadow_ai_int ~ C(modeling_cat, Treatment(reference='Mixed')) "
        "+ pain_ownership + pain_speed + pain_maintain + pain_tools "
        "+ C(bottleneck_type, Treatment(reference='Technical')) "
        "+ C(org_size) + C(industry) + C(role_grouped, Treatment(reference='Data Engineer')) + C(team_growth_2026)"
    )
    m = smf.logit(formula, data=daily_plus).fit(method="newton", maxiter=500, disp=False)
    me = m.get_margeff(at="overall", method="dydx").summary_frame()

    params = m.params
    conf = m.conf_int()
    or_table = pd.DataFrame(
        {
            "odds_ratio": np.exp(params),
            "or_ci_low": np.exp(conf[0]),
            "or_ci_high": np.exp(conf[1]),
            "p_value": m.pvalues,
        }
    )

    key_terms = [
        "pain_ownership[T.True]",
        "C(modeling_cat, Treatment(reference='Mixed'))[T.Ad-hoc]",
        "C(org_size)[T.10,000+]",
        "C(bottleneck_type, Treatment(reference='Technical'))[T.Organizational]",
    ]

    out_tbl = (
        pd.DataFrame({"term": key_terms})
        .merge(or_table.loc[key_terms].reset_index().rename(columns={"index": "term"}), on="term", how="left")
        .merge(
            me.loc[key_terms][["dy/dx", "Conf. Int. Low", "Cont. Int. Hi.", "Pr(>|z|)"]]
            .reset_index()
            .rename(columns={"index": "term", "dy/dx": "ame", "Conf. Int. Low": "ame_ci_low", "Cont. Int. Hi.": "ame_ci_high", "Pr(>|z|)": "ame_p"}),
            on="term",
            how="left",
        )
    )
    out_tbl[["ame", "ame_ci_low", "ame_ci_high"]] = out_tbl[["ame", "ame_ci_low", "ame_ci_high"]] * 100
    out_tbl.to_csv(os.path.join(analysis_dir, "table2_shadow_ai_logit_key_predictors.csv"), index=False)

    # Stack Overflow daily distrust model (key terms)
    so_acc = so[~so["AIAcc"].isna()].copy()
    so_daily = so_acc[so_acc["usage_bucket"] == "Daily"].copy()
    so_daily = so_daily[so_daily["trust_class"] != "Neutral"].copy()
    so_daily["distrust"] = so_daily["trust_class"].eq("Distrust").astype(int)

    # Drop ambiguous Other/Missing agent categories to avoid Hessian singularity
    so_daily = so_daily[~so_daily["agent_bucket"].isin(["Other", "Missing"])].copy()

    so_daily["agent_bucket"] = pd.Categorical(
        so_daily["agent_bucket"],
        categories=["Agents daily", "Agents weekly", "Agents monthly+", "Copilot only", "Plan to use agents", "Don't plan"],
    )
    so_daily["OrgSize"] = so_daily["OrgSize"].fillna("Missing").astype("category")

    mso = smf.logit(
        "distrust ~ C(agent_bucket, Treatment(reference='Agents daily')) + frust_debug_time + frust_almost_right + frust_hard_understand + C(OrgSize)",
        data=so_daily,
    ).fit(method="bfgs", maxiter=500, disp=False)

    params = mso.params
    conf = mso.conf_int()
    so_or = pd.DataFrame(
        {
            "odds_ratio": np.exp(params),
            "or_ci_low": np.exp(conf[0]),
            "or_ci_high": np.exp(conf[1]),
            "p_value": mso.pvalues,
        }
    )

    so_key_terms = [
        "frust_debug_time[T.True]",
        "frust_almost_right[T.True]",
        "C(agent_bucket, Treatment(reference='Agents daily'))[T.Copilot only]",
        "C(agent_bucket, Treatment(reference='Agents daily'))[T.Don't plan]",
    ]
    so_or.loc[so_key_terms].reset_index().rename(columns={"index": "term"}).to_csv(
        os.path.join(analysis_dir, "table3_so_daily_distrust_logit_key_predictors.csv"),
        index=False,
    )

    # Descriptive: usage–maturity Spearman (Practical Data)
    rho, p = spearmanr(practical["ai_usage_frequency"].cat.codes, practical["ai_adoption"].cat.codes)
    pd.DataFrame({"spearman_rho": [rho], "p_value": [p]}).to_csv(os.path.join(analysis_dir, "spearman_usage_vs_maturity.csv"), index=False)

    # Generate all figures
    make_figures(practical, so, figures_dir)


if __name__ == "__main__":
    main()
