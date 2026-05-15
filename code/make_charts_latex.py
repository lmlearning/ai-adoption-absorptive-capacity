#!/usr/bin/env python3
"""Generate publication-quality figures for the CADE 2026 LaTeX paper.

Outputs PDF figures sized for the IET two-column template:
  - Single-column width: 8.8 cm (~3.46 in)
  - Double-column width: 18.0 cm (~7.09 in)

Usage:
  python make_charts_latex.py \
    --practical ../survey.parquet \
    --stackoverflow ../so2025/survey_results_public.csv \
    --outdir figures
"""

from __future__ import annotations
import argparse, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.stats import spearmanr
import statsmodels.formula.api as smf

# ── Layout constants ──
SINGLE_COL = 3.46  # inches
DOUBLE_COL = 7.09  # inches
DPI = 300

ADOPTION_ORDER = [
    "No meaningful adoption yet", "Experimenting",
    "Using AI for tactical tasks", "Building internal AI platforms",
    "AI embedded in most workflows",
]
FREQ_ORDER = ["Never", "Rarely", "Weekly", "Daily", "Multiple times per day"]

# ── Colours (print-safe, colour-blind friendly) ──
C_BLUE = "#2c7bb6"
C_RED  = "#d7191c"
C_ORANGE = "#fc8d59"
C_LTBLUE = "#91bfdb"
C_DKBLUE = "#313695"
C_GREY = "#bdbdbd"

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'pdf.fonttype': 42,    # TrueType fonts (avoid Type 3)
    'ps.fonttype': 42,
})


# ── Data loading (reused from run_all.py) ──

def clean_modeling(x):
    if pd.isna(x): return "Other"
    s = str(x).strip().lower()
    if s.startswith("mixed") or "mix of" in s or "depends" in s: return "Mixed"
    if "ad-hoc" in s or "ad hoc" in s or "tables added" in s: return "Ad-hoc"
    if "one big table" in s or re.search(r"\bobt\b", s) or "1 table per report" in s:
        if any(k in s for k in ["kimball","data vault","datavault","semantic","canonical"]): return "Mixed"
        return "One Big Table"
    if "data vault" in s or "datavault" in s or re.search(r"\bdv2?(?:\.0)?\b", s):
        if any(k in s for k in ["kimball","obt"]): return "Mixed"
        return "Data Vault"
    if "kimball" in s or "dimensional" in s:
        if any(k in s for k in ["mixed","mix","obt","data vault","datavault"]): return "Mixed"
        return "Kimball"
    if "event" in s:
        if any(k in s for k in ["canonical","kimball"]): return "Mixed"
        return "Event-driven"
    if "canonical" in s or "semantic" in s or "knowledge graph" in s: return "Canonical/Semantic"
    return "Other"


def build_practical(path):
    df = pd.read_parquet(path)
    df["ai_adoption"] = pd.Categorical(df["ai_adoption"], categories=ADOPTION_ORDER, ordered=True)
    df["ai_usage_frequency"] = pd.Categorical(df["ai_usage_frequency"], categories=FREQ_ORDER, ordered=True)
    df["daily_plus"] = df["ai_usage_frequency"].isin(["Daily","Multiple times per day"])
    df["org_beyond_experimentation"] = df["ai_adoption"].isin(["Building internal AI platforms","AI embedded in most workflows"])
    df["shadow_ai"] = df["daily_plus"] & df["ai_adoption"].isin(["No meaningful adoption yet","Experimenting"])
    df["modeling_cat"] = df["modeling_approach"].apply(clean_modeling)

    pain_tokens = {
        "pain_ownership": "Lack of clear ownership",
        "pain_speed": "Pressure to \u201cmove fast\u201d",
        "pain_maintain": "Hard to maintain over time",
        "pain_tools": "Tools don\u2019t support good modeling",
        "pain_ai_schema": "AI tools produce inconsistent schemas",
        "pain_none": "None / modeling is going well",
    }
    for col, tok in pain_tokens.items():
        df[col] = df["modeling_pain_points"].fillna("").str.contains(re.escape(tok))

    main_bottlenecks = {
        "Legacy / technical debt": "Technical", "Data quality": "Technical",
        "Compute costs": "Technical", "Tool complexity": "Technical",
        "Lack of leadership direction": "Organizational",
        "Poor requirements / upstream issues": "Organizational",
        "Talent / hiring": "Organizational", "Other": "Other",
    }
    options = set(main_bottlenecks.keys())
    df["bottleneck_cat"] = df["biggest_bottleneck"].apply(lambda x: x if x in options else "Other")
    df["bottleneck_type"] = df["bottleneck_cat"].map(main_bottlenecks).fillna("Other")
    return df


def build_stackoverflow(path):
    usecols = ["ResponseId","MainBranch","DevType","OrgSize","Industry",
               "AISelect","AIAcc","AISent","AIAgents","AIAgentChange",
               "AIFrustration","AIComplex"]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)

    df["usage_bucket"] = df["AISelect"].map({
        "Yes, I use AI tools daily": "Daily",
        "Yes, I use AI tools weekly": "Weekly",
        "Yes, I use AI tools monthly or infrequently": "Monthly/infrequent",
        "No, but I plan to soon": "Plan to soon",
        "No, and I don't plan to": "Don't plan",
    })
    df["trust_class"] = df["AIAcc"].map({
        "Highly trust": "Trust", "Somewhat trust": "Trust",
        "Neither trust nor distrust": "Neutral",
        "Somewhat distrust": "Distrust", "Highly distrust": "Distrust",
    })

    def agent_bucket(x):
        if pd.isna(x): return "Missing"
        x = str(x)
        if x.startswith("Yes, I use AI agents at work daily"): return "Agents daily"
        if x.startswith("Yes, I use AI agents at work weekly"): return "Agents weekly"
        if x.startswith("Yes, I use AI agents at work monthly"): return "Agents monthly+"
        if x.startswith("No, I use AI exclusively in copilot"): return "Copilot only"
        if x.startswith("No, but I plan"): return "Plan to use agents"
        if x.startswith("No, and I don't plan"): return "Don't plan"
        return "Other"

    df["agent_bucket"] = df["AIAgents"].apply(agent_bucket)
    frust = df["AIFrustration"].fillna("")
    df["frust_almost_right"] = frust.str.contains(re.escape("AI solutions that are almost right, but not quite"))
    df["frust_debug_time"] = frust.str.contains(re.escape("Debugging AI-generated code is more time-consuming"))
    df["frust_hard_understand"] = frust.str.contains(re.escape("It's hard to understand how or why the code works"))
    return df


# ── Figure generators ──

def fig1_usage_and_adoption(pr, outdir):
    """Double-column: usage frequency + adoption maturity side by side."""
    freq_counts = pr["ai_usage_frequency"].value_counts().reindex(FREQ_ORDER)
    freq_pct = freq_counts / freq_counts.sum() * 100
    adopt_counts = pr["ai_adoption"].value_counts().reindex(ADOPTION_ORDER)
    adopt_pct = adopt_counts / adopt_counts.sum() * 100

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.2))

    ax = axes[0]
    y = range(len(freq_counts))
    bars = ax.barh(list(y), freq_counts.values, color=C_BLUE, height=0.65)
    ax.set_yticks(list(y), labels=[l if l != "Multiple times per day" else "Multiple/day" for l in freq_counts.index], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Respondents")
    ax.set_title("(a) AI tool usage frequency")
    for i, (v, p) in enumerate(zip(freq_counts.values, freq_pct.values)):
        ax.text(v + max(freq_counts.values)*0.02, i, f"{p:.0f}%", va="center", fontsize=6.5)

    ax = axes[1]
    y = range(len(adopt_counts))
    short_labels = ["No adoption", "Experimenting", "Tactical", "Building\nplatforms", "Embedded"]
    bars = ax.barh(list(y), adopt_counts.values, color=C_RED, height=0.65)
    ax.set_yticks(list(y), labels=short_labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Respondents")
    ax.set_title("(b) Organizational AI adoption maturity")
    for i, (v, p) in enumerate(zip(adopt_counts.values, adopt_pct.values)):
        ax.text(v + max(adopt_counts.values)*0.02, i, f"{p:.0f}%", va="center", fontsize=6.5)

    plt.tight_layout(w_pad=1.5)
    fig.savefig(os.path.join(outdir, "fig1_usage_and_adoption.pdf"))
    fig.savefig(os.path.join(outdir, "fig1_usage_and_adoption.png"))
    plt.close(fig)


def fig2_shadow_ai(pr, outdir):
    """Double-column: stacked bar of adoption within daily+ vs weekly-or-less."""
    daily = pr[pr["daily_plus"]]
    nondaily = pr[~pr["daily_plus"]]

    daily_dist = daily.groupby("ai_adoption", observed=True).size().reindex(ADOPTION_ORDER).fillna(0)
    daily_pct = daily_dist / daily_dist.sum() * 100
    nondaily_dist = nondaily.groupby("ai_adoption", observed=True).size().reindex(ADOPTION_ORDER).fillna(0)
    nondaily_pct = nondaily_dist / nondaily_dist.sum() * 100

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 1.8))
    groups = ["Daily+ users", "Weekly or less"]
    data = np.vstack([daily_pct.values, nondaily_pct.values])
    left = np.zeros(2)
    colors = [C_RED, C_ORANGE, C_LTBLUE, C_BLUE, C_DKBLUE]
    labels = ["No adoption", "Experimenting", "Tactical", "Platforms", "Embedded"]

    for i in range(5):
        ax.barh(groups, data[:, i], left=left, color=colors[i], label=labels[i], height=0.55)
        left += data[:, i]

    ax.set_xlim(0, 100)
    ax.set_xlabel("Share within group (%)")
    ax.xaxis.set_major_formatter(PercentFormatter())
    ax.legend(ncol=5, bbox_to_anchor=(0.5, -0.35), loc="upper center", frameon=False, fontsize=6.5)

    shadow_share = float(daily_pct.iloc[0] + daily_pct.iloc[1])
    ax.text(shadow_share/2, 0, f"Shadow AI\n{shadow_share:.1f}%", va="center", ha="center",
            color="white", fontsize=7, fontweight="bold")

    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "fig2_shadow_ai_segmentation.pdf"))
    fig.savefig(os.path.join(outdir, "fig2_shadow_ai_segmentation.png"))
    plt.close(fig)


def fig3_bottleneck(pr, outdir):
    """Single-column: stacked bar of bottleneck type by adoption stage."""
    counts = pr.groupby(["ai_adoption","bottleneck_type"], observed=True).size().reset_index(name="n")
    counts["pct"] = counts.groupby("ai_adoption")["n"].transform(lambda x: x/x.sum()*100)
    pivot = counts.pivot(index="ai_adoption", columns="bottleneck_type", values="pct").fillna(0).reindex(ADOPTION_ORDER)

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.4))
    x = np.arange(5)
    bottom = np.zeros(5)
    palette = {"Organizational": C_RED, "Technical": C_BLUE, "Other": C_GREY}
    for col in ["Organizational", "Technical", "Other"]:
        ax.bar(x, pivot[col].values, bottom=bottom, label=col, color=palette[col], width=0.7)
        bottom += pivot[col].values

    ax.set_xticks(x, labels=["No adopt.", "Exper.", "Tactical", "Platf.", "Embed."], fontsize=6)
    ax.set_ylabel("Share (%)")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.legend(ncol=1, loc="upper right", frameon=False, fontsize=6)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "fig3_bottleneck_inversion.pdf"))
    fig.savefig(os.path.join(outdir, "fig3_bottleneck_inversion.png"))
    plt.close(fig)


def fig4_modeling(pr, outdir):
    """Single-column: stacked bar of modeling approach by adoption stage."""
    tmp = pr.groupby(["ai_adoption","modeling_cat"], observed=True).size().reset_index(name="n")
    tmp["pct"] = tmp.groupby("ai_adoption")["n"].transform(lambda x: x/x.sum()*100)
    model_pivot = tmp.pivot(index="ai_adoption", columns="modeling_cat", values="pct").fillna(0).reindex(ADOPTION_ORDER)

    cats = ["Ad-hoc","Kimball","Mixed","Canonical/Semantic","Data Vault","Event-driven","One Big Table","Other"]
    cats = [c for c in cats if c in model_pivot.columns]
    cat_colors = {
        "Ad-hoc": C_RED, "Kimball": C_ORANGE, "Mixed": C_BLUE,
        "Canonical/Semantic": C_LTBLUE, "Data Vault": "#ffffbf",
        "Event-driven": "#a6d96a", "One Big Table": "#1a9641", "Other": C_GREY,
    }

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.4))
    x = np.arange(5)
    bottom = np.zeros(5)
    for c in cats:
        ax.bar(x, model_pivot[c].values, bottom=bottom, label=c, color=cat_colors.get(c, C_GREY), width=0.7)
        bottom += model_pivot[c].values

    ax.set_xticks(x, labels=["No adopt.", "Exper.", "Tactical", "Platf.", "Embed."], fontsize=6)
    ax.set_ylabel("Share (%)")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.legend(ncol=2, bbox_to_anchor=(0.5, -0.3), loc="upper center", frameon=False, fontsize=5.5)

    if "Ad-hoc" in model_pivot.columns:
        ax.text(0, model_pivot.iloc[0]["Ad-hoc"]/2, f'{model_pivot.iloc[0]["Ad-hoc"]:.0f}%',
                ha="center", va="center", color="white", fontsize=6, fontweight="bold")
        ax.text(4, model_pivot.iloc[4]["Ad-hoc"]/2, f'{model_pivot.iloc[4]["Ad-hoc"]:.0f}%',
                ha="center", va="center", color="white", fontsize=6, fontweight="bold")

    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "fig4_modeling_mix.pdf"))
    fig.savefig(os.path.join(outdir, "fig4_modeling_mix.png"))
    plt.close(fig)


def fig5_shadow_predictors(pr, outdir):
    """Double-column: forest plot of AME for Shadow AI predictors."""
    daily_plus = pr[pr["daily_plus"]].copy()
    daily_plus["shadow_ai_int"] = daily_plus["shadow_ai"].astype(int)
    daily_plus["modeling_cat"] = pd.Categorical(
        daily_plus["modeling_cat"],
        categories=["Mixed","Ad-hoc","Kimball","Canonical/Semantic","Data Vault","Event-driven","One Big Table","Other"],
    )
    daily_plus["bottleneck_type"] = pd.Categorical(daily_plus["bottleneck_type"], categories=["Technical","Organizational","Other"])

    # Group roles exactly as in run_all.py to match the official analysis
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
        categories=["Data Engineer","Manager/Director/VP","Analytics Engineer","Data Architect",
                     "Software Engineer","Platform Engineer","ML/AI Engineer","Other"],
    )
    for c in ["org_size","industry","team_growth_2026"]:
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
        ("pain_ownership[T.True]", "Unclear ownership"),
        ("C(modeling_cat, Treatment(reference='Mixed'))[T.Ad-hoc]", "Ad-hoc modelling"),
        ("C(org_size)[T.10,000+]", "Org size: 10,000+"),
        ("C(bottleneck_type, Treatment(reference='Technical'))[T.Organizational]", "Organizational bottleneck"),
    ]

    rows = []
    for key, label in terms:
        r = me.loc[key]
        rows.append((label, r["dy/dx"]*100, r["Conf. Int. Low"]*100, r["Cont. Int. Hi."]*100, r["Pr(>|z|)"]))

    plot_df = pd.DataFrame(rows, columns=["term","pp","low","high","p"]).iloc[::-1]

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 1.8))
    y = np.arange(len(plot_df))
    ax.hlines(y, plot_df["low"], plot_df["high"], color="black", linewidth=1.2)
    ax.plot(plot_df["pp"], y, "ko", markersize=5)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_yticks(y, labels=plot_df["term"])
    ax.set_xlabel("AME on P(Shadow AI), pp")
    # Stars for significance
    def sig_stars(p):
        if p < 0.001: return "***"
        if p < 0.01: return "**"
        if p < 0.05: return "*"
        return ""
    for i, (pp, p) in enumerate(zip(plot_df["pp"], plot_df["p"])):
        stars = sig_stars(p)
        label = f"{pp:+.1f}{stars}"
        # Place all annotations to the right of the CI endpoint to avoid collision
        right_edge = max(plot_df.iloc[i]["high"], pp)
        ax.text(right_edge + 1.5, i, label, va="center", ha="left", fontsize=7)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "fig5_shadow_ai_predictors.pdf"))
    fig.savefig(os.path.join(outdir, "fig5_shadow_ai_predictors.png"))
    plt.close(fig)


def fig6_so_trust(so, outdir):
    """Double-column: (a) trust by agent depth, (b) debugging overhead."""
    so_trust = so[~so["AIAcc"].isna()].copy()
    so_trust = so_trust[so_trust["usage_bucket"] == "Daily"].copy()

    agent_order = ["Agents daily","Agents weekly","Agents monthly+","Copilot only","Plan to use agents","Don't plan"]
    tab = so_trust[so_trust["agent_bucket"].isin(agent_order)].groupby(["agent_bucket","trust_class"]).size().reset_index(name="n")
    tab["pct"] = tab.groupby("agent_bucket")["n"].transform(lambda x: x/x.sum()*100)
    pivot = tab.pivot(index="agent_bucket", columns="trust_class", values="pct").fillna(0).reindex(agent_order)

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.4))

    ax = axes[0]
    x = np.arange(len(agent_order))
    bottom = np.zeros(len(agent_order))
    palette = {"Trust": C_BLUE, "Neutral": C_GREY, "Distrust": C_RED}
    for cls in ["Distrust","Neutral","Trust"]:
        ax.bar(x, pivot[cls].values, bottom=bottom, label=cls, color=palette[cls], width=0.7)
        bottom += pivot[cls].values
    ax.set_xticks(x, labels=["Agents\ndaily","Agents\nweekly","Agents\nmonth+","Copilot\nonly","Plan\nsoon","Don't\nplan"], fontsize=5.5)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_ylabel("Share (%)")
    ax.set_title("(a) Trust by AI-agent usage depth")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False, fontsize=6)

    ax = axes[1]
    sub = so_trust[so_trust["trust_class"] != "Neutral"].copy()
    sub["distrust"] = sub["trust_class"].eq("Distrust")
    rates = (sub.groupby("frust_debug_time")["distrust"].mean() * 100).to_dict()
    labels = ["No debug\noverhead", "Debug\noverhead"]
    vals = [rates.get(False, np.nan), rates.get(True, np.nan)]
    ax.bar(labels, vals, color=[C_BLUE, C_RED], width=0.55)
    ax.set_ylabel("Distrust rate (%)\n(excl. neutral)")
    ax.set_ylim(0, 80)
    ax.set_title("(b) Debugging overhead and distrust")
    for i, v in enumerate(vals):
        ax.text(i, v + 2, f"{v:.1f}%", ha="center", fontsize=7)

    plt.tight_layout(w_pad=1.5)
    fig.savefig(os.path.join(outdir, "fig6_so_trust_mechanism.pdf"))
    fig.savefig(os.path.join(outdir, "fig6_so_trust_mechanism.png"))
    plt.close(fig)


def fig7_dual_gaps(pr, so, outdir):
    """Double-column: side-by-side bar charts of the two adoption gaps."""
    daily_pct = float(pr["daily_plus"].mean() * 100)
    beyond_pct = float(pr["org_beyond_experimentation"].mean() * 100)

    so_nn = so[~so["AISelect"].isna()]
    weekly_pct = float(so_nn["AISelect"].isin(["Yes, I use AI tools daily","Yes, I use AI tools weekly"]).mean() * 100)
    so_acc = so[~so["AIAcc"].isna()]
    trust_pct = float(so_acc["AIAcc"].isin(["Highly trust","Somewhat trust"]).mean() * 100)

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.0))

    ax = axes[0]
    vals = [daily_pct, beyond_pct]
    ax.bar(["Daily+ AI use", "Org: platforms\nor embedded"], vals, color=[C_BLUE, C_RED], width=0.55)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_title("(a) Practical Data (n=1,101)")
    for i, v in enumerate(vals):
        ax.text(i, v + 2, f"{v:.1f}%", ha="center", fontsize=8, fontweight="bold")

    ax = axes[1]
    vals = [weekly_pct, trust_pct]
    ax.bar(["Weekly+ AI use", "Trust AI\naccuracy"], vals, color=[C_BLUE, C_RED], width=0.55)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PercentFormatter())
    so_usage_n = so_nn.shape[0]
    so_trust_n = so_acc.shape[0]
    ax.set_title(f"(b) Stack Overflow\n(usage n={so_usage_n:,}; trust n={so_trust_n:,})", fontsize=7.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 2, f"{v:.1f}%", ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout(w_pad=1.5)
    fig.savefig(os.path.join(outdir, "fig7_dual_survey_gaps.pdf"))
    fig.savefig(os.path.join(outdir, "fig7_dual_survey_gaps.png"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--practical", required=True)
    parser.add_argument("--stackoverflow", required=True)
    parser.add_argument("--outdir", default="figures")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    pr = build_practical(args.practical)
    so = build_stackoverflow(args.stackoverflow)

    fig1_usage_and_adoption(pr, args.outdir)
    print("  fig1 done")
    fig2_shadow_ai(pr, args.outdir)
    print("  fig2 done")
    fig3_bottleneck(pr, args.outdir)
    print("  fig3 done")
    fig4_modeling(pr, args.outdir)
    print("  fig4 done")
    fig5_shadow_predictors(pr, args.outdir)
    print("  fig5 done")
    fig6_so_trust(so, args.outdir)
    print("  fig6 done")
    fig7_dual_gaps(pr, so, args.outdir)
    print("  fig7 done")
    print("All figures generated.")


if __name__ == "__main__":
    main()
