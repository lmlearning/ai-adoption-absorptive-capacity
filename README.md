# AI Adoption and the Absorptive Capacity Gap

[![Tests](https://github.com/lmlearning/ai-adoption-absorptive-capacity/actions/workflows/tests.yml/badge.svg)](https://github.com/lmlearning/ai-adoption-absorptive-capacity/actions/workflows/tests.yml)

**Why can widespread individual AI use coexist with limited organizational adoption?** This repository contains the data preparation, statistical models, figures and manuscript for *The Absorptive Capacity Gap in AI Adoption: A Segmented, Stage-Based Explanation from Two Developer Surveys*.

Lars Malmqvist, Research and Implementation ApS. Accepted at CADE 2026 (International Conference on AI and the Digital Economy).

## Verify an included result in minutes

Use Python 3.11 in an activated virtual environment, from the repository root:

```bash
python -m pip install -r requirements.txt pytest
python code/verify_results.py
python -m pytest -q tests
```

The verifier recomputes the sample-composition table and usage-versus-maturity correlation from the included 1,101-response Practical Data survey. It checks them against the exported CSVs and reports 905 daily-or-more AI users and Spearman rho approximately 0.203. No external download is needed for this check.

## Inspect the research

| Material | Purpose |
| --- | --- |
| [paper/main.tex](paper/main.tex) | Research question, construct definitions, methods and interpretation. |
| [code/run_all.py](code/run_all.py) | Preparation, both survey models, tables and figures. |
| [code/analysis_stats.py](code/analysis_stats.py) | Pairwise-complete ordinal correlation. |
| [outputs](outputs/) | Exported descriptive and regression tables. |
| [figures](figures/) | Figures used by the paper. |
| [data/README.md](data/README.md) | Sources, pinned download, populations and data-license scope. |

The study is cross-sectional: reported associations are not causal effects. The two surveys have different sampling frames and question wordings. The Practical Data sample contains 1,101 responses; the Stack Overflow analysis uses 49,191 public-microdata rows and question-specific subsets, distinguished from the publisher's 49,009 qualified-response headline.

## Reproduce both surveys

```bash
python data/download_stackoverflow.py
python code/run_all.py --practical data/practical_data_survey_2026.parquet --stackoverflow data/stackoverflow_2025_survey.csv --outdir reproduction
```

The downloader retrieves about 141 MB from the official Stack Exchange archive at a pinned commit and verifies its SHA-256 before making it available. Existing files with unexpected contents are preserved and reported. The Bash download script is a wrapper for the same cross-platform implementation.

Tables are written to `reproduction/analysis_outputs/` and PNG/PDF charts to `reproduction/figures/`, keeping committed artifacts intact. A full CPU reproduction was run during the repository review: all four exported CSVs matched numerically after aligning row keys (relative tolerance `1e-6`, absolute tolerance `1e-12`), and 14 figure files were generated. CI runs the lightweight included-data verifier and tests rather than downloading the large survey on every PR.

For publication-layout charts:

```bash
python code/make_charts_latex.py --practical data/practical_data_survey_2026.parquet --stackoverflow data/stackoverflow_2025_survey.csv --outdir reproduction/latex_figures
```

## Statistical conventions

Usage/maturity correlation aligns respondents and removes missing response pairs before category encoding. Missing values cannot become an artificial lowest rank. Constant variables or fewer than two complete pairs yield undefined statistics. The shipped Practical Data sample has complete pairs, so the original reported correlation is unchanged.

The historical exploratory JSON summaries in `data/` are separate from the manuscript's exported tables. Use `run_all.py` and `outputs/` when checking reported model results.

## Build, contribute and cite

To build the paper, use a LaTeX distribution with the packages listed in `paper/main.tex`:

```bash
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

The manuscript already resolves figures from `../figures/`; no symlink is needed. A proposed analytical change should include the sample definition, affected estimand and comparison with recorded outputs.

[Citation metadata](CITATION.cff). Code is [MIT-licensed](LICENSE); survey data retains the [publishers' terms and attribution](data/README.md).
