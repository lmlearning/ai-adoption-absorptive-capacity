# Reproducibility materials

**The Absorptive Capacity Gap in AI Adoption: A Segmented, Stage-Based Explanation from Two Developer Surveys**

Lars Malmqvist, Research and Implementation ApS

Accepted at CADE 2026 (International Conference on AI and the Digital Economy).

## Repository structure

```
repo/
├── code/
│   ├── run_all.py              # Full analysis: data prep, models, figures, CSV exports
│   └── make_charts_latex.py    # Publication-quality figures for the LaTeX paper
├── data/
│   ├── practical_data_survey_2026.parquet   # Practical Data survey (n=1,101)
│   ├── stackoverflow_2025_schema.csv        # SO survey schema/codebook
│   ├── download_stackoverflow.sh            # Script to download SO microdata (~135 MB)
│   ├── practical_data_analysis_results.json # Pre-computed Practical Data results
│   └── stackoverflow_analysis_results.json  # Pre-computed SO results
├── paper/
│   ├── main.tex                # LaTeX source
│   ├── references.bib          # Bibliography
│   └── chronnat.bst            # Custom BibTeX style (chronological sorting)
├── figures/                    # All figure PDFs used in the paper
├── outputs/                    # Model output CSVs
│   ├── table1_practical_sample_composition.csv
│   ├── table2_shadow_ai_logit_key_predictors.csv
│   ├── table3_so_daily_distrust_logit_key_predictors.csv
│   └── spearman_usage_vs_maturity.csv
└── requirements.txt
```

## Reproducing the analysis

```bash
pip install -r requirements.txt

# Download Stack Overflow survey data (required for Model 2 and Figures 6-7)
cd data && bash download_stackoverflow.sh && cd ..

# Run full analysis and regenerate all figures
python code/run_all.py \
  --practical data/practical_data_survey_2026.parquet \
  --stackoverflow data/stackoverflow_2025_survey.csv \
  --outdir .

# Generate publication-quality LaTeX figures
python code/make_charts_latex.py \
  --practical data/practical_data_survey_2026.parquet \
  --stackoverflow data/stackoverflow_2025_survey.csv \
  --outdir figures
```

## Building the paper

Requires a LaTeX distribution with `pdflatex`, `bibtex`, and `natbib`.

```bash
cd paper
ln -s ../figures .
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Data sources

- **Practical Data Community State of Data Engineering Survey (2026)**: https://joereis.github.io/practical_data_data_eng_survey/
- **Stack Overflow Developer Survey 2025**: https://survey.stackoverflow.co/2025/ (Open Database License)

## License

Code: MIT. Data: see respective survey licenses above.
