#!/usr/bin/env bash
# Download the 2025 Stack Overflow Developer Survey public microdata.
# Source: https://survey.stackoverflow.co/2025/ (Open Database License)
#
# The file is ~135 MB and is required to reproduce Model 2 and Figures 6-7.

set -euo pipefail

URL="https://cdn.stackoverflow.co/files/jo7n4k8s/production/131f9e02-33ab-4371-afd0-0e8ad4451f0c.zip"
OUT="stackoverflow_2025_survey.csv"

if [ -f "$OUT" ]; then
    echo "$OUT already exists, skipping download."
    exit 0
fi

echo "Downloading Stack Overflow 2025 survey data..."
curl -L -o so2025.zip "$URL"
unzip -o so2025.zip survey_results_public.csv
mv survey_results_public.csv "$OUT"
rm -f so2025.zip
echo "Saved to $OUT"
