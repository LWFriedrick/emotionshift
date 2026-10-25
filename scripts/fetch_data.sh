#!/usr/bin/env bash
# Download MELD and DailyDialog into data/raw/ (run from the repo root).
#
# MELD:        official CSVs from declare-lab/MELD.
# DailyDialog: official 11,118 / 1,000 / 1,000 split as JSON lines, taken from
#              declare-lab/conv-emotion (DialogueRNN/dailydialog.zip), because
#              the original host (yanran.li) is often unreachable.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p data/raw/meld data/raw/dailydialog

MELD=https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD
for s in train dev test; do
  curl -fsSL "$MELD/${s}_sent_emo.csv" -o "data/raw/meld/${s}_sent_emo.csv"
done

tmp=$(mktemp -d)
curl -fsSL https://raw.githubusercontent.com/declare-lab/conv-emotion/master/DialogueRNN/dailydialog.zip -o "$tmp/dd.zip"
python3 - "$tmp/dd.zip" <<'EOF'
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
for s in ["train", "valid", "test"]:
    open(f"data/raw/dailydialog/{s}.json", "wb").write(z.read(f"dailydialog/{s}.json"))
EOF
rm -rf "$tmp"

echo "Done:"; wc -l data/raw/meld/*.csv data/raw/dailydialog/*.json
