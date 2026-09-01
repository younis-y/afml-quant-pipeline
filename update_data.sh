#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"   # always run from project root regardless of where Terminal is

echo "=== Step 1: Upgrading packages ==="
pip install --upgrade pip
pip install --upgrade -r requirements.txt

echo ""
echo "=== Step 2: Downloading latest data to $(date +%Y-%m-%d) ==="
cd scripts
python 00_data_download.py

echo ""
echo "=== Done! Data updated to $(date +%Y-%m-%d) ==="
