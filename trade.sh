#!/bin/bash

# Research helper
# Usage: ./trade.sh <SYMBOL> [STRATEGY]

STOCK=$1
STRATEGY=${2:-"General Technical and Fundamental Analysis"}
PROJECT_ID="${GCP_PROJECT_ID}"
LOCATION="${GCP_LOCATION:-us-central1}"

if [ -z "$STOCK" ]; then
  echo "Usage: ./trade.sh <SYMBOL> [STRATEGY]"
  exit 1
fi

if [ -z "$PROJECT_ID" ]; then
  echo "GCP_PROJECT_ID is not set. See .env.example."
  exit 1
fi

echo "Analysing $STOCK with strategy '$STRATEGY'..."
echo "-----------------------------------------------------"

python scripts/research_trade.py \
  --stock "$STOCK" \
  --strategy "$STRATEGY" \
  --project "$PROJECT_ID" \
  --location "$LOCATION"
