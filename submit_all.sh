#!/bin/bash
# Usage: ./submit_all.sh <folder> ["N1 N2 ..."]
# Example: ./submit_all.sh CUDA/01_single_techniques "256 512 1024"
DIR=${1:?usage: ./submit_all.sh <folder> [sizes]}
SIZES=${2:-"256 512 1024"}
for f in "$DIR"/*.cu; do
  [ -e "$f" ] || { echo "No .cu files in $DIR"; exit 1; }
  echo "Submitting $f"
  ./submit.sh "$f" "$SIZES"
done
