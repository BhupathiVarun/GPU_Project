#!/bin/bash
# Usage: ./submit.sh <path/to/file.cu> ["N1 N2 ..."]
mkdir -p slurm_logs
SRC=$1; SIZES=${2:-"256 512 1024"}
sbatch --job-name="$(basename "$SRC" .cu)" \
       --export=ALL,SRC="$SRC",SIZES="$SIZES" run_cuda.sbatch
