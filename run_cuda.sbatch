#!/bin/bash
#SBATCH --job-name=cuda_run
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# Needs env var: SRC (e.g. CUDA/01_single_techniques/t5_register_blocking.cu)
# Optional:      SIZES (default "256 512 1024")
cd "$SLURM_SUBMIT_DIR" || exit 1
SRC=${SRC:?SRC not set}
SIZES=${SIZES:-"256 512 1024"}

NAME=$(basename "$SRC" .cu)
SUBDIR=$(basename "$(dirname "$SRC")")
OUTDIR=Results/bhavani_cluster/$SUBDIR
OUT=$OUTDIR/${NAME}_result.txt
mkdir -p bin "$OUTDIR"

module purge
module load cuda/11.3

# Try to match the GPU architecture; fall back to nvcc's default if that fails
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
ARCH=${CC:+-arch=sm_$CC}

if ! nvcc -O3 -std=c++14 $ARCH -o "bin/$NAME" "$SRC"; then
  echo "NOTE: compile with '$ARCH' failed, retrying with default arch"
  ARCH="(default)"
  nvcc -O3 -std=c++14 -o "bin/$NAME" "$SRC" || { echo "COMPILE FAILED"; exit 1; }
fi

{
  echo "# Job $SLURM_JOB_ID on $(hostname) at $(date)"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  nvcc --version | tail -2
  echo "# nvcc flags: -O3 $ARCH"
  for N in $SIZES; do
    echo; echo "########## N = $N ##########"
    ./bin/$NAME $N
  done
} > "$OUT" 2>&1

echo "Wrote $OUT"
