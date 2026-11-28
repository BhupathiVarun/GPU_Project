#!/bin/bash
#SBATCH --job-name=cuda_run
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

cd "$SLURM_SUBMIT_DIR" || exit 1
SRC=${SRC:?SRC not set}
SIZES=${SIZES:-"256 512 1024"}

NAME=$(basename "$SRC" .cu)
SUBDIR=$(basename "$(dirname "$SRC")")
[ "$SUBDIR" = "03_final" ] && SUBDIR=final
OUTDIR=Results/bhavani_cluster/$SUBDIR
OUT=$OUTDIR/${NAME}_result.txt
mkdir -p bin "$OUTDIR"

module purge
module load cuda/11.3

CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
if [[ "$CC" =~ ^[0-9]+$ ]] && [ "$CC" -ge 60 ] && [ "$CC" -le 86 ]; then
  ARCH="-arch=sm_$CC"
else
  ARCH="-gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86"
fi

nvcc -O3 -std=c++14 -ccbin /usr/bin/g++ $ARCH -o "bin/$NAME" "$SRC" -lcublas || { echo "COMPILE FAILED"; exit 1; }

{
  echo "# Job $SLURM_JOB_ID on $(hostname) at $(date)"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  nvcc --version | tail -2
  echo "# nvcc flags: -O3 -ccbin /usr/bin/g++ $ARCH -lcublas"
  for N in $SIZES; do
    echo; echo "########## N = $N ##########"
    ./bin/$NAME $N
  done
} > "$OUT" 2>&1

echo "Wrote $OUT"
