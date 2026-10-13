# Shared setup for the run scripts: repo root, GPUs, Python, and a torchrun launcher.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
GPUS="${GPUS:-0}"                      # comma-separated GPU ids, e.g. GPUS=0,1,2,3
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
NGPU="${#GPU_LIST[@]}"
PYTHON="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="$GPUS"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
launch() { "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NGPU" "$@"; }
