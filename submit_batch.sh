#!/bin/bash
#SBATCH --job-name=eureka
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=eureka-%j.out

set -euo pipefail

# Submit from the repository root. Arguments to sbatch override Eureka defaults below.
cd "${SLURM_SUBMIT_DIR:?}"
if [[ ! -f eureka/eureka.py ]]; then
    echo "Run sbatch submit_batch.sh from the Eureka repository root." >&2
    exit 1
fi

IFS=, read -r qwen_gpu training_gpu extra_gpu <<< "${CUDA_VISIBLE_DEVICES:-}"
if [[ -z "$qwen_gpu" || -z "$training_gpu" || -n "$extra_gpu" ]]; then
    echo "Expected two GPUs in CUDA_VISIBLE_DEVICES, got '${CUDA_VISIBLE_DEVICES:-}'." >&2
    exit 1
fi

model=cyankiwi/Qwen3.8-27B-AWQ-INT4
port=${QWEN_PORT:-8000}
base_url="http://127.0.0.1:$port/v1"
server_log="$PWD/vllm-${SLURM_JOB_ID}.log"

eval "$(conda shell.bash hook)"
conda activate eureka-vllm
if curl -s --max-time 2 "$base_url/models" >/dev/null; then
    echo "Port $port is already serving HTTP on $(hostname). Set QWEN_PORT to another port." >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES="$qwen_gpu" VLLM_USE_FLASHINFER_SAMPLER=0 \
    vllm serve "$model" --host 127.0.0.1 --port "$port" \
    --language-model-only --gpu-memory-utilization 0.97 \
    --max-model-len 4096 --max-num-seqs 1 --enforce-eager \
    >"$server_log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Waiting for Qwen on $(hostname), GPU $qwen_gpu. Server log: $server_log"
deadline=$((SECONDS + 900))
until curl -fsS --max-time 2 "$base_url/models" >/dev/null 2>&1; do
    if ! kill -0 "$server_pid" 2>/dev/null || (( SECONDS >= deadline )); then
        echo "Qwen did not start. See $server_log." >&2
        tail -n 30 "$server_log" >&2
        exit 1
    fi
    sleep 5
done

conda activate eureka
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="$training_gpu"
cd eureka
echo "Starting Eureka on GPU $training_gpu"
python -c 'import isaacgym'
python -u eureka.py env=humanoid sample=1 iteration=1 max_iterations=1000 \
    capture_video=False llm_max_tokens=1024 \
    "model=$model" "llm_base_url=$base_url" "$@"
