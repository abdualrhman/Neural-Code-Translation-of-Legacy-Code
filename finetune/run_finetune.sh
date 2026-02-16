#!/bin/sh
#BSUB -q gpuh100
#BSUB -J finetune_job
#BSUB -n 8
#BSUB -W 10:00
#BSUB -R "rusage[mem=64GB]"
#BSUB -R "select[gpu80gb]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o batch_output/qwen_%J.out
#BSUB -e batch_output/qwen_%J.err

llm_name="Qwen/Qwen3-32B"
file_name="finetune_qwen.py"

mkdir -p batch_output

cat << EOF
==========================================
$llm_name Fine-tuning Job
==========================================
Job ID:       $LSB_JOBID
Job Name:     $LSB_JOBNAME
Queue:        $LSB_QUEUE
Host:         $LSB_HOSTS
Cores:        $LSB_MAX_NUM_PROCESSORS
Start Time:   $(date)
Working Dir:  $(pwd)
==========================================

EOF

echo "Activating python environment..."
source ../../.venv/bin/activate

echo "Environment Information:"
echo "------------------------"
echo "Python: $(python --version 2>&1)"
echo ""

python << 'PYEOF'
import torch
import transformers
import peft
import bitsandbytes

print(f"PyTorch version:       {torch.__version__}")
print(f"Transformers version:  {transformers.__version__}")
print(f"PEFT version:          {peft.__version__}")
print(f"bitsandbytes version:  {bitsandbytes.__version__}")
print(f"CUDA available:        {torch.cuda.is_available()}")
print(f"CUDA version:          {torch.version.cuda}")
print(f"cuDNN version:         {torch.backends.cudnn.version()}")
print(f"Number of GPUs:        {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU 0:                 {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory:            {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
PYEOF

echo ""
echo "GPU Status:"
echo "-----------"
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv
echo ""

export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT=apl-to-csharp
export WANDB_RUN_NAME="$llm_name-lora-${LSB_JOBID}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "Training Configuration:"
echo "----------------------"
echo "Model: $llm_name"
echo "Output dir: /work3/s204476/$llm_name"
echo "Epochs: 20"
echo "Batch size: 4 (train), 8 (eval)"
echo "Gradient accumulation: 4 steps"
echo "Effective batch size: 16"
echo "Learning rate: 1e-4"
echo "LoRA rank: 64"
echo ""

echo "=========================================="
echo "Starting Fine-tuning"
echo "Time: $(date)"
echo "=========================================="
echo ""

set -e
python $file_name || {
    echo ""
    echo "ERROR: Training failed with exit code $?"
    echo "Time: $(date)"
    exit 1
}

echo ""
echo "=========================================="
echo "Training Completed Successfully"
echo "Time: $(date)"
echo "Output saved to: /work3/s204476/$model_name"
echo "=========================================="

echo ""
echo "Final GPU Memory Status:"
nvidia-smi
