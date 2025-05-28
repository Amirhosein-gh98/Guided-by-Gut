# !/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CMD="CUDA_VISIBLE_DEVICES=0,1 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes 1 src/open_r1/grpo.py --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/LIMO_ConfReward.yaml"

while true; do
    echo "Running training at $(date)"
    eval $CMD
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "Training finished successfully."
        break
    else
        echo "Training crashed with exit code $EXIT_CODE. Restarting in 5 minutes..."
        sleep 3000  # 5 minutes
    fi
done
