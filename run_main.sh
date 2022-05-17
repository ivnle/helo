export TRANSFORMERS_CACHE='/trunk/ivanlee/hf_cache'
export CUDA_VISIBLE_DEVICES=3

# MODEL='facebook/blenderbot-400M-distill'
# OUTPUT_DIR="/home/ivanlee/npc/mauve4dialog/gen/"

python main.py \
    --seed 0 \
    --debug True \
    --trunk_dir "/trunk/ivanlee/cyoa/" \
