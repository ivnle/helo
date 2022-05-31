export TRANSFORMERS_CACHE='/trunk/ivanlee/hf_cache'
export HF_DATASETS_CACHE='/trunk/ivanlee/hf_cache'
export CUDA_VISIBLE_DEVICES=2

MODEL='facebook/blenderbot-400M-distill'
# OUTPUT_DIR="/home/ivanlee/npc/mauve4dialog/gen/"

python astar.py \
    --model "facebook/blenderbot-400M-distill" \
    --dataset "blended_skill_talk" \
    --seed 0 \
    --debug False \
    --split "test" \
    --astar_strength 10 \
    --astar_top_k 40 \
    --output_dir gen \
