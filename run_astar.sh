export TRANSFORMERS_CACHE='/trunk/ivanlee/hf_cache'
export HF_DATASETS_CACHE='/trunk/ivanlee/hf_cache'
export CUDA_VISIBLE_DEVICES=1

MODEL='facebook/blenderbot-400M-distill'
# MODEL='facebook/blenderbot-1B-distill'

# DATA="blended_skill_talk"
# DATA='meena'
# DATA='persona' # maybe get rid of first couple utts
# DATA='empath'
# DATA='wow'
DATA='otters'

DELIMIT='tab'
# DELIMIT='raw'

DEBUG=True
# DEBUG=False

STR=15
C=0

# for DATA in blended_skill_talk
# # for DATA in meena persona empath wow
# do
# for STR in 1 5 10 20 40 80 160
# do
# for C in 0 #1 2 3 4
# do
python astar.py \
    --model $MODEL \
    --dataset $DATA \
    --seed 0 \
    --debug $DEBUG \
    --split "test" \
    --do_astar True \
    --do_cosine False \
    --do_prompt False \
    --output_dir gen \
    --start_idx 0 \
    --delimiter $DELIMIT \
    --astar_top_k 40 \
    --astar_strength $STR \
    --c $C \
    --max_samples 250 \
    --cosine_mode 'extend' \
    --do_bst_truncate True \

# done
# done
# done

# for DATA in meena persona empath wow
# do
# python astar.py \
#     --model $MODEL \
#     --dataset $DATA \
#     --seed 0 \
#     --debug False \
#     --split "test" \
#     --do_astar False \
#     --do_cosine False \
#     --do_prompt True \
#     --output_dir gen \
#     --start_idx 0 \
#     --delimiter $DELIMIT \
#     --astar_top_k 40 \
#     --astar_strength 10 \
#     --c 2 \
#     --max_samples 50 \

# done

# python astar.py \
#     --model "facebook/blenderbot-400M-distill" \
#     --dataset "blended_skill_talk" \
#     --seed 0 \
#     --debug False \
#     --split "test" \
#     --do_astar False \
#     --do_cosine False \
#     --do_prompt True \
#     --output_dir gen \
#     --start_idx 0 \
#     --max_samples 50 \
#     --delimiter 'tab' \
#     --astar_top_k 40 \
#     --astar_strength 10 \
#     --c 2 \

# python astar.py \
#     --model "facebook/blenderbot-400M-distill" \
#     --dataset "blended_skill_talk" \
#     --seed 0 \
#     --debug True \
#     --split "test" \
#     --do_astar False \
#     --do_cosine False \
#     --do_prompt False \
#     --astar_strength 10 \
#     --astar_top_k 40 \
#     --output_dir gen \
#     --start_idx 5 \
#     --max_samples 50 \
#     --delimiter 'raw' \

# python astar.py \
#     --model $MODEL \
#     --dataset $DATA \
#     --seed 0 \
#     --debug True \
#     --split "test" \
#     --do_astar False \
#     --do_cosine False \
#     --do_prompt False \
#     --astar_strength 10 \
#     --astar_top_k 40 \
#     --output_dir gen \
#     --start_idx 5 \
#     --max_samples 50 \

# for TOPK in 40 20 10 60
# do
# for STR in 5 10 15 20
# do
# python astar.py \
#     --model "facebook/blenderbot-400M-distill" \
#     --dataset "blended_skill_talk" \
#     --seed 0 \
#     --debug False \
#     --split "test" \
#     --astar_strength $STR \
#     --astar_top_k $TOPK \
#     --output_dir gen \
#     --max_samples 50 \

# done
# done