MODEL='facebook/blenderbot-400M-distill'

# Which dataset to generate over
DATA="blended_skill_talk"
# DATA='meena'
# DATA='persona'
# DATA='empath'
# DATA='wow'

# How to delimit the dialogue history. We use 'tab' in the experiments.
DELIMIT='tab'
# DELIMIT='raw'

# DEBUG=True
DEBUG=False

# Annealing schedule hyperparameters
STR=15 # lambda in the paper
C=0

# Runs HeLo
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

# Runs beam search baseline
python astar.py \
    --model $MODEL \
    --dataset $DATA \
    --seed 0 \
    --debug $DEBUG \
    --split "test" \
    --do_astar False \
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

# Runs prefix + beam baseline
python astar.py \
    --model $MODEL \
    --dataset $DATA \
    --seed 0 \
    --debug $DEBUG \
    --split "test" \
    --do_astar False \
    --do_cosine False \
    --do_prompt True \
    --output_dir gen \
    --start_idx 0 \
    --delimiter $DELIMIT \
    --astar_top_k 40 \
    --astar_strength $STR \
    --c $C \
    --max_samples 250 \
    --cosine_mode 'extend' \
    --do_bst_truncate True \

# Runs CoSim baseline
python astar.py \
    --model $MODEL \
    --dataset $DATA \
    --seed 0 \
    --debug $DEBUG \
    --split "test" \
    --do_astar False \
    --do_cosine True \
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

    
