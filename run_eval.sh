export TRANSFORMERS_CACHE='/trunk/ivanlee/hf_cache'
export HF_DATASETS_CACHE='/trunk/ivanlee/hf_cache'
export CUDA_VISIBLE_DEVICES=3

# MODEL='facebook/blenderbot-400M-distill'
MODEL='facebook/blenderbot-1B-distill'

# EXP='astar_hyper_search'
# EXP='empath'
# EXP='persona'
# EXP='meena'
# EXP='wow'
# EXP='blended-skill-talk'
EXP='combined'
# EXP='otters'

DEBUG=False
# DEBUG=True

# DIR=eval
# DIR=eval_mauve
# DIR=debug
DIR=eval_1b

# for EXP in empath persona meena wow blended-skill-talk

# do
python eval.py \
    --model $MODEL \
    --output_dir $DIR \
    --experiment $EXP \
    --debug $DEBUG \
    --do_mauve True \

# done

# python eval.py \
#     --model $MODEL \
#     --output_dir eval \
#     --experiment 'meena' \
#     --debug False \
