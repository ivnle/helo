MODEL='facebook/blenderbot-1B-distill'

# EXP='astar_hyper_search'
# EXP='empath'
# EXP='persona'
# EXP='meena'
# EXP='wow'
# EXP='blended-skill-talk'
EXP='combined'

DEBUG=False
# DEBUG=True

DIR=eval_1b

python eval.py \
    --model $MODEL \
    --output_dir $DIR \
    --experiment $EXP \
    --debug $DEBUG \
    --do_mauve True \