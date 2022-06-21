# HeLo: Learning-Free Lookahead Decoding for Conversation Infilling

## Infilling Conversations
Use `run_astar.sh` to generate infilled conversations for various decoding strategies and datasets.

## Evaluation
Use `run_eval.sh` to run evaluation harness over the generated conversations

## Directories
- `datasets` contains the Meena chatlogs and where you should place the `datasets/wow_test_random_split.json` file downloaded form ParlAI for Wizard of Wikipedia. Other datasets are automatically downloaded via Huggingface when you run `run_eval.sh`.
- `gen` contains generations from running `run_astar.sh`. Includes generations used for the paper results and hyperparameter sweeps.
- `eval1b` contains the experiment results from the paper