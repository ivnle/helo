# HeLo: Learning-Free Lookahead Decoding for Conversation Infilling

## Infilling Conversations
Use `run_astar.sh` to generate infilled conversations for various decoding strategies and datasets.

## Evaluation
Use `run_eval.sh` to run evaluation harness over the generated conversations

## Directories
- `datasets` contains the Meena chatlogs and where you should place the `datasets/wow_test_random_split.json` file downloaded form ParlAI for Wizard of Wikipedia. Other datasets are downloaded via Huggingface.
- `gen` contains generations from running `run_astar.sh`.
- `eval1b` contains the experiment results from the paper