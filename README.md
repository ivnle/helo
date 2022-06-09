# cyoa

- Important Files
    - astar.py, run_astar.sh : entry point for astar inspired utterance steering.

- Notes
    - store large files at `/trunk/ivanlee/cyoa`
    - raw, str=10 produces good Roman Empire output

- TODO            
    - in progress
        - run_astar_strc_gen2.sh
        - run_astar_strc_gen.sh
        - run_astar_topk_gen.sh
    
    - implement transition smoothness ppl metric
        - avg( avg (avg ppl of utterance) of conversation) of dataset)
        - sd(diff(y))/abs(mean(diff(y)))
        - avg ppl  of last utterance
        - avg ppl of first utterance
    
    - compute eval stats over full human dataset bst and meena
    

- Done
    - pick a delimiter strategy and stick with it: tabs or raw output from generate, lean towards latter for simplicity
        - right now, we use tabs in outter loop and raw in inner loop
    - write context for each generation to file
    - tmux session `cyoa_gen` generating raw delimiter in dialog history, compare against tab delimited when done.
    - tmux sessions `cyoa_tab_gen` tab delimiter to sanity check that it matches `gen/gen_split-test_samples50_strength10_topk40_seed0.jsonl`
    - annealing strategy for heuristic strength, see K2T paper: exponential schedule based on current time step relative to budget
    - sanity check beam score works with stock transformers lib
    - vocab embedding steering
        - get vocab dict, average over embeddings of target utterance
        - cosine simlarity between embedded target utterance 
        - tokens with higher cosine similarity with embedded target uttrance get more weight
    - test that dev transformers works
    - test that HF cache works
    - env variable for hf cache   
    - install dev version of huggingface    
    - install transformers
    - download backwards and forwards dialogpt models
    

- Shelved
    - figure out how dialogpt MMI decoding works, read source code