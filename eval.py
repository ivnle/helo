#%%
import pandas as pd
from datasets import load_metric
metric = load_metric("bleu")

#%%
# read jsonlines
for n_gram in [1, 2, 3, 4]:
    for strength in [5, 10, 15]:
        df = pd.read_json(f"gen/gen_split-test_samples50_strength{strength}_topk40_seed0.jsonl", lines=True)
        predictions = df['middle_utt']
        references = df['gold_utt']

        predictions = [' '.join(x) for x in predictions]
        references = [' '.join(x) for x in references]
        predictions = [x.split() for x in predictions]
        references_0 = [[x.split()] for x in references]
        references_1 = [x.split() for x in references]

        bleu = load_metric("bleu")
        results = bleu.compute(predictions=predictions, references=references_0, max_order=n_gram)
        print(f"strength {strength} n-gram {n_gram}: {results['bleu']}")

#%%
for n_gram in [1, 2, 3, 4]:
    df = pd.read_json(f"gen/gen_beam_split-test_samples50_seed0.jsonl", lines=True)
    predictions = df['middle_utt']
    references = df['gold_utt']

    predictions = [' '.join(x) for x in predictions]
    references = [' '.join(x) for x in references]
    predictions = [x.split() for x in predictions]
    references_0 = [[x.split()] for x in references]
    references_1 = [x.split() for x in references]

    bleu = load_metric("bleu")
    results = bleu.compute(predictions=predictions, references=references_0, max_order=n_gram)
    print(f"n-gram {n_gram}: {results['bleu']}")

#%%
# sacrebleu
for strength in [5, 10, 15, 20]:
    df = pd.read_json(f"gen/gen_split-test_samples50_strength{strength}_topk40_seed0.jsonl", lines=True)
    predictions = df['middle_utt']
    references = df['gold_utt']
    predictions = [' '.join(x) for x in predictions]
    references = [[' '.join(x)] for x in references]
    print(predictions[0])
    print(references[0])

    sacrebleu = load_metric("sacrebleu")
    results = sacrebleu.compute(predictions=predictions, 
                                references=references)
    print(f"strength {strength}: {round(results['score'],1)} {[round(x,1) for x in results['precisions']]}")
    print()

#%%
# df = pd.read_json(f"gen/gen_beam_split-test_samples50_seed0.jsonl", lines=True)
df = pd.read_json(f"gen/gen_beam_do-prompt_split-test_samples50_seed0.jsonl", lines=True)
predictions = df['middle_utt']
references = df['gold_utt']
predictions = [' '.join(x) for x in predictions]
references = [[' '.join(x)] for x in references]
print(predictions[3])
print(references[3])

sacrebleu = load_metric("sacrebleu")
results = sacrebleu.compute(predictions=predictions, 
                            references=references)
print(f"Score: {round(results['score'],1)} {[round(x,1) for x in results['precisions']]}")

#%%
# PPL of seeing the next utterance.


