#%%
import pandas as pd
from datasets import load_metric
import torch
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, AutoTokenizer
import numpy as np


def compute_avg_ppl(df):
    cp ='facebook/blenderbot-400M-distill'
    model = BlenderbotForConditionalGeneration.from_pretrained(cp)
    model.to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(cp)
    tokenizer.truncation_side = 'left'

    first_utt = df['first_utt']
    predictions = df['middle_utt']
    # predictions = df['gold_utt']
    target_utt = df['target_utt']


    first_utt = first_utt.apply(lambda x: [x])
    target_utt = target_utt.apply(lambda x: [x])

    # concatenate first_utt, predictions, target_utt

    df['concat_gen'] = first_utt + predictions + target_utt
    df['concat_gen'].iloc[0]

    conversations = df['concat_gen'].to_list()
    SEP_TOK = '    '
    dataset_ppl = []
    conversation_variance = []
    for i, c in enumerate(conversations[:3]):
        conv_ppl = []        
        for j, utt in enumerate(c[1:], 1):        
            dh_str = SEP_TOK.join(c[:j])        
            inputs = tokenizer([dh_str], truncation=True, return_tensors="pt").to('cuda').input_ids
            labels = tokenizer([utt], truncation=True, return_tensors="pt").to('cuda').input_ids
            with torch.no_grad():
                output = model(input_ids=inputs, labels=labels, return_dict=True)
            neg_log_like = output['loss'].mean()
            utt_ppl = torch.exp(neg_log_like)
            conv_ppl.append(utt_ppl.item())
        
        dataset_ppl.append(np.mean(conv_ppl))
        conversation_variance.append(np.var(conv_ppl))

    avg_dataset_ppl = np.mean(dataset_ppl)
    avg_conversation_variance = np.mean(conversation_variance)
    return avg_dataset_ppl, avg_conversation_variance

# df = pd.read_json(f"gen/gen_beam_do-prompt_split-test_samples50_seed0.jsonl", lines=True)
df = pd.read_json(f"gen/gen_beam_split-test_samples50_seed0.jsonl", lines=True)
strength=10
# df = pd.read_json(f"gen/gen_split-test_samples50_strength{strength}_topk40_seed0.jsonl", lines=True)    
print(compute_avg_ppl(df)) # TODO maybe normalized variance?
        



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


