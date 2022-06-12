# %%
import pandas as pd
from datasets import load_metric
import torch
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, AutoTokenizer
import numpy as np
import os

SEP_TOK = '    '

os.environ["HF_DATASETS_CACHE"] = '/trunk/ivanlee/hf_cache'
os.environ["TRANSFORMERS_CACHE"] = '/trunk/ivanlee/hf_cache'

def compute_bleu(df):
    predictions = df['middle_utt']
    references = df['gold_utt']
    predictions = [' '.join(x) for x in predictions]
    references = [[' '.join(x)] for x in references]
    # print(predictions[0])
    # print(references[0])

    sacrebleu = load_metric("sacrebleu")
    results = sacrebleu.compute(predictions=predictions,
                                references=references)

    return {'bleu_score': round(results['score'], 2),
            'bleu_precisions': [round(x, 1) for x in results['precisions']]}


def compute_smooth(df, do_human=False):
    # cp = 'facebook/blenderbot-400M-distill'
    cp = "facebook/blenderbot-1B-distill"
    model = BlenderbotForConditionalGeneration.from_pretrained(cp)
    model.to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(cp)    

    first_utt = df['first_utt']
    predictions = df['gold_utt'] if do_human else df['middle_utt']
    # predictions = df['gold_utt']
    target_utt = df['target_utt']

    first_utt = first_utt.apply(lambda x: [x])
    target_utt = target_utt.apply(lambda x: [x])

    # concatenate first_utt, predictions, target_utt

    df['concat_gen'] = first_utt + predictions + target_utt
    df['concat_gen'].iloc[0]

    conversations = df['concat_gen'].to_list()

    dataset_ppl = []
    dataset_cov = []
    dataset_smooth_cov = []
    dataset_std = []
    ppl_first = []
    ppl_last = []
    ppl_mid = []

    for conv_idx, conv in enumerate(conversations):
        conv_ppls = []
        for utt_idx, utt in enumerate(conv[1:], 1):
            dh_str = SEP_TOK.join(conv[:utt_idx])

            tokenizer.truncation_side = 'left'
            inputs = tokenizer([dh_str], truncation=True,
                               return_tensors="pt").to('cuda').input_ids

            tokenizer.truncation_side = 'right'
            labels = tokenizer([utt], truncation=True,
                               return_tensors="pt").to('cuda').input_ids
            with torch.no_grad():
                output = model(input_ids=inputs,
                               labels=labels, return_dict=True)
            neg_log_like = output['loss'].mean()
            utt_ppl = torch.exp(neg_log_like)
            conv_ppls.append(utt_ppl.item())

            # if utt_ppl.item() > 500 or utt_ppl.item() < 3:
            # # if utt == '':
            #     print('dh:', dh_str)
            #     print('utt:', utt)
            #     print(utt_ppl.item())

        dataset_ppl.append(np.mean(conv_ppls))

        diff_conv_ppls = np.abs(np.diff(conv_ppls))
        # assert(len(conv_ppls) == len(diff_conv_ppls) + 1)
        # print(conv_ppls)
        # print(diff_conv_ppls)

        # coefficient of variation, lower is better
        smoothness = np.std(diff_conv_ppls) / np.abs(np.mean(diff_conv_ppls))
        dataset_smooth_cov.append(smoothness)
        if np.mean(smoothness) > 500:
            print('dh:', conv)
            print('utt:', utt)
            print(utt_ppl.item())
            print(smoothness)
            print([round(c) for c in conv_ppls])
            print(diff_conv_ppls)
            print(np.std(diff_conv_ppls))
            print(np.abs(np.mean(diff_conv_ppls)))
            print()

        dataset_cov.append(np.std(conv_ppls) / np.abs(np.mean(conv_ppls)))
        dataset_std.append(np.std(conv_ppls))

        ppl_first.append(conv_ppls[0])
        ppl_last.append(conv_ppls[-1])
        ppl_mid.append(np.mean(conv_ppls[1:-1]))

    return {'ppl': round(np.mean(dataset_ppl), 2),
            'smooth_cov': round(np.mean(dataset_smooth_cov), 2),
            'cov': round(np.mean(dataset_cov), 2),
            'std': round(np.mean(dataset_std), 2),
            'ppl_first': round(np.mean(ppl_first), 2),
            'ppl_last': round(np.mean(ppl_last), 2),
            'ppl_mid': round(np.mean(ppl_mid), 2)
            }


def main():

    files_to_eval = [
        # "gen/blended-skill-talk_test_samples0:50_raw_beam_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_raw_beam_prompt_seed0.jsonl",
        
        "gen/blended-skill-talk_test_samples0:50_tab_astar_str10_c0.0_top5_seed0.jsonl",
        "gen/blended-skill-talk_test_samples0:50_tab_astar_str10_c0.0_top10_seed0.jsonl",
        "gen/blended-skill-talk_test_samples0:50_tab_astar_str10_c0.0_top20_seed0.jsonl",
        "gen/blended-skill-talk_test_samples0:50_tab_astar_str10_c0.0_top40_seed0.jsonl",
        "gen/blended-skill-talk_test_samples0:50_tab_astar_str10_c0.0_top80_seed0.jsonl",
        
        # "gen/blended-skill-talk_test_samples0:50_tab_astar_str5_c0.0_top40_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_astar_str10_c0.0_top40_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_astar_str15_c0.0_top40_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_astar_str20_c0.0_top40_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_astar_str25_c0.0_top40_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_astar_str30_c0.0_top40_seed0.jsonl",

        # "gen/blended-skill-talk_test_samples0:50_tab_astar_str5_c-1.0_top40_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_astar_str10_c-1.0_top40_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_astar_str15_c-1.0_top40_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_astar_str20_c-1.0_top40_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_astar_str25_c-1.0_top40_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_astar_str30_c-1.0_top40_seed0.jsonl",

        # "gen/blended-skill-talk_test_samples0:50_tab_astar_str5_c1.0_top40_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_astar_str10_c1.0_top40_seed0.jsonl",
        
        # "gen/blended-skill-talk_test_samples0:50_tab_astar_str5_c2.0_top40_seed0.jsonl",        
        # "gen/blended-skill-talk_test_samples0:50_tab_astar_str10_c2.0_top40_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_astar_str15_c2.0_top40_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_astar_str20_c2.0_top40_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_astar_str25_c2.0_top40_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_astar_str30_c2.0_top40_seed0.jsonl",

        # "gen/blended-skill-talk_test_samples0:50_tab_astar_str5_c1.0_top40_seed0.jsonl",        
        # "gen/blended-skill-talk_test_samples0:50_tab_astar_str10_c1.0_top40_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_astar_str15_c1.0_top40_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_astar_str20_c1.0_top40_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_astar_str25_c1.0_top40_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_astar_str30_c1.0_top40_seed0.jsonl",

        # """ Data sets, c2, str10 vs baselines """
        # "gen/blended-skill-talk_test_samples0:50_tab_beam_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_beam_prompt_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_cosine_str50_c0.0_top40_extend_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_astar_str10_c1.0_top40_seed0.jsonl",

        # "gen/persona_test_samples0:50_tab_beam_seed0.jsonl",
        # "gen/persona_test_samples0:50_tab_beam_prompt_seed0.jsonl",
        # "gen/persona_test_samples0:50_tab_cosine_str50_c0.0_top40_extend_seed0.jsonl",
        # "gen/persona_test_samples0:50_tab_astar_str10_c2.0_top40_seed0.jsonl",
        
        # "gen/meena_test_samples0:50_tab_beam_seed0.jsonl",
        # "gen/meena_test_samples0:50_tab_beam_prompt_seed0.jsonl",
        # "gen/meena_test_samples0:50_tab_cosine_str50_c0.0_top40_extend_seed0.jsonl",
        # "gen/meena_test_samples0:50_tab_astar_str10_c2.0_top40_seed0.jsonl",
        
        # "gen/empath_test_samples0:50_tab_beam_seed0.jsonl",
        # "gen/empath_test_samples0:50_tab_beam_prompt_seed0.jsonl",
        # "gen/empath_test_samples0:50_tab_cosine_str50_c0.0_top40_extend_seed0.jsonl",
        # "gen/empath_test_samples0:50_tab_astar_str10_c2.0_top40_seed0.jsonl",
        
        # "gen/wow_test_samples0:50_tab_beam_seed0.jsonl",
        # "gen/wow_test_samples0:50_tab_beam_prompt_seed0.jsonl",
        # "gen/wow_test_samples0:50_tab_cosine_str50_c0.0_top40_extend_seed0.jsonl",
        # "gen/wow_test_samples0:50_tab_astar_str10_c2.0_top40_seed0.jsonl",

        # cosine extend-pos test
        # "gen/blended-skill-talk_test_samples0:50_tab_cosine_str20_c0.0_top40_extend-pos_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_cosine_str40_c0.0_top40_extend-pos_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_cosine_str80_c0.0_top40_extend-pos_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_cosine_str160_c0.0_top40_extend-pos_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_cosine_str200_c0.0_top40_extend-pos_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_cosine_str240_c0.0_top40_extend-pos_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_cosine_str280_c0.0_top40_extend-pos_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_cosine_str320_c0.0_top40_extend-pos_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_cosine_str420_c0.0_top40_extend-pos_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_cosine_str520_c0.0_top40_extend-pos_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_cosine_str620_c0.0_top40_extend-pos_seed0.jsonl",

        # """ cosine str test """
        # "gen/blended-skill-talk_test_samples0:50_tab_cosine_str10_c0.0_top40_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_cosine_str20_c0.0_top40_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_cosine_str40_c0.0_top40_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_cosine_str50_c0.0_top40_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_cosine_str60_c0.0_top40_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_cosine_str70_c0.0_top40_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_cosine_str80_c0.0_top40_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_cosine_str90_c0.0_top40_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_cosine_str110_c0.0_top40_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_cosine_str130_c0.0_top40_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_cosine_str150_c0.0_top40_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_cosine_str160_c0.0_top40_seed0.jsonl",        

        # "gen/blended-skill-talk_test_samples0:50_tab_beam_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_beam_prompt_seed0.jsonl",
        # "debug/blended-skill-talk_test_samples0:50_tab_astar_str5_c0.0_top40_seed0.jsonl",
        # "debug/blended-skill-talk_test_samples0:50_tab_astar_str10_c0.0_top40_seed0.jsonl",
        # "debug/blended-skill-talk_test_samples0:50_tab_astar_str15_c0.0_top40_seed0.jsonl",
        # "debug/blended-skill-talk_test_samples0:50_tab_astar_str20_c0.0_top40_seed0.jsonl",
        
        
        # 'gen/meena_test_samples0:50_tab_beam_seed0.jsonl',
        # "/home/ivanlee/cyoa/gen/meena_test_samples0:50_tab_beam_prompt_seed0.jsonl",

        # "gen/blended-skill-talk_test_samples0:980_tab_beam_seed0.jsonl",
        # "gen/gen_beam_split-test_samples50_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_beam_seed0.jsonl",
        # "gen/gen_beam_do-prompt_split-test_samples50_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_beam_prompt_seed0.jsonl",

        # "debug/gen_split-test_samples50_strength5_topk40_seed0.jsonl",
        # "debug/gen_split-test_samples50_strength10_topk40_seed0.jsonl",
        # "debug/gen_split-test_samples50_strength15_topk40_seed0.jsonl",
        # "debug/gen_split-test_samples50_strength20_topk40_seed0.jsonl",

        # "gen/blended-skill-talk_test_samples0:50_tab_astar_str5_c2.0_top40_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_astar_str10_c2.0_top40_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_tab_astar_str15_c2.0_top40_seed0.jsonl",
        # "gen/gen_astar_delimit-tab_split-test_samples50_strength5_c2.0_topk40_seed0.jsonl",
        # "gen/gen_astar_delimit-tab_split-test_samples50_strength10_topk40_seed0.jsonl"
        # "gen/blended-skill-talk_test_samples0:50_raw_beam_seed0.jsonl",
        # "gen/blended-skill-talk_test_samples0:50_raw_beam_prompt_seed0.jsonl",
        # "gen/gen_astar_delimit-raw_split-test_samples50_strength10_topk40_seed0.jsonl",

        # "/home/ivanlee/cyoa/gen/blended-skill-talk_test_samples0:50_tab_beam_seed0.jsonl",
        # "/home/ivanlee/cyoa/gen/empath_test_samples0:50_tab_beam_seed0.jsonl",
        # "/home/ivanlee/cyoa/gen/meena_test_samples0:50_tab_beam_seed0.jsonl",
        # "/home/ivanlee/cyoa/gen/persona_test_samples0:50_tab_beam_seed0.jsonl",
        # "/home/ivanlee/cyoa/gen/wow_test_samples0:50_tab_beam_seed0.jsonl",

    ]

    # for i, f in enumerate(files_to_eval):
    #     df = pd.read_json(f, lines=True)
    #     if len(df) != 50:
    #         print('WARNING: {} has {} samples'.format(f, len(df)))
    #     print(f)
    #     print(compute_smooth(df, do_human=False))
    #     print()

    for i, f in enumerate(files_to_eval):
        df = pd.read_json(f, lines=True)
        if len(df) != 50:
            print('WARNING: {} has {} samples'.format(f, len(df)))

        # if i % 4 == 0:
        if i == 0:
            print('human:', f.split('_')[0])
            print(compute_smooth(df, do_human=True))
            print()

        print(f)
        print(compute_bleu(df))
        print(compute_smooth(df))
        print()


if __name__ == "__main__":
    main()


# # %%
# # read jsonlines
# for n_gram in [1, 2, 3, 4]:
#     for strength in [5, 10, 15]:
#         df = pd.read_json(
#             f"gen/gen_split-test_samples50_strength{strength}_topk40_seed0.jsonl", lines=True)
#         predictions = df['middle_utt']
#         references = df['gold_utt']

#         predictions = [' '.join(x) for x in predictions]
#         references = [' '.join(x) for x in references]
#         predictions = [x.split() for x in predictions]
#         references_0 = [[x.split()] for x in references]
#         references_1 = [x.split() for x in references]

#         bleu = load_metric("bleu")
#         results = bleu.compute(predictions=predictions,
#                                references=references_0, max_order=n_gram)
#         print(f"strength {strength} n-gram {n_gram}: {results['bleu']}")

# # %%
# for n_gram in [1, 2, 3, 4]:
#     df = pd.read_json(
#         f"gen/gen_beam_split-test_samples50_seed0.jsonl", lines=True)
#     predictions = df['middle_utt']
#     references = df['gold_utt']

#     predictions = [' '.join(x) for x in predictions]
#     references = [' '.join(x) for x in references]
#     predictions = [x.split() for x in predictions]
#     references_0 = [[x.split()] for x in references]
#     references_1 = [x.split() for x in references]

#     bleu = load_metric("bleu")
#     results = bleu.compute(predictions=predictions,
#                            references=references_0, max_order=n_gram)
#     print(f"n-gram {n_gram}: {results['bleu']}")

# # %%
# # sacrebleu
# for strength in [5, 10, 15, 20]:
#     df = pd.read_json(
#         f"gen/gen_split-test_samples50_strength{strength}_topk40_seed0.jsonl", lines=True)
#     predictions = df['middle_utt']
#     references = df['gold_utt']
#     predictions = [' '.join(x) for x in predictions]
#     references = [[' '.join(x)] for x in references]
#     print(predictions[0])
#     print(references[0])

#     sacrebleu = load_metric("sacrebleu")
#     results = sacrebleu.compute(predictions=predictions,
#                                 references=references)
#     print(
#         f"strength {strength}: {round(results['score'],1)} {[round(x,1) for x in results['precisions']]}")
#     print()

# # %%
# # df = pd.read_json(f"gen/gen_beam_split-test_samples50_seed0.jsonl", lines=True)
# df = pd.read_json(
#     f"gen/gen_beam_do-prompt_split-test_samples50_seed0.jsonl", lines=True)
# predictions = df['middle_utt']
# references = df['gold_utt']
# predictions = [' '.join(x) for x in predictions]
# references = [[' '.join(x)] for x in references]
# print(predictions[3])
# print(references[3])

# sacrebleu = load_metric("sacrebleu")
# results = sacrebleu.compute(predictions=predictions,
#                             references=references)
# print(
#     f"Score: {round(results['score'],1)} {[round(x,1) for x in results['precisions']]}")

# # %%
# # PPL of seeing the next utterance.
