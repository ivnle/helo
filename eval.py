# %%
from importlib.metadata import files
import pandas as pd
from datasets import load_metric
from dataclasses import dataclass, field
import torch
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, AutoTokenizer
import numpy as np
import os
from tqdm import tqdm
import json
import logging
import transformers
import sys
import jsonlines
import mauve 

SEP_TOK = '    '

logger = logging.getLogger(__name__)


@dataclass
class Arguments:
    experiment: str = field(default=None, metadata={
                            "help": "Which experiment to run."})
    output_dir: str = field(default=None, metadata={
                            "help": "Where to save the output."})
    model: str = field(default=None, metadata={"help": "Which model to use."})
    debug: bool = field(default=False, metadata={
                        "help": "Whether to run in debug mode."})


def compute_mauve(df, do_human=False):
    first_utt = df['first_utt']
    target_utt = df['target_utt']
    predictions = df['gold_utt'] if do_human else df['middle_utt']
    references = df['gold_utt']

    updated_pred = []
    updated_ref = []
    for (fu, p, tu) in zip(first_utt, predictions, target_utt):
        updated_pred.append([fu] + p + [tu])
    for (fu, p, tu) in zip(first_utt, references, target_utt):
        updated_ref.append([fu] + p + [tu])

    updated_pred = [' '.join(x) for x in updated_pred]
    updated_ref = [' '.join(x) for x in updated_ref]

    # print(updated_pred[0])
    # print(updated_ref[0])
    # foo
    
    mauve_output = mauve.compute_mauve(q_text=updated_pred, p_text=updated_ref)
    
    return {'mauve': round(mauve_output.mauve, 2),
            }

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
            # 'bleu_precisions': [round(x, 1) for x in results['precisions']]
            }


def compute_smooth(df, args, do_human=False):
    cp = args.model
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
            'cov': round(np.mean(dataset_cov), 2),
            'smooth_cov': round(np.mean(dataset_smooth_cov), 2),
            # 'std': round(np.mean(dataset_std), 2),
            'ppl_first': round(np.mean(ppl_first), 2),
            'ppl_last': round(np.mean(ppl_last), 2),
            # 'ppl_mid': round(np.mean(ppl_mid), 2)
            }


def bst(args):
    data = 'blended-skill-talk'
    files_to_eval = [
        f"gen/{data}_test_samples0:250_tab_beam_seed0.jsonl",
        f"gen/{data}_test_samples0:250_tab_beam_prompt_seed0.jsonl",
        f"gen/{data}_test_samples0:250_tab_cosine_str20_c3.0_top40_extend_seed0.jsonl",
        f"gen/{data}_test_samples0:250_tab_astar_str15_c0.0_top40_seed0.jsonl",
        f"gen/{data}_test_samples0:250_tab_astar_str5_c3.0_top40_seed0.jsonl",
    ]
    output = {}
    output['files_to_eval'] = files_to_eval
    output['expected_samples'] = 250
    output['do_human'] = True
    return output


def wow(args):
    data = 'wow'
    files_to_eval = [
        f"gen/{data}_test_samples0:250_tab_beam_seed0.jsonl",
        f"gen/{data}_test_samples0:250_tab_beam_prompt_seed0.jsonl",
        f"gen/{data}_test_samples0:250_tab_cosine_str20_c3.0_top40_extend_seed0.jsonl",
        f"gen/{data}_test_samples0:250_tab_astar_str15_c0.0_top40_seed0.jsonl",
        f"gen/{data}_test_samples0:250_tab_astar_str5_c3.0_top40_seed0.jsonl",
    ]
    output = {}
    output['files_to_eval'] = files_to_eval
    output['expected_samples'] = 250
    output['do_human'] = True
    return output


def empath(args):
    data = 'empath'
    files_to_eval = [
        f"gen/{data}_test_samples0:250_tab_beam_seed0.jsonl",
        f"gen/{data}_test_samples0:250_tab_beam_prompt_seed0.jsonl",
        f"gen/{data}_test_samples0:250_tab_cosine_str20_c3.0_top40_extend_seed0.jsonl",
        f"gen/{data}_test_samples0:250_tab_astar_str15_c0.0_top40_seed0.jsonl",
        f"gen/{data}_test_samples0:250_tab_astar_str5_c3.0_top40_seed0.jsonl",
    ]
    output = {}
    output['files_to_eval'] = files_to_eval
    output['expected_samples'] = 134
    output['do_human'] = True
    return output


def persona(args):
    data = 'persona'
    files_to_eval = [
        f"gen/{data}_test_samples0:250_tab_beam_seed0.jsonl",
        f"gen/{data}_test_samples0:250_tab_beam_prompt_seed0.jsonl",
        f"gen/{data}_test_samples0:250_tab_cosine_str20_c3.0_top40_extend_seed0.jsonl",
        f"gen/{data}_test_samples0:250_tab_astar_str15_c0.0_top40_seed0.jsonl",
        f"gen/{data}_test_samples0:250_tab_astar_str5_c3.0_top40_seed0.jsonl",
    ]
    output = {}
    output['files_to_eval'] = files_to_eval
    output['expected_samples'] = 250
    output['do_human'] = True
    return output


def meena(args):

    files_to_eval = [
        "gen/meena_test_samples0:250_tab_beam_seed0.jsonl",
        "gen/meena_test_samples0:250_tab_beam_prompt_seed0.jsonl",
        "gen/meena_test_samples0:250_tab_cosine_str20_c3.0_top40_extend_seed0.jsonl",
        "gen/meena_test_samples0:250_tab_astar_str15_c0.0_top40_seed0.jsonl",
        "gen/meena_test_samples0:250_tab_astar_str5_c3.0_top40_seed0.jsonl",
    ]
    output = {}
    output['files_to_eval'] = files_to_eval
    output['expected_samples'] = 250
    output['do_human'] = True
    return output


def combined(args):
    data = 'combined'
    files_to_eval = [
        f"gen/{data}_test_samples0:250_tab_beam_seed0.jsonl",
        f"gen/{data}_test_samples0:250_tab_beam_prompt_seed0.jsonl",
        f"gen/{data}_test_samples0:250_tab_cosine_str20_c3.0_top40_extend_seed0.jsonl",
        f"gen/{data}_test_samples0:250_tab_astar_str15_c0.0_top40_seed0.jsonl",
        f"gen/{data}_test_samples0:250_tab_astar_str5_c3.0_top40_seed0.jsonl",
    ]
    output = {}
    output['files_to_eval'] = files_to_eval
    output['expected_samples'] = 964
    output['do_human'] = True
    return output


def astar_hyper_search(args):
    # cosine hyperparameter sweep
    files_to_eval = []
    if args.debug:
        files_to_eval.append(
            f"gen/blended-skill-talk_test_samples0:250_tab_cosine_str5_c1.0_top40_extend_seed0.jsonl")
    else:
        for c in [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]:
            for lam in [5, 10, 15, 20, 25, 30]:
                files_to_eval.append(
                    f"gen/blended-skill-talk_test_samples0:50_tab_astar_str{lam}_c{c}_top40_seed0.jsonl")
    output = {}
    output['files_to_eval'] = files_to_eval
    output['expected_samples'] = 50
    output['do_human'] = False
    return output


def cosine_hyper_search(args):
    # cosine hyperparameter sweep
    files_to_eval = []
    if args.debug:
        files_to_eval.append(
            f"gen/blended-skill-talk_test_samples0:250_tab_cosine_str5_c1.0_top40_extend_seed0.jsonl")
    else:
        for c in [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]:
            for lam in [5, 10, 20, 40, 80, 160]:
                files_to_eval.append(
                    f"gen/blended-skill-talk_test_samples0:250_tab_cosine_str{lam}_c{c}_top40_extend_seed0.jsonl")
    output = {}
    output['files_to_eval'] = files_to_eval
    output['expected_samples'] = 250
    output['do_human'] = False
    return output


def main():
    # for i, f in enumerate(files_to_eval):
    #     df = pd.read_json(f, lines=True)
    #     if len(df) != 50:
    #         print('WARNING: {} has {} samples'.format(f, len(df)))
    #     print(f)
    #     print(compute_smooth(df, do_human=False))
    #     print()

    parser = transformers.HfArgumentParser(
        (Arguments))
    args = parser.parse_args_into_dataclasses()[0]

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if args.debug:
        logger.setLevel('DEBUG')
    logger.warning(args)

    if args.experiment == 'cosine-hyper-sweep':
        exp_cfg = cosine_hyper_search(args)
    elif args.experiment == 'blended-skill-talk':
        exp_cfg = bst(args)
    elif args.experiment == 'wow':
        exp_cfg = wow(args)
    elif args.experiment == 'astar_hyper_search':
        exp_cfg = astar_hyper_search(args)
    elif args.experiment == 'persona':
        exp_cfg = persona(args)
    elif args.experiment == 'meena':
        exp_cfg = meena(args)
    elif args.experiment == 'empath':
        exp_cfg = empath(args)
    elif args.experiment == 'combined':
        exp_cfg = combined(args)
    else:
        raise ValueError('Unknown experiment: {}'.format(args.experiment))

    files_to_eval = exp_cfg['files_to_eval']
    expected_samples = exp_cfg['expected_samples']
    do_human = exp_cfg['do_human']

    # Construct output filename
    fp = os.path.join(args.output_dir, args.experiment + '.jsonl')
    if not os.path.exists(fp):
        with open(fp, 'w') as f:
            pass
    logger.warning(f'Writing to {fp}')

    # results = []
    for i, f in enumerate(tqdm(files_to_eval)):
        df = pd.read_json(f, lines=True)
        if len(df) != expected_samples:
            logger.warning('WARNING: {} has {} samples'.format(f, len(df)))

        if do_human and i == 0:
            # print('human:', f.split('_')[0])
            out_human = {'file': 'human', 'bleu_score': 100, 'mauve': 1}
            out_human.update(compute_smooth(df, args, do_human=True))
            # results.append(out_human)
            with jsonlines.open(fp, mode='a') as writer:
                writer.write(out_human)

            print(out_human.pop('file'))
            print(out_human)

        out = {'file': f}
        out.update(compute_mauve(df))
        out.update(compute_bleu(df))
        out.update(compute_smooth(df, args))
        # results.append(out)

        with jsonlines.open(fp, mode='a') as writer:
            writer.write(out)

        # print(out.pop('file'))
        # print(out)

    # print(results)

    # with open(fp, 'w') as f:
    #     json.dump(results, f)


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
