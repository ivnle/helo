from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, AutoTokenizer
import torch
import datasets
from dataclasses import dataclass, field
import transformers
import logging
from tqdm import tqdm
import pandas as pd
import os
import sys
import jsonlines

SEP_TOK = '    '

logger = logging.getLogger(__name__)


@dataclass
class Arguments:
    debug: bool = field(default=None, metadata={"help": "Debug mode or not."})
    seed: int = field(default=None, metadata={"help": "Random seed."})
    model: str = field(default=None, metadata={"help": "Model to use."})
    dataset: str = field(default=None, metadata={"help": "Dataset to use."})
    split: str = field(default=None, metadata={"help": "Split to use."})
    astar_strength: int = field(default=None, metadata={
                                "help": "Strength of heuristic."})
    astar_top_k: int = field(default=None, metadata={"help": "Top k to use."})
    output_dir: str = field(default=None, metadata={
                            "help": "Output directory."})
    # trunk_dir: str = field(default=None, metadata={
    #                        "help": "Trunk directory for large files."})


def prepare_dataset(args):
    dataset = datasets.load_dataset(args.dataset)

    def process_dataset(examples):
        context = examples['previous_utterance']
        guided = examples['guided_messages']
        free = examples['free_messages']

        # context = [[c_.strip() for c_ in c] for c in context]
        conv = [[c_.strip() for c_ in c] for c in context]

        # context = [SEP_TOK.join(c) for c in context]

        # unguided (free) speaks first
        human_utt = []
        # conv_len = []
        for f, g in zip(free, guided):
            assert(len(f) == len(g))
            # single_conv = ''
            single_conv = []
            for _f, _g in zip(f, g):
                _f = _f.strip()
                _g = _g.strip()
                # single_conv += SEP_TOK.join([_f, _g]) + SEP_TOK
                single_conv.append(_f)
                single_conv.append(_g)
            human_utt.append(single_conv)
            # conv_len.append(len(f) + len(g))

        full_conv = []
        for c, h in zip(context, human_utt):
            full_conv.append(c + h)

        first_utt = [c[0] for c in full_conv]
        last_utt = [c[-1] for c in full_conv]
        between_utt = [c[1:-1] for c in full_conv]
        between_utt_len = [len(c) for c in between_utt]

        return {'full_conv': full_conv, 'first_utt': first_utt,
                'last_utt': last_utt, 'between_utt': between_utt, 'between_utt_len': between_utt_len}

    dataset = dataset.map(process_dataset, batched=True,
                          remove_columns=dataset['train'].column_names)

    return dataset


def main():
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

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformers.set_seed(args.seed)

    model = BlenderbotForConditionalGeneration.from_pretrained(args.model)
    model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.truncation_side = 'left'

    dataset = prepare_dataset(args)
    # print(dataset['test'][0]['full_conv'])
    # print(dataset['test'][0]['first_utt'])
    # print(dataset['test'][0]['last_utt'])
    # print(dataset['test'][0]['between_utt'])
    # print(dataset['test'][0]['between_utt_len'])
    # foo

    """
    Given first utterance, last utterance, and conversion length L , generate L-2 utterances that bridge the given utterances.
    To evaluate, take BLEU score between the concatenated L-2 human utterances and the concatenated L-2 generated utterances.
    """

    UTTERANCE = "My friends are cool but they eat too many carbs."

    target_utterances = ['The glory of the Roman empire is forever.',
                         'The beaches in San Diego are beautiful.',
                         'When you step on wet cement you can leave footprints.',
                         'Gondor calls for aid!',
                         'The best way to get to the moon is by spaceship.',
                         ]

    dataset = dataset[args.split]

    if args.debug:
        dataset = dataset.select(range(0, 2))
        args.astar_top_k = 5

    fp = os.path.join(args.output_dir, "gen")
    fp += f"_split{args.split}"
    fp += f"_strength{args.astar_strength}"
    fp += f"_topk{args.astar_top_k}"
    fp += f"_seed{args.seed}"
    fp += f"_debug" if args.debug else ""
    fp += '.jsonl'
    logger.warning(f"Writing generated utterances to {fp}")
    # if fp doesn't exist, create it    
    if not os.path.exists(fp):
        with open(fp, 'w') as f:
            pass

    # first_utts = []
    # target_utts = []
    # middle_utts = []
    # gold_utts = []

    for _, sample in enumerate(tqdm(dataset)):
        source_utt = sample['first_utt']
        target_utt = sample['last_utt']
        conv_len = sample['between_utt_len']

        logger.debug(f"Source: {source_utt}")
        logger.debug(f"Target: {target_utt}")
        logger.debug(f"Conv len: {conv_len}")

        conv_so_far = []
        conv_so_far.append(source_utt)

        target = tokenizer([target_utt], return_tensors="pt").to(
            args.device).input_ids
        for i in range(conv_len):

            conv_so_far_str = SEP_TOK.join(conv_so_far)
            inputs = tokenizer([conv_so_far_str], truncation=True, return_tensors="pt").to(
                args.device).input_ids

            logger.debug(
                f"History {i}: {repr(tokenizer.batch_decode(inputs, skip_special_tokens=False)[0])}")

            reply_ids = model.generate(input_ids=inputs,
                                       # decoder_input_ids=decoder_input_ids,
                                       num_beams=3,
                                       do_astar=True,
                                       target_utterance=target,
                                       astar_strength=args.astar_strength,
                                       astar_top_k=args.astar_top_k,
                                       )

            # logger.debug(f"Utt {i}: {repr(tokenizer.batch_decode(reply_ids, skip_special_tokens=False)[0])}")

            response = tokenizer.batch_decode(
                reply_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
            conv_so_far.append(response.strip())

        output = {"first_utt": source_utt,
                  "target_utt": target_utt,
                  "middle_utt": conv_so_far[1:],
                  "gold_utt": sample['between_utt']
                  }
        with jsonlines.open(fp, mode='a') as writer:
            # writer.write(tdauve_output.__dict__)
            writer.write(output)
        # first_utts.append(source_utt)
        # middle_utts.append(conv_so_far[1:])
        # gold_utts.append(sample['between_utt'])
        # target_utts.append(target_utt)

    # to dataframe
    # df = pd.DataFrame({"first_utt": first_utts, "target_utt": target_utts,
    #                   "middle_utt": middle_utts, "gold_utt": gold_utts})
    # save to pkl

    # df.to_pickle(fp)
    


if __name__ == "__main__":
    main()
