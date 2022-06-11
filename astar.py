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
import math
import demoji

SEP_TOK = '    '

logger = logging.getLogger(__name__)


@dataclass
class Arguments:
    debug: bool = field(default=None, metadata={"help": "Debug mode or not."})
    seed: int = field(default=None, metadata={"help": "Random seed."})
    model: str = field(default=None, metadata={"help": "Model to use."})
    dataset: str = field(default=None, metadata={"help": "Dataset to use."})
    split: str = field(default=None, metadata={"help": "Split to use."})
    do_astar: bool = field(default=False, metadata={"help": "Do A* or not."})
    do_cosine: bool = field(default=False, metadata={"help": "Do cosine or not."})
    do_prompt: bool = field(default=False, metadata={"help": "Do prompt or not."})
    astar_strength: int = field(default=None, metadata={
                                "help": "Starting strength of heuristic. Outside exponential."})
    astar_top_k: int = field(default=None, metadata={"help": "Top k to use."})
    output_dir: str = field(default=None, metadata={
                            "help": "Output directory."})    
    max_samples: int = field(default=None, metadata={"help": "Max samples to use."})
    start_idx: int = field(default=0, metadata={"help": "Start index."})    
    delimiter: str = field(default='tab', metadata={"help": "Delimiter to use. Choices = [tab, raw]"})
    # do_anneal: bool = field(default=False, metadata={"help": "Do annealing or not."})
    c: float = field(default=0, metadata={"help": "c for annealing. inside exponential. set to 0 for no annealing"})
    beam_size: int = field(default=3, metadata={"help": "Beam size."})
    min_conv_len: int = field(default=6, metadata={"help": "Min conversation length."})
    max_conv_len: int = field(default=8, metadata={"help": "Max conversation length."})
    # lam: float = field(default=None, metadata={"help": "Lambda for annealing. outside exponential"})
    # trunk_dir: str = field(default=None, metadata={
    #                        "help": "Trunk directory for large files."})

def prepare_empath_dataset(min_conv_len=6, max_conv_len=8):
    dataset = datasets.load_dataset("empathetic_dialogues")
    conversations = dataset['test']
    print(len(conversations))
    conversations = conversations.map(lambda examples: {'utt': [[x.replace('_comma_', ',')] for x in examples['utterance']]}, batched=True)
    # conversations = conversations.map(lambda examples: {'utt': [[x] for x in examples['utterance']]}, batched=True)
    df = conversations.to_pandas()
    # convert df['utt'] to lists
    df['utt'] = df['utt'].apply(lambda x: x.tolist())

    df = df.groupby('conv_id').agg({'utt': 'sum'})
    # convert df to list
    conversations = df['utt'].to_list()
    
    # Filter out conversations that are too short
    conversations = [conv for conv in conversations if len(conv) >= min_conv_len]
    # Truncate conversations that are too long
    conversations = [conv[:max_conv_len] for conv in conversations]
    
    full_conv = conversations
    first_utt = [conv[0] for conv in conversations]
    last_utt = [conv[-1] for conv in conversations]
    between_utt  = [conv[1:-1] for conv in conversations]
    between_utt_len = [len(utt) for utt in between_utt]
    
    d = {'full_conv': full_conv, 'first_utt': first_utt,
                'last_utt': last_utt, 'between_utt': between_utt, 'between_utt_len': between_utt_len}    
    
    dataset = datasets.Dataset.from_dict(d)
    dataset = datasets.DatasetDict({'test': dataset})

    return dataset

def prepare_personachat_dataset(min_conv_len=6, max_conv_len=8):
    dataset = datasets.load_dataset("bavard/personachat_truecased")
    conversations = dataset['validation']
    df = conversations.to_pandas()
    conv_map = {}
    for i in range(len(df)):    
        row = df.iloc[i]
        cid = str(row['conv_id'])

        if cid in conv_map:
            if len(row['history']) > len(conv_map[cid]):
                conv_map[cid] = row['history'].tolist()
        else:
            conv_map[cid] = row['history'].tolist()

    # conv_map to list
    conversations = [conv_map[cid] for cid in conv_map]

    # Filter out conversations that are too short
    conversations = [conv for conv in conversations if len(conv) >= min_conv_len]
    # Truncate conversations that are too long
    conversations = [conv[:max_conv_len] for conv in conversations]
    
    full_conv = conversations
    first_utt = [conv[0] for conv in conversations]
    last_utt = [conv[-1] for conv in conversations]
    between_utt  = [conv[1:-1] for conv in conversations]
    between_utt_len = [len(utt) for utt in between_utt]
    
    d = {'full_conv': full_conv, 'first_utt': first_utt,
                'last_utt': last_utt, 'between_utt': between_utt, 'between_utt_len': between_utt_len}    
    
    dataset = datasets.Dataset.from_dict(d)
    dataset = datasets.DatasetDict({'test': dataset})

    return dataset

def prepare_wow_dataset(min_conv_len=6, max_conv_len=8):
    df = pd.read_json('/home/ivanlee/cyoa/datasets/wow_test_random_split.json')
    conversations_with_meta = df['dialog']
    conversations = []
    for conv in conversations_with_meta:
        only_conv = [utt['text'] for utt in conv]
        conversations.append(only_conv)

    # Filter out conversations that are too short
    conversations = [conv for conv in conversations if len(conv) >= min_conv_len]
    # Truncate conversations that are too long
    conversations = [conv[:max_conv_len] for conv in conversations]
    
    full_conv = conversations
    first_utt = [conv[0] for conv in conversations]
    last_utt = [conv[-1] for conv in conversations]
    between_utt  = [conv[1:-1] for conv in conversations]
    between_utt_len = [len(utt) for utt in between_utt]
    
    d = {'full_conv': full_conv, 'first_utt': first_utt,
                'last_utt': last_utt, 'between_utt': between_utt, 'between_utt_len': between_utt_len}

    dataset = datasets.Dataset.from_dict(d)
    dataset = datasets.DatasetDict({'test': dataset})

    return dataset

def prepare_meena_dataset(min_conv_len=6, max_conv_len=8):
    with open('/home/ivanlee/cyoa/datasets/meena-human2human.txt','r') as f:
        utts = f.read()
        utts = demoji.replace(utts, '')
    utts = utts.split("\n")

    conversations = []
    conv = None
    for utt in utts:
        if utt == '':
            continue
        if 'Human Conversation ' in utt:
            if conv is not None:
                conversations.append(conv.copy())
            conv = []
        else:
            utt = utt.replace('Human 1: ', '')
            utt = utt.replace('Human 2: ', '')
            conv.append(utt)
    conversations.append(conv)

    # Remove first and last two utterances which are usually greetings and good byes
    conversations = [conv[2:-2] for conv in conversations]

    # Filter out conversations that are too short
    conversations = [conv for conv in conversations if len(conv) >= min_conv_len]
    # Truncate conversations that are too long
    conversations = [conv[:max_conv_len] for conv in conversations]

    # Filter out conversations with redacted term
    conversations = [conv for conv in conversations if 'REDACTED' not in ''.join(conv)]
    
    full_conv = conversations
    first_utt = [conv[0] for conv in conversations]
    last_utt = [conv[-1] for conv in conversations]
    between_utt  = [conv[1:-1] for conv in conversations]
    between_utt_len = [len(utt) for utt in between_utt]
    
    d = {'full_conv': full_conv, 'first_utt': first_utt,
                'last_utt': last_utt, 'between_utt': between_utt, 'between_utt_len': between_utt_len}

    dataset = datasets.Dataset.from_dict(d)
    dataset = datasets.DatasetDict({'test': dataset})

    return dataset


def prepare_bst_dataset(args):
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

def get_strength(i, conv_len, args):
    strength = args.astar_strength * math.exp(args.c * i / conv_len)
    logger.debug(f"Strength: {strength}")
    return strength


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

    if args.dataset == "blended_skill_talk":
        dataset = prepare_bst_dataset(args)
    elif args.dataset == "meena":
        dataset = prepare_meena_dataset(args.min_conv_len, args.max_conv_len)
    elif args.dataset == "wow":
        dataset = prepare_wow_dataset(args.min_conv_len, args.max_conv_len)
    elif args.dataset == "persona":
        dataset = prepare_personachat_dataset(args.min_conv_len, args.max_conv_len)
    elif args.dataset == "empath":
        dataset = prepare_empath_dataset(args.min_conv_len, args.max_conv_len)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
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

    if args.max_samples is None:
        args.max_samples = len(dataset)

    # Default debug settings
    if args.debug:
        dataset = dataset.select(range(0, 2)) if args.start_idx is None else dataset.select(range(args.start_idx, args.start_idx + 2))
        args.astar_top_k = 5 if (args.astar_top_k is None) else args.astar_top_k
        args.output_dir = 'debug'
    else:
        dataset = dataset.select(range(args.start_idx, args.start_idx + args.max_samples))
    
    # Move this file path making stuff into a function
    fp = os.path.join(args.output_dir, f"{args.dataset.replace('_', '-')}")
    fp += f"_{args.split}"
    fp += f"_samples{args.start_idx}:{args.max_samples}"
    # fp += f"_{args.dataset}"
    fp += f"_{args.delimiter}"    

    fp += f"_astar" if args.do_astar else "_cosine" if args.do_cosine else "_beam"
    fp += f"_prompt" if args.do_prompt else ""    
    # fp += f"_startidx{args.start_idx}"    
    if args.do_astar:
        fp += f"_str{args.astar_strength}" if (args.astar_strength is not None) else ""
        fp += f"_c{args.c}"
        fp += f"_top{args.astar_top_k}" if (args.astar_top_k is not None) else ""

    fp += f"_seed{args.seed}"
    fp += f"_debug" if args.debug else ""
    fp += '.jsonl'
    logger.warning(f"Writing generated utterances to {fp}")
    # if fp doesn't exist, create it    
    if not os.path.exists(fp):
        with open(fp, 'w') as f:
            pass
    
    # Main loop
    for _, sample in enumerate(tqdm(dataset)):
        source_utt = sample['first_utt']
        target_utt = sample['last_utt']
        conv_len = sample['between_utt_len']
        # if args.debug:
            # source_utt = "My friends are cool but they eat too many carbs."
            # target_utt = 'The glory of the Roman empire is forever.'
            # conv_len = 10

        logger.debug(f"Source: {source_utt}")
        logger.debug(f"Target: {target_utt}")
        logger.debug(f"Conv len: {conv_len}")

        # Parlai delimiter mode
        if args.delimiter == 'tab':
            
            tokenizer.truncation_side = 'right'
            target = tokenizer([target_utt], truncation=True, return_tensors="pt").to(
                args.device).input_ids            

            conv_so_far = []
            conv_so_far.append(source_utt)
            contexts = []
            
            for i in range(conv_len):
                
                conv_so_far_str = SEP_TOK.join(conv_so_far)
                tokenizer.truncation_side = 'left'
                inputs = tokenizer([conv_so_far_str], truncation=True, return_tensors="pt").to(
                    args.device).input_ids

                if args.do_prompt:
                    if target.shape[-1] > 64:
                        prefix = target[:, :64]
                    else:
                        prefix = target
                    truncate_to = -(128 - prefix.shape[-1])
                    inputs = torch.cat((prefix, inputs[:, truncate_to:]), -1)

                logger.debug(
                    f"History {i}: {repr(tokenizer.batch_decode(inputs, skip_special_tokens=False)[0])}")     

                contexts.append(inputs[0].tolist())

                reply_ids = model.generate(input_ids=inputs,
                                        num_beams=args.beam_size,
                                        do_astar=args.do_astar,
                                        do_cosine=args.do_cosine,
                                        target_utterance=target,
                                        astar_strength=get_strength(i, conv_len, args),
                                        astar_top_k=args.astar_top_k,
                                        )

                logger.debug(f"Utt {i}: {repr(tokenizer.batch_decode(reply_ids, skip_special_tokens=False)[0])}\n")

                response = tokenizer.batch_decode(
                    reply_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
                conv_so_far.append(response.strip())


            # Write out the generated utterances
            output = {"first_utt": source_utt,
                    "target_utt": target_utt,
                    "middle_utt": conv_so_far[1:],
                    "gold_utt": sample['between_utt'],
                    "contexts": contexts,
                    }

        elif args.delimiter == 'raw':
            tokenizer.truncation_side = 'right'
            target = tokenizer([target_utt], truncation=True, return_tensors="pt").to(
                args.device).input_ids
            tokenizer.truncation_side = 'left'
            inputs = tokenizer([source_utt], truncation=True, return_tensors="pt").to(
                    args.device).input_ids

            conv_so_far = []
            contexts = []
            for i in range(conv_len):
                
                if args.do_prompt:
                    truncate_to = -(128 - target.shape[-1])
                    prompted_inputs = torch.cat((target, inputs[:, truncate_to:]), -1)

                logger.debug(
                    f"History {i}: {repr(tokenizer.batch_decode(prompted_inputs if args.do_prompt else inputs, skip_special_tokens=False)[0])}")
                contexts.append(prompted_inputs[0].tolist() if args.do_prompt else inputs[0].tolist())

                reply_ids = model.generate(input_ids=prompted_inputs if args.do_prompt else inputs,
                                        num_beams=3,
                                        do_astar=args.do_astar,
                                        do_cosine=args.do_cosine,
                                        target_utterance=target,
                                        astar_strength=get_strength(i, conv_len, args),
                                        astar_top_k=args.astar_top_k,
                                        )
                
                # Update dialogue history and truncate
                inputs = torch.cat((inputs, reply_ids), dim=1)
                inputs = inputs[:, -128:]

                logger.debug(f"Utt {i}: {repr(tokenizer.batch_decode(reply_ids, skip_special_tokens=False)[0])}")

                response = tokenizer.batch_decode(
                    reply_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
                conv_so_far.append(response.strip())


            # Write out the generated utterances
            output = {"first_utt": source_utt,
                    "target_utt": target_utt,
                    "middle_utt": conv_so_far,
                    "gold_utt": sample['between_utt'],
                    "contexts": contexts,
                    }
            
        else:
            raise ValueError(f"Invalid delimiter: {args.delimiter}")
        
        with jsonlines.open(fp, mode='a') as writer:
                writer.write(output)
    


if __name__ == "__main__":
    main()
