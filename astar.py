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

SEP_TOK = '    '

logger = logging.getLogger(__name__)


@dataclass
class Arguments:
    debug: bool = field(default=None, metadata={"help": "Debug mode or not."})
    seed: int = field(default=None, metadata={"help": "Random seed."})
    model: str = field(default=None, metadata={"help": "Model to use."})
    dataset: str = field(default=None, metadata={"help": "Dataset to use."})
    split: str = field(default=None, metadata={"help": "Split to use."})
    astar_strength: int = field(default=None, metadata={"help": "Strength of heuristic."})
    astar_top_k: int = field(default=None, metadata={"help": "Top k to use."})
    output_dir: str = field(default=None, metadata={"help": "Output directory."})
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


    first_utts = []
    target_utts = []
    middle_utts = []
    gold_utts = []

    for _, sample in enumerate(tqdm(dataset)):
        source_utt = sample['first_utt']
        target_utt = sample['last_utt']
        conv_len = sample['between_utt_len']

        logger.debug(f"Source: {source_utt}")
        logger.debug(f"Target: {target_utt}")
        logger.debug(f"Conv len: {conv_len}")

        conv_so_far = []
        conv_so_far.append(source_utt)
        
        target = tokenizer([target_utt], return_tensors="pt").to(args.device).input_ids
        for i in range(conv_len):
            
            conv_so_far_str = SEP_TOK.join(conv_so_far)
            inputs = tokenizer([conv_so_far_str], truncation=True, return_tensors="pt").to(args.device).input_ids
            
            logger.debug(f"History {i}: {repr(tokenizer.batch_decode(inputs, skip_special_tokens=False)[0])}")            

            reply_ids = model.generate(input_ids=inputs,
                                       # decoder_input_ids=decoder_input_ids,
                                       num_beams=3,
                                       do_astar=True,
                                       target_utterance=target,
                                       astar_strength=args.astar_strength,
                                       astar_top_k=args.astar_top_k,
                                       )

            # logger.debug(f"Utt {i}: {repr(tokenizer.batch_decode(reply_ids, skip_special_tokens=False)[0])}")

            response = tokenizer.batch_decode(reply_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
            conv_so_far.append(response.strip())

            # concatenate reply_ids to original utterance
            # inputs = torch.cat((inputs, reply_ids), dim=1)
            # truncate left side of inputs if longer than 128
            # inputs = inputs[:, -128:]

            
            
            # print()
        
        first_utts.append(source_utt)
        middle_utts.append(conv_so_far[1:])
        gold_utts.append(sample['between_utt'])
        target_utts.append(target_utt)
    
    # to dataframe
    df = pd.DataFrame({"first_utt": first_utts, "target_utt": target_utts, "middle_utt": middle_utts, "gold_utt": gold_utts})
    # save to pkl   
    fp = os.path.join(args.output_dir, "gen")
    fp += f"_strength{args.astar_strength}"
    fp += f"_topk{args.astar_top_k}"
    fp += f"_debug" if args.debug else ""
    fp += '.pkl'
    df.to_pickle(fp)
    logger.warning(f"Writing generated utterances to {fp}")



    



if __name__ == "__main__":
    main()

"""
V: '<s><s><s> I know that feeling. I have a friend who eats way too much junk food.</s>'
A: "<s><s> That's unfortunate. I'm sorry to hear that. I know how that goes.</s>"

V: "<s> Yeah, it's hard to watch them eat so much. I try to limit my carbs as much as I can.</s>"
A: '<s><s><s> I know, I was so jealous of them. I was going to go to the casino with them.</s>'

V: <s> I am trying to do the same, but it is hard when you are trying to lose weight.</s>'
A:  '<s> Oh no, I am sorry.  I hope you can still go to a casino.</s>'

History:  " shape too. I need to lose a few pounds.</s><s><s> Yes, it is very important to eat healthy.  It will make you feel so much better.  Good luck!</s><s> Thank you! I hope you are able to achieve your goals as well. I know it can be hard.</s><s> Thanks! I'm determined too, I just need to find a way to make it happen.</s><s> You can do it!  I believe in you!  What kind of work do you do?</s><s> I work in an office. It's not very exciting, but it pays the bills. </s>"

UTTERANCE = "My friends are cool but they eat too many carbs."
target_utterance = 'The glory of the Roman empire is forever.'
 "</s><s><s> That's funny. I'm the same way. I love pizza. I don't know why I love it so much.</s><s> Pizza's my favorite food. I can't believe it's been around almost as long as the Roman empire.</s><s> Yeah, it's amazing how long it's stayed around. I like it with vegetables and meats.</s><s> I know right! I love the sauce and cheese and all the meats and vegetables.</s><s><s> The Roman Empire was a great time to be alive. It was a very important part of the history of the world.</s>"


UTTERANCE = "My friends are cool but they eat too many carbs."
target_utterance = 'When you step on wet cement you can leave footprints.'
"My friends are cool but they eat too many carbs.</s><s> I am sorry to hear that.  I know that can be hard to deal with.  What do you do to help?</s><s> I just don't know how to get them to stop. I'm not sure how to help.</s><s> They are not my friends. They are just annoying. I am not sure what to do.</s><s> I have no idea how to stop them either. I guess I just have to be patient.</s>"



UTTERANCE = "My friends are cool but they eat too many carbs."
    target_utterance = 'The glory of the Roman empire is forever.'
Bot:  '<s> I know how that is. I have a friend who eats way too much bread and pasta.</s>'
History:  ' My friends are cool but they eat too many carbs.</s><s> I know how that is. I have a friend who eats way too much bread and pasta.</s>'

Bot:  '<s><pad><unk_AGAIN><s>$+<unk_AGAIN>*<pad_AGAIN><pad_AGAIN>&&&)+\'-,",%(&*,*%(<s></s_AGAIN>,,,-,<s>,,<s>&,,#,<s>\'&&**</s>'
History:  ' My friends are cool but they eat too many carbs.</s><s> I know how that is. I have a friend who eats way too much bread and pasta.</s><s><pad><unk_AGAIN><s>$+<unk_AGAIN>*<pad_AGAIN><pad_AGAIN>&&&)+\'-,",%(&*,*%(<s></s_AGAIN>,,,-,<s>,,<s>&,,#,<s>\'&&**</s>'

Bot:  '<s><s><s> That is a good one, I like it.  I like to eat a lot of pasta and bread.</s>'
History:  ' My friends are cool but they eat too many carbs.</s><s> I know how that is. I have a friend who eats way too much bread and pasta.</s><s><pad><unk_AGAIN><s>$+<unk_AGAIN>*<pad_AGAIN><pad_AGAIN>&&&)+\'-,",%(&*,*%(<s></s_AGAIN>,,,-,<s>,,<s>&,,#,<s>\'&&**</s><s><s><s> That is a good one, I like it.  I like to eat a lot of pasta and bread.</s>'

Bot:  '<s> I like pasta too. I like all kinds of pastas. I love the Italian cuisine.</s>'
History:  '<s> I know how that is. I have a friend who eats way too much bread and pasta.</s><s><pad><unk_AGAIN><s>$+<unk_AGAIN>*<pad_AGAIN><pad_AGAIN>&&&)+\'-,",%(&*,*%(<s></s_AGAIN>,,,-,<s>,,<s>&,,#,<s>\'&&**</s><s><s><s> That is a good one, I like it.  I like to eat a lot of pasta and bread.</s><s> I like pasta too. I like all kinds of pastas. I love the Italian cuisine.</s>'

^[[A^[[A^[[A^[[A^[[A^[[A^[[A^[[A^[[A^[[A^[[A^[[A^[[A^[[A^[[A^[[A^[[A^[[A^[[A^[[A^[[A^[[A^[[A^[[A^[[A^[[A^[[ABot:  '<s> I do too.  Italians have such a rich history.  They have been around since the 5th century BCE.</s>'
History:  '<unk_AGAIN>*<pad_AGAIN><pad_AGAIN>&&&)+\'-,",%(&*,*%(<s></s_AGAIN>,,,-,<s>,,<s>&,,#,<s>\'&&**</s><s><s><s> That is a good one, I like it.  I like to eat a lot of pasta and bread.</s><s> I like pasta too. I like all kinds of pastas. I love the Italian cuisine.</s><s> I do too.  Italians have such a rich history.  They have been around since the 5th century BCE.</s>'

Bot:  "<s><s> Wow, that's a long time.  The history of Italia goes back to the beginning of the Roman Empire.</s>"
History:  "<s>,,<s>&,,#,<s>'&&**</s><s><s><s> That is a good one, I like it.  I like to eat a lot of pasta and bread.</s><s> I like pasta too. I like all kinds of pastas. I love the Italian cuisine.</s><s> I do too.  Italians have such a rich history.  They have been around since the 5th century BCE.</s><s><s> Wow, that's a long time.  The history of Italia goes back to the beginning of the Roman Empire.</s>"

Bot:  "<s> I know! It's so interesting to think about how much history has changed over the years.</s>"
History:  " a good one, I like it.  I like to eat a lot of pasta and bread.</s><s> I like pasta too. I like all kinds of pastas. I love the Italian cuisine.</s><s> I do too.  Italians have such a rich history.  They have been around since the 5th century BCE.</s><s><s> Wow, that's a long time.  The history of Italia goes back to the beginning of the Roman Empire.</s><s> I know! It's so interesting to think about how much history has changed over the years.</s>"

Bot:  "<s><s><s> Yes, it's amazing how much we have changed.  We have come a long way since then. </s>"
History:  "a too. I like all kinds of pastas. I love the Italian cuisine.</s><s> I do too.  Italians have such a rich history.  They have been around since the 5th century BCE.</s><s><s> Wow, that's a long time.  The history of Italia goes back to the beginning of the Roman Empire.</s><s> I know! It's so interesting to think about how much history has changed over the years.</s><s><s><s> Yes, it's amazing how much we have changed.  We have come a long way since then. </s>"
[]
Bot:  "<s> It really has. I can't believe how much has changed since the Roman empire. It's crazy how much they've changed.</s>"
History:  " Italians have such a rich history.  They have been around since the 5th century BCE.</s><s><s> Wow, that's a long time.  The history of Italia goes back to the beginning of the Roman Empire.</s><s> I know! It's so interesting to think about how much history has changed over the years.</s><s><s><s> Yes, it's amazing how much we have changed.  We have come a long way since then. </s><s> It really has. I can't believe how much has changed since the Roman empire. It's crazy how much they've changed.</s>"

Bot:  "<s> Yes! It is amazing how far we've come. It was a long, long time ago.</s>"
History:  "</s><s><s> Wow, that's a long time.  The history of Italia goes back to the beginning of the Roman Empire.</s><s> I know! It's so interesting to think about how much history has changed over the years.</s><s><s><s> Yes, it's amazing how much we have changed.  We have come a long way since then. </s><s> It really has. I can't believe how much has changed since the Roman empire. It's crazy how much they've changed.</s><s> Yes! It is amazing how far we've come. It was a long, long time ago.</s>"

Source: My friends are cool but they eat too many carbs.
Target: The glory of the Roman empire is forever.
Utt 0:  '<s><s><s> I know that feeling. I had a friend who ate way too much pizza last night.</s>'
Utt 1:  '<s> Pizza is my lifeblood.  I could eat it every day for the rest of my life.</s>'
Utt 2:  "<s> Me too, I really like pizza. I can't believe it's been around as long as the Roman Empire.</s>"
Utt 3:  "<s><s> Yeah, it's crazy how long it has been around. It's crazy to think that it was invented in Naples.</s>"
Utt 4:  "<s><s> That's right! It's been a long time. It was first recorded in the 10th century in a Latin manuscript found in the Southern Italy town of Gaeta in Lazio.</s>"
Utt 5:  '<s><s> I know, right? It seems like it was so long ago. I remember reading about it in a book I was reading about when I was a kid.</s>'
Utt 6:  "<s> It's amazing to think how far we've come since then, isn't it? It's hard to believe it's been that long since the Roman empire was founded.</s>"
Utt 7:  "<s><s> It really is.  It's crazy to think that the Roman Empire lasted so long.</s>"
Utt 8:  "<s><s> Yes, it really is! It's incredible to think about how long it has been around.</s>"
Utt 9:  "<s><s><s> Yeah, it's amazing how far it has come. It's been around since the 4th century BCE.</s>"

Source: My friends are cool but they eat too many carbs.
Target: The beaches in San Diego are beautiful.
Utt 0:  '<s><s> I know what you mean. I have a friend who eats a lot of carbs too.</s>'
Utt 1:  "<s><s> Yeah, it's hard to watch them eat so much. I try to limit my carbs, but it's not always easy.</s>"
Utt 2:  '<s><s> It is hard, but I am trying to be more healthy and eat more vegetables and fruits.</s>'
Utt 3:  "<s><s><s> That's a good idea. I'm trying to do the same. I need to eat more veggies.</s>"
Utt 4:  "<s><s> Yes, I'm doing the same thing. I've been trying to eat healthier, but sometimes I just can't help it.</s>"
Utt 5:  "<s> I know what you mean. It's so hard to stay on a diet when you're trying to lose weight.</s>"
Utt 6:  '<s> Yes it is. I am looking forward to the summer so I can go swimming. I love the beach.</s>'
Utt 7:  '<s><s> I love swimming too!  I would love to go to a beach more often but I live in the city.</s>'
Utt 8:  '<s> I live near a really nice beach, so I go there a lot. It is a great place to relax and enjoy the water.</s>'
Utt 9:  '<s> Yes, I love going to the beaches in San Diego. There are so many beautiful beaches there.</s>'

Source: My friends are cool but they eat too many carbs.
Target: When you step on wet cement you can leave footprints.
Utt 0:  '<s> I am sorry to hear that.  I know that can be hard to deal with.  Do you have any suggestions on how to help them?</s>'
Utt 1:  "<s> Well, I'm not sure how to really help them. I just try to be there for them when they need me.</s>"
Utt 2:  "<s><s><s> You can't help them, but you can always try to talk to them and tell them how you feel.</s>"
Utt 3:  "<s><s> I've tried, but they don't seem to care. I don't know what to do.</s>"
Utt 4:  "<s><s> What do you mean? I mean, you can't really do anything about it unless you want to help.</s>"
Utt 5:  "<s> I think you're right. I guess I'll just have to suck it up and hope for the best.</s>"
Utt 6:  "<s> That's true. I'm sorry you have to deal with that. I hope you can find a way to get through it.</s>"
Utt 7:  "<s> I'm not sure what you mean by that, but thank you for your kind words. </s>"
Utt 8:  '<s> I am sorry to hear that.  I hope things get better for you soon.  </s>'
Utt 9:  '<s> Thank you. I wish I could help you, but I am not sure how to help you.</s>'

Source: My friends are cool but they eat too many carbs.
Target: Gondor calls for aid!
Utt 0:  '<s> I am sorry to hear that.  I know that can be hard to deal with.  Do you have any suggestions on how to help them?</s>'
Utt 1:  "<s> Well, I'm not sure how to really help them, but I do know that they need to watch their carbs intake.</s>"
Utt 2:  "<s><s> Well I don't really know how to do that either. I just try to tell them to eat more vegetables and fruits.</s>"
Utt 3:  "<s><s><s> I know it's hard, but you can't always help them.  They just have to be careful.  It's hard for them to watch what they eat, but they have to do it for themselves.</s>"
Utt 4:  "<s> Yeah, I try to be there for them as much as I can, but it's so hard sometimes.</s>"
Utt 5:  '<s> I just wish they would eat more veggies and fruit. I love them so much.</s>'
Utt 6:  "<s> I wish I could help them, but I'm not sure how to help them with that.</s>"
Utt 7:  "<s><s> That's a good point. I think it's just a matter of time and patience.</s>"
Utt 8:  "<s><s> I think you're right. It's just hard to watch them lose their appetite.</s>"
Utt 9:  '<s> I know what you mean. It can be hard to be patient with them sometimes. I hope they get better soon.</s>'

Source: My friends are cool but they eat too many carbs.
Target: The best way to get to the moon is by spaceship.
Utt 0:  "<s> I do too. I need to cut back. I'm trying to eat healthier but it's hard.</s>"
Utt 1:  '<s> It is hard, but it is worth it in the end.  I am trying to do the same.</s>'
Utt 2:  "<s><s><s> It's hard to do, but I'm sure you'll get there! I'm proud of you for trying.</s>"
Utt 3:  '<s> Thank you so much! I appreciate it. I am determined to get there. I know I can do it.</s>'
Utt 4:  '<s><s> You can do this! I believe in you! I know you can do anything you set your mind to!</s>'
Utt 5:  '<s> Thanks for the encouragement! I really appreciate it! I hope you can get there too!</s>'
Utt 6:  "<s> Thanks. I'm trying to keep my head up. It's tough but I know it's worth it. </s>"
Utt 7:  '<s> You are very welcome. I will keep my fingers crossed for you. I believe you will do great!</s>'
Utt 8:  "<s> Thanks! I'm sure I will too. I just need to be patient.   </s>"
Utt 9:  '<s><s><s> I am sure you will get there! I am rooting for you! You got this!</s>'

"""
