import sys
import logging
import transformers
import torch
from dataclasses import dataclass, field
import os

logger = logging.getLogger(__name__)
red = "\x1b[31;20m"
reset = "\x1b[0m"


@dataclass
class Arguments:
    debug: bool = field(default=None, metadata={"help": "Debug mode or not."})
    seed: int = field(default=None, metadata={"help": "Random seed."})
    trunk_dir: str = field(default=None, metadata={
                           "help": "Trunk directory for large files."})


def main():
    parser = transformers.HfArgumentParser(
        (Arguments))
    args = parser.parse_args_into_dataclasses()[0]
    logger.info(args)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformers.set_seed(args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel('DEBUG') if args.debug else logger.setLevel('INFO')
    logger.info(args)

    tokenizer = transformers.GPT2Tokenizer(os.path.join(args.trunk_dir, 'dialogpt345M_forward/vocab.json'),
                                           os.path.join(args.trunk_dir, 'dialogpt345M_forward/merges.txt'))
    # logger.debug(f'eos token: {tokenizer.eos_token}')
    # logger.debug(f'eos token id: {tokenizer.eos_token_id}')
    
    weights = torch.load(os.path.join(
        args.trunk_dir, 'dialogpt345M_forward/medium_ft.pkl'))
    # fix misused key value
    weights["lm_head.weight"] = weights["lm_head.decoder.weight"]
    weights.pop("lm_head.decoder.weight", None)
    # logger.debug(f"Weight keys: {weights.keys()}")

    cfg = transformers.GPT2Config.from_json_file(os.path.join(
        args.trunk_dir, 'dialogpt345M_forward/config.json'))
    logger.debug(cfg)

    model = transformers.GPT2LMHeadModel(cfg)
    incompatible_keys = model.load_state_dict(weights, strict=False)
    logger.debug(f"missing_keys: {incompatible_keys.missing_keys}")
    logger.debug(f"unexpected_keys: {incompatible_keys.unexpected_keys}")
    for key in incompatible_keys.missing_keys:
        assert 'attn.masked_bias' in key
    assert len(incompatible_keys.unexpected_keys) == 0

    tokenizer = transformers.AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
    # model = transformers.AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-medium')

    if args.device == 'cuda':
        model.half()
    model.to(args.device)
    model.eval()

    

    for step in range(5):
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(
            input(">> User:") + tokenizer.eos_token, return_tensors='pt')
        new_user_input_ids = new_user_input_ids.to(args.device)

        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat(
            [chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids
        bot_input_ids = bot_input_ids.to(args.device)

        # generated a response while limiting the total chat history to 1000 tokens,
        print(bot_input_ids)
        chat_history_ids = model.generate(
            bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

        # pretty print last ouput tokens from bot
        # print("DialoGPT: {}".format(tokenizer.decode(
        #     chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
        print(chat_history_ids)
        
        print("DialoGPT: {}".format(tokenizer.batch_decode(
            chat_history_ids, skip_special_tokens=False)))

    foo
    weights = torch.load('medium/small_reverse.pkl')
    # fix misused key value
    weights["lm_head.weight"] = weights["lm_head.decoder.weight"]
    weights.pop("lm_head.decoder.weight", None)

    reverse_model: GPT2LMHeadModel = GPT2LMHeadModel(cfg)
    reverse_model.load_state_dict(weights)
    if device_r == 'cuda':
        reverse_model.half()
    reverse_model.to(device_r)
    reverse_model.eval()


if __name__ == "__main__":
    main()
