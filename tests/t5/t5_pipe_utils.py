import datetime
import json
import logging
import os
import pathlib
import string
import loguru
import sh
import random
from typing import Tuple, List, Dict, Any

import numpy as np
import pytz

logger = loguru.logger


def is_rank_0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def log_dist(message: str, ranks: List[int] = [], level: int = logging.INFO) -> None:
    """Log messages for specified ranks only"""
    my_rank = int(os.environ.get("RANK", "0"))
    if my_rank in ranks:
        if level == logging.INFO:
            logger.info(f'[Rank {my_rank}] {message}')
        if level == logging.ERROR:
            logger.error(f'[Rank {my_rank}] {message}')
        if level == logging.DEBUG:
            logger.debug(f'[Rank {my_rank}] {message}')


def replace_function(text, tokenizer, corruption_rate, corruption_span, max_length):
    tokenized_ids = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_length)
    labels = []
    # calculate number of corruption
    corruption_num = int(len(tokenized_ids) * corruption_rate // corruption_span)
    if corruption_num == 0:
        if corruption_span < len(tokenized_ids):
            corruption_num = 1
        else:
            raise Exception('corruption span({}) > tokenized ids({})!'.format(corruption_span, len(tokenized_ids)))

    # get list of position to corruption
    rand_list = []
    choice_list = list(range(0, len(tokenized_ids) - corruption_span))
    for _ in range(corruption_num):
        choice = random.randint(0, len(choice_list) - corruption_span)
        rand_list.append(choice_list[choice])
        choice_list = choice_list[:max(choice - corruption_span + 1, 0)] + choice_list[choice+corruption_span:]
    rand_list.sort()

    # with list of position to corruption, get labels
    assert len(rand_list) <= len(tokenizer.additional_special_tokens_ids), 'Number of corruption is more than special tokens'
    for i, rand_pos in enumerate(rand_list):
        labels += [tokenizer.additional_special_tokens_ids[i]] + tokenized_ids[rand_pos:rand_pos + corruption_span]
        tokenized_ids = tokenized_ids[:rand_pos] + ([-1] * corruption_span) + tokenized_ids[rand_pos + corruption_span:]
        tokenized_ids[rand_pos] = tokenizer.additional_special_tokens_ids[i]
    tokenized_ids = [x for x in tokenized_ids if x != -1]

    return tokenized_ids, labels + [tokenizer.eos_token_id]


def mask_function(text, tokenizer, corruption_rate, corruption_span, max_length):
    tokenized_ids = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_length)
    labels = tokenized_ids[:]
    corruption_num = int(len(tokenized_ids) * corruption_rate)
    if corruption_num == 0:
        if corruption_span < len(tokenized_ids):
            corruption_num = 1
        else:
            raise Exception('corruption span({}) > tokenized ids({})!'.format(corruption_span, len(tokenized_ids)))

    for i in random.sample(list(range(0, len(tokenized_ids))), corruption_num):
        tokenized_ids[i] = tokenizer.mask_token_id
    return tokenized_ids, labels


def drop_function(text, tokenizer, corruption_rate, corruption_span, max_length):
    tokenized_ids = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_length)
    labels = []
    corruption_num = int(len(tokenized_ids) * corruption_rate)
    if corruption_num == 0:
        if corruption_span < len(tokenized_ids):
            corruption_num = 1
        else:
            raise Exception('corruption span({}) > tokenized ids({})!'.format(corruption_span, len(tokenized_ids)))

    for i in sorted(random.sample(list(range(0, len(tokenized_ids))), corruption_num)):
        labels.append(tokenized_ids[i])
        tokenized_ids[i] = -1
    tokenized_ids = list(filter(lambda x: x != -1, tokenized_ids))
    return tokenized_ids, labels


def masking_function(
        text: str,
        tokenizer,
        mask_prob: float,
        random_replace_prob: float,
        unmask_replace_prob: float,
        max_length: int,
) -> Tuple[List[int], List[int]]:
    # Note: By default, encode does add the BOS and EOS token
    # Disabling that behaviour to make this more clear
    tokenized_ids = ([tokenizer.bos_token_id] +
                     tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=max_length - 2) +
                     [tokenizer.eos_token_id])
    seq_len = len(tokenized_ids)
    tokenized_ids = np.array(tokenized_ids)
    subword_mask = np.full(len(tokenized_ids), False)

    # Masking the BOS and EOS token leads to slightly worse performance
    low = 1
    high = len(subword_mask) - 1
    mask_choices = np.arange(low, high)
    num_subwords_to_mask = max(int((mask_prob * (high - low)) + np.random.rand()), 1)
    subword_mask[np.random.choice(mask_choices, num_subwords_to_mask, replace=False)] = True

    # Create the labels first
    labels = np.full(seq_len, tokenizer.pad_token_id)
    labels[subword_mask] = tokenized_ids[subword_mask]

    tokenized_ids[subword_mask] = tokenizer.mask_token_id

    # Now of the masked tokens, choose how many to replace with random and how many to unmask
    rand_or_unmask_prob = random_replace_prob + unmask_replace_prob
    if rand_or_unmask_prob > 0:
        rand_or_unmask = subword_mask & (np.random.rand(len(tokenized_ids)) < rand_or_unmask_prob)
        if random_replace_prob == 0:
            unmask = rand_or_unmask
            rand_mask = None
        elif unmask_replace_prob == 0:
            unmask = None
            rand_mask = rand_or_unmask
        else:
            unmask_prob = unmask_replace_prob / rand_or_unmask_prob
            decision = np.random.rand(len(tokenized_ids)) < unmask_prob
            unmask = rand_or_unmask & decision
            rand_mask = rand_or_unmask & (~decision)
        if unmask is not None:
            tokenized_ids[unmask] = labels[unmask]
        if rand_mask is not None:
            weights = np.ones(tokenizer.vocab_size + 2)  # add two special tokens
            weights[tokenizer.all_special_ids] = 0
            probs = weights / weights.sum()
            num_rand = rand_mask.sum()
            tokenized_ids[rand_mask] = np.random.choice(tokenizer.vocab_size + 2, num_rand, p=probs)
    return tokenized_ids.tolist(), labels.tolist()


def get_unique_identifier(length: int = 8) -> str:
    alphabet = string.ascii_lowercase + string.digits
    uuid = "".join(alphabet[ix] for ix in np.random.choice(len(alphabet), length))
    return uuid


def create_experiment_dir(checkpoint_dir: pathlib.Path, all_arguments: Dict[str, Any]) -> pathlib.Path:
    # experiment name follows the following convention
    # {exp_type}.{YYYY}.{MM}.{DD}.{HH}.{MM}.{SS}.{uuid}
    current_time = datetime.datetime.now(pytz.timezone("US/Pacific"))
    expname = "t5_pretrain_{0}_{1}_{2}_{3}_{4}_{5}_{6}".format(
        current_time.year,
        current_time.month,
        current_time.day,
        current_time.hour,
        current_time.minute,
        current_time.second,
        get_unique_identifier(),
    )
    exp_dir = checkpoint_dir / expname
    if not is_rank_0():
        return exp_dir
    exp_dir.mkdir(exist_ok=False)
    hparams_file = exp_dir / "hparams.json"
    with hparams_file.open("w") as handle:
        json.dump(obj=all_arguments, fp=handle, indent=2)
    # Save the git hash
    try:
        gitlog = sh.git.log("-1", format="%H", _tty_out=False, _fg=False)
        with (exp_dir / "githash.log").open("w") as handle:
            handle.write(gitlog.stdout.decode("utf-8"))
    except sh.ErrorReturnCode_128:
        log_dist('Seems like the code is not running from within a git repo, so hash will not be stored. However, it'
                 ' is strongly advised to use version control.', ranks=[0], level=logging.INFO)
    # And the git diff
    try:
        gitdiff = sh.git.diff(_fg=False, _tty_out=False)
        with (exp_dir / "gitdiff.log").open("w") as handle:
            handle.write(gitdiff.stdout.decode("utf-8"))
    except sh.ErrorReturnCode_129:
        log_dist('Seems like the code is not running from within a git repo, so diff will not be stored. However, it'
                 ' is strongly advised to use version control.', ranks=[0], level=logging.INFO)
    # Finally create the Tensorboard Dir
    tb_dir = exp_dir / "tb_dir"
    tb_dir.mkdir(exist_ok=False)
    return exp_dir
