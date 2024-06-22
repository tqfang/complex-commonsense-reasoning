# Importing stock libraries
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import json
from typing import List
import argparse

# Importing the GPT2 modules from huggingface/transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup, get_constant_schedule, GPTNeoForCausalLM

# Import os for env varibles via Beaker
import os

# WandB – Import the wandb library
import wandb
import logging
import pdb
from torch import cuda
import sys
sys.path.append(os.getcwd())
sys.path.append("eval/")
from eval_generation import return_eval_scores
from split.utils import write_items
from models.utils import CS_RELATIONS_2NL
from comet_m.functions import beam_generations_new, get_gen_comet_m

from optparse import OptionParser

device = 'cuda' if cuda.is_available() else 'cpu'

logger = logging.getLogger("gpt2-comet")
logging.basicConfig(level=logging.DEBUG)

# logger.info for allenai beaker verification
logger.info(device)
logger.info(torch.cuda.device_count())

from mosaic.infra.modeling import train, validate, beam_generations
from mosaic.datasets.KGDataset import KGDataset

DEBUG = False
NUM_INST = 100


def read_jsonl_lines(input_file: str) -> List[dict]:
    with open(input_file) as f:
        lines = f.readlines()
        return [json.loads(l.strip()) for l in lines]


def main():

    config = argparse.Namespace()

    config.SEED = int(os.environ.get("SEED", 42))
    config.IN_LEN = int(os.environ.get("IN_LEN", 16))
    config.OUT_LEN = int(os.environ.get("OUT_LEN", 34))
    config.OUT_DIR = os.environ.get("OUT_DIR", "models")

    config.DO_PRED = os.environ.get("DO_PRED", "True") == "True"
    config.USE_PROMPT_RELATION = os.environ.get("USE_PROMPT_RELATION", "False") == "True"
    config.PRED_FILE = str(os.environ.get("PRED_FILE", "system_eval/test_eval_for_gpt2.csv"))
    config.PRED_TARGET = str(os.environ.get("PRED_FILE", "system_eval/test_eval_for_gpt2.csv"))
    config.TOP_K = int(os.environ.get("TOP_K", 40))
    config.NUM_GEN = int(os.environ.get("NUM_GEN", 1))

    config.TOKENIZER = os.environ.get('TOKENIZER', "gpt2-xl")
    config.GPT2_MODEL = os.environ.get('GPT2_MODEL', "gpt2-xl")


    torch.manual_seed(config.SEED)  # pytorch random seed
    np.random.seed(config.SEED)  # numpy random seed
    torch.backends.cudnn.deterministic = True

    model_name = config.GPT2_MODEL
    # "gpt2" if 'GPT2_MODEL' not in os.environ else os.environ['GPT2_MODEL']

    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    except:
        tokenizer = GPT2Tokenizer.from_pretrained(config.TOKENIZER)

    tokenizer.add_special_tokens({
        'eos_token': '[EOS]',
        'additional_special_tokens': [
            'LocationOfAction',
            'HinderedBy',
            'HasFirstSubevent',
            'NotHasProperty',
            'NotHasA',
            'HasA',
            'AtLocation',
            'NotCapableOf',
            'CausesDesire',
            'HasPainCharacter',
            'NotDesires',
            'MadeUpOf',
            'InstanceOf',
            'SymbolOf',
            'xReason',
            'isAfter',
            'HasPrerequisite',
            'UsedFor',
            'MadeOf',
            'MotivatedByGoal',
            'Causes',
            'oEffect',
            'CreatedBy',
            'ReceivesAction',
            'NotMadeOf',
            'xWant',
            'PartOf',
            'DesireOf',
            'HasPainIntensity',
            'xAttr',
            'DefinedAs',
            'oReact',
            'xIntent',
            'HasSubevent',
            'oWant',
            'HasProperty',
            'IsA',
            'HasSubEvent',
            'LocatedNear',
            'Desires',
            'isFilledBy',
            'isBefore',
            'InheritsFrom',
            'xNeed',
            'xEffect',
            'xReact',
            'HasLastSubevent',
            'RelatedTo',
            'CapableOf',
            'NotIsA',
            'ObjectUse',
            '[GEN]',
            '[INTERSECTION]',
            '[V1?]',
            '[V2?]',
            '<TGT>',
        ]
    })
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    val_params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 0
    }

    logging.info("Loading model from {}".format(model_name))
    if model_name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]:
        model = GPT2LMHeadModel.from_pretrained(model_name)
    elif model_name in ["EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B"]:
        model = GPTNeoForCausalLM.from_pretrained(model_name)
    else:
        # for checkpoint loading
        try:
            model = GPT2LMHeadModel.from_pretrained(model_name)
        except:
            raise NotImplementedError
    logging.info("Move model to device {}".format(device))
    model = model.to(device)
    model.resize_token_embeddings(len(tokenizer))


    if config.DO_PRED:

        pred_dataset = pd.DataFrame({"head_event":open(config.PRED_FILE).readlines(), "tail_event":""})
        pred_dataset = pred_dataset.drop_duplicates(['head_event'], ignore_index=True)

        if DEBUG:
            pred_dataset = pred_dataset.head(NUM_INST)

        # pred_dataset.tail_event = pred_dataset.tail_event + ' [EOS]'
        logger.info(pred_dataset.tail_event)
        logger.info(pred_dataset.head())

        pred_set = KGDataset(pred_dataset, tokenizer, config.IN_LEN, config.OUT_LEN - config.IN_LEN, model="gpt2", is_eval=True)
        pred_loader = DataLoader(pred_set, **val_params, drop_last=False)

        pred_generations = beam_generations_new(tokenizer, model, device, pred_loader, out_len=config.OUT_LEN, top_k=config.TOP_K, num_gen=config.NUM_GEN, min_new_tokens=1)
        os.makedirs(config.OUT_DIR, exist_ok=True)
        write_items(os.path.join(config.OUT_DIR, f"pred_generations.jsonl"),
                    [json.dumps(r) for r in pred_generations])

        # eval comet-m


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-t", "--test_install",
                      action="store_true", default=False,
                      help="Test install, without running any modeling code.")

    (options, args) = parser.parse_args()
    if not options.test_install:
        main()
