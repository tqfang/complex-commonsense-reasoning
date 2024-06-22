# Importing stock libraries
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

import logging
from tqdm import tqdm

logger = logging.getLogger("modeling")

def beam_generations_new(tokenizer, model, device, loader, out_len=34, top_k=40, num_gen=1, min_new_tokens=1):
    # This method assumes batch size of 1
    model.eval()
    predictions = []
    actuals = []
    sources = []
    records = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype=torch.long)
            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)
            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                temperature=1.0,
                do_sample=False,
                max_length=out_len,
                top_p=0.9,
                top_k=top_k,
                repetition_penalty=1.0,
                num_return_sequences=num_gen if top_k > 1 else 1,
                num_beams=10, 
                min_new_tokens=min_new_tokens
            )
            preds = [tokenizer.decode(g, clean_up_tokenization_spaces=True) for g in generated_ids]
            try:
                target = [tokenizer.decode(t, clean_up_tokenization_spaces=True) for t in y]
            except:
                target = ['']
            source = [tokenizer.decode(s, clean_up_tokenization_spaces=True) for s in ids]
            records.append({
                'source': source[0],
                'target': target[0],
                'generations': preds
            })
            if _ % 100 == 0:
                logger.info(f'Completed {_}')
    return records

def get_gen_comet_m(strs):
    strs = strs.split()
    st = 0
    ed = 0
    for i in range(len(strs)):
        if strs[i] == "[GEN]":
            st = i
        if strs[i] == "[EOS]" or (st > 0 and strs[i] == "[PAD]"):
            ed = i
            break
    return " ".join(strs[st+1:ed])

# def get_gen_comet_m(strs):
    
#     st = strs.find('[GEN]')
#     strs = strs[st+5:]
#     end = strs.find('\n')
#     strs = strs[:end].replace('.', '')

#     return strs.strip()

# def get_gen(strs):
    
#     st = strs.find('[GEN]')
#     strs = strs[st+5:]
#     end = strs.find('\n')
#     strs = strs[:end].replace('.', '')

#     return strs.strip()