import re
import os
import openai
import pandas as pd
import time
from tqdm import tqdm
import numpy as np
import math

import asyncio
import json
import time
from sklearn.metrics import accuracy_score

def candidate_template(options):
    return f"A: {options[0]}\nB: {options[1]}\nC: {options[2]}\nD: {options[3]}\nE: {options[4]}"

async def dispatch_openai_requests(
    messages_list,
    model="gpt-3.5-turbo-0613",
    max_tokens=4,
    ):
    """Dispatches requests to OpenAI API asynchronously."""
    async_responses = [
        openai.ChatCompletion.acreate(
            model=model,
            messages=x,
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            request_timeout=20,
        )
        for x in messages_list
    ]
    time.sleep(1)
    return await asyncio.gather(*async_responses)

def evaluate_result(preds, labels, types):
    types = [t[:2] if not t.startswith("2i_neg") else "2i_neg" for t in types]
    all_types = ["2i", "2i_neg", "3i", "2p", "ip", "pi"]
    return {
        "accuracy": accuracy_score(preds, labels),
        "acc_by_types": dict(
            [(t, accuracy_score(np.array(preds)[np.array(types)==t], np.array(labels)[np.array(types)==t])) for t in all_types]
        ),
    }

relation_prompt_new = {
    "xIntent": "the intention of PersonX before",
    "xNeed": "what PersonX needed to do before",
    "xWant": "what PersonX wants to do after",
    "xEffect": "the effect on PersonX after",
    "xReact": "what PersonX feels after",
    "xAttr": "what PersonX is seen as given",
    "oEffect": "the effect on PersonY after",
    "oReact": "what PersonY feels after",
    "oWant": "what PersonY wants to do after",
    "HinderedBy": "what hindered",
    "isAfter": "what happens before",
    "isBefore": "what happens after",
}