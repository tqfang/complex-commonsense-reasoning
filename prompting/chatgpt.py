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
openai.api_key = os.getenv("OPENAI_API_KEY")

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

from sklearn.metrics import accuracy_score

def candidate_template(options):
    return f"A: {options[0]}\nB: {options[1]}\nC: {options[2]}\nD: {options[3]}\nE: {options[4]}"

df = pd.read_csv("../AMT/results/cqa_atomic_v1.0_name.csv")

##############################
##### 1. zero-shot
##############################

prompt_list = []

for i, item in df.iterrows():
	question = item['question']
	context = item['context']
	options = json.loads(item['options'])
	options_text = candidate_template(options)
	prompt = question + "\n\n" + options_text

	messages=[
		{"role": "system", "content": "Answer this commonsense reasoning question, where you are supposed to handle a multiple-chioce question answering task to select the correct answer. Select one correct answer from A to E."},
		{"role": "user", "content": prompt},
	]
	prompt_list.append(messages)

bs = 5
zs_pred = []
for i in tqdm(range(0, len(prompt_list), bs)):
    msg_list = prompt_list[i:i+bs]
    while True:
        try:
            preds = await dispatch_openai_requests(messages_list = msg_list, max_tokens=128)
            break
        except Exception as e:
            print(e)
            time.sleep(10)
        time.sleep(0.5)
    zs_pred.extend([p["choices"][0]["message"]["content"].strip() for p in preds])


ground_truth_option = list(df['label'])
pattern_zs_answer = r"[ABCDE]:"
def parse_zs_answer(text):

    try:
        ans = re.findall(pattern_zs_answer, text)[0][0]
    except:
        ans = 'E'
    return ans

results_zs_id = [answer_2_idx[parse_zs_answer(text)] for text in zs_pred]

def evaluate_result(preds, labels, types):
	types = [t[:2] if not t.startswith("2i_neg") else "2i_neg" for t in types]
	all_types = ["2i", "2i_neg", "3i", "2p", "ip", "pi"]
	return {
		"accuracy": accuracy_score(preds, labels),
		"acc_by_types": dict(
			[(t, accuracy_score(np.array(preds)[np.array(types)==t], np.array(labels)[np.array(types)==t])) for t in all_types]
		),
	}

evaluate_result(results_zs_id, ground_truth_option, list(df['type']) )

def write_results_to_file(preds, out_file, overwrite=False):
	if not overwrite and os.path.exists(out_file):
		raise FileExistsError
	with open(out_file, "w") as writer:
		writer.writelines(json.dumps(preds))
# json.load(open(file_name))
write_results_to_file(results_zs_id, "./predictions/zs_pred_id.json")
write_results_to_file(zs_pred, "./predictions/zs_pred.json", overwrite=True)
