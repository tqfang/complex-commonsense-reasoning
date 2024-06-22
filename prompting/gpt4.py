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

from utils import candidate_template, dispatch_openai_requests, evaluate_result
from utils import relation_prompt_new

openai.api_key = os.getenv("OPENAI_API_KEY")

df = pd.read_csv("../cqa_atomic_v1.0_name_new.csv")

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
            preds = await dispatch_openai_requests(messages_list = msg_list, max_tokens=2, model='gpt-4-1106-preview')
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

evaluate_result(results_zs_id, ground_truth_option, list(df['type']) )

def write_results_to_file(preds, out_file, overwrite=False):
	if not overwrite and os.path.exists(out_file):
		raise FileExistsError
	with open(out_file, "w") as writer:
		writer.writelines(json.dumps(preds))

# json.load(open(file_name))
write_results_to_file(results_zs_id, "data/gpt4_pred/zs_pred_id.json")
write_results_to_file(zs_pred, "data/gpt4_pred/zs_pred.json", overwrite=True)

##############################
##### 2. zero-shot CoT
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
		{"role": "user", "content": prompt + "\nLet's think step by step."},
	]
	prompt_list.append(messages)

bs = 5
zs_cot_pred = []
for i in tqdm(range(0, len(prompt_list), bs)):
    msg_list = prompt_list[i:i+bs]
    while True:
        try:
            preds = await dispatch_openai_requests(messages_list = msg_list, max_tokens=256)
            break
        except Exception as e:
            print(e)
            time.sleep(10)
        # time.sleep(0.5)
    zs_cot_pred.extend([p["choices"][0]["message"]["content"].strip() for p in preds])


pattern_zs_cot = r"[ABCDE]:"

def parse_zs_cot_answer(text):

    for kw in ['answer is', 'answer would be']:
        if kw in text:
            ans_text = text.split("answer is")[-1].strip()
            try:
                ans = re.findall(pattern_zs_cot, ans_text)[0][0]
            except:
                ans = 'E'
            return ans
    
    # else
    return 'E'

answer_2_idx = {"A":0, "B":1, "C":2, "D":3, "E":4}
results_zs_cot_id = [answer_2_idx[parse_zs_cot_answer(text)] for text in zs_cot_pred]

evaluate_result(results_zs_cot_id, ground_truth_option, list(df['type'].apply(lambda x:x[:2])) )
evaluate_result(results_zs_cot_id, ground_truth_option, list(df['type']) )

write_results_to_file(results_zs_cot_id, "data/gpt4_pred/zs_cot_pred_id.json")
write_results_to_file(zs_cot_pred, "data/gpt4_pred/zs_cot_pred.json")




def question_template_new(e1, e2, r1, r2):
	return f"What event or state is both {relation_prompt_new[r1]} {e1} and also {relation_prompt_new[r2]} {e2}?"

context = "Michael strengthens Michael's position and Michael makes things better"
h1 = "Michael strengthens Michael's position"
h2 = "Michael makes things better"
r1 = "xWant"
r2 = "xWant"
gt = "Michael gets promoted"
negs = ['Michael give all the reasons why they should come',
       'The dogs ate their books.', 'Michael Fail', 'None of the answers are correct.']
effect_cot = "Strengthening their position and improving things would indicate that Michael is effectively carrying out their responsibilities and contributing positively to the organization or situation. And in many cases, one of the natural rewards for such performance is a promotion, which reflects recognition and advancement within the hierarchy. If Michael strengthens their position and makes things better, the most logical outcome that aligns with their goals is to get promoted. Therefore, B, \"PersonX gets promoted,\" is the event or state that aligns with PersonX's goals after both events."
effect_exemp = question_template_new(h1, h2, r1, r2) + "\n" + candidate_template([negs[0], gt, negs[1], negs[2], negs[3]])
effect_exemp_ans = "B"



context = "Gianna learns everything and Gianna passes Gianna's exams"
h1 = "Gianna learns everything"
h2 = "Gianna passes Gianna's exams"
r1 = "xNeed"
r2 = "xNeed"
gt = "Gianna attend classes"
negs = ['Mike threatened Gianna.', 'Gianna see if people liked it',
       'Gianna has never seen a horse before.', 'None of the answers are correct.']
cause_cot = "In order for Gianna to learn everything, they needed to attend their classes. Attending classes is a fundamental step towards gaining knowledge and understanding the material being taught. By actively participating in classes, Gianna is also able to pass the exam. Therefore, attending classes is the event or state that is both what Gianna needed to do before learning everything and what they needed to do before passing their exams. Thus, option C is the correct answer."
cause_exemp = question_template_new(h1, h2, r1, r2) + "\n" + candidate_template([negs[0], negs[1], gt, negs[2], negs[3]])
cause_exemp_ans = "C"

context = "David like hiking and David stops and smell the roses"
h1 = "David like hiking"
h2 = "David stops and smell the roses"
r1 = "xAttr"
r2 = "xAttr"
gt = "David is nature-loving"
# negs = np.random.choice(all_nodes, 3, replace=False)
negs = ['David smiles because he found his necklace',
       'PersonY make David stop acting the way they do',
       'David Courts wife',
       'None of the answers are correct.'
       ]

attr_cot = "It is stated that David likes hiking, which suggests an appreciation for outdoor activities and the beauty of nature. It is also mentioned that David stops and smells the roses, indicating a deliberate effort to take in and enjoy the natural surroundings. These two statements combined strongly suggest that David has a genuine love for nature. Therefore, A is the most appropriate choice."
attr_exemp = question_template_new(h1, h2, r1, r2) + "\n" + candidate_template([gt, negs[0], negs[1], negs[2], negs[3]])
attr_exemp_ans = "A"


context = "Elias wants to play a game and later Elias buys a game"
h1 = "Elias wants to play a game"
h2 = "Elias buys a game"
r1 = "xWant"
r2 = "xNeed"
gt = "Elias makes a purchase"
negs = ['Elias done for the day', 'Elias put the change away',
       "They're inside a library and can't be loud", 'None of the answers are correct.']
script_cot = "If Elias wants to play a game when he may purchase a game later. In order for Elias to buy a game, they would need to complete the action of making a purchase. Putting change away (B), being done for the day (A), or being inside a library (C) do not directly relate to the actions of forgetting sugar and buying a game. Therefore, the correct answer is D: Elias makes a purchase."
script_exemp = question_template_new(h1, h2, r1, r2) + "\n" + candidate_template([negs[0], negs[1], negs[2], gt, negs[3]])
script_exemp_ans = "D"

context = "Sofia rushes to the dentist and then Sofia didn't warmly kissed Shelly's necked"
h1 = "Sofia rushes to the dentist"
h2 = "Sofia warmly kissed Shelly's necked"
r1 = "xEffect"
r2 = "HinderedBy"
gt = "Sofia is nervous"
negs = ['Sofia they have a drink', "Sofia know Jeremy's feelings",
       'Sofia sees moose', 'None of the answers are correct.']
neg_effect_cot = "Sofia rushing to the dentist implies that they are already in a state of anxiety or nervousness. This state of mind can hinder their ability to warmly kiss Shelly's necked later on because being nervous can affect one's ability to engage in physical intimacy and display affection. Therefore, being nervous can be the common factor that is both a result of rushing to the dentist and a hindrance to warm kissing."
neg_effect_exemp = question_template_new(h1, h2, r1, r2) + "\n" + candidate_template([negs[0], negs[1], negs[2], gt, negs[3]])
neg_effect_exemp_ans = "D"

# context = "Mark bails Nico out and then Mark decides not to quit"
# h2 = "Mark decides to quit"
# h1 = "Mark bails Nico out"
# r1 = "xNeed"
# r2 = "HinderedBy"
# gt = "Mark is scared"
# negs = ['Mark live near bus stop', 'Mark drive the car to mall',
#        'Mark bring the horse back to the stable to rest', 'None of the answers are correct.']
# neg_cause_cot = "The question asks for an event or state that is both what Mark needed to do before deciding to quit and also what hindered baling Nico out. None of the options provided (A, B, C) seem to be relevant to the given context. Option D, \"Mark is scared,\" could potentially be a hindrance to Mark deciding to quit, but it does not align with the event or state that Mark needed to do before bailing Nico out. Therefore, none of the given options are correct."
# neg_cause_exemp = question_template_new(h1, h2, r1, r2) + "\n" + candidate_template([negs[0], negs[1], negs[2], gt, negs[3]])
# neg_cause_exemp_ans = "E"

context = "Mark bails Nico out and then Mark decides not to indulge him further"
h2 = "Mark decides to indulge him further"
h1 = "Mark bails Nico out"
r1 = "xNeed"
r2 = "HinderedBy"
gt = "Mark wants Nico to turn over a new leaf"
negs = ['Mark live near bus stop', 'Mark drive the car to mall',
       'Mark bring the horse back to the stable to rest', 'None of the answers are correct.']
neg_cause_cot = "The question asks for an event or state that is both what Mark needed to do before bailing out Nico and what hindered Mark from indulging Nico further. The reason to bail someone out should be something like Nico is a friend of Mark, Mark wants the best for Nico, and Mark wants Nico to change his way and turn over a new leaf. All can also be the reasons why Mark decides to stop indulging him. Other options are not relevant with the reasons of the context. Therefore, the correct answer is A: Mark wants Nico to turn over a new leaf."
neg_cause_exemp = question_template_new(h1, h2, r1, r2) + "\n" + candidate_template([gt, negs[0], negs[1], negs[2], negs[3]])
neg_cause_exemp_ans = "A"


exemplar_2i = [
	[
		{"role": "user", "content": effect_exemp},
		{"role": "assistant", "content": effect_exemp_ans},
	],
	[
		{"role": "user", "content": cause_exemp},
		{"role": "assistant", "content": cause_exemp_ans},
	],
	[
		{"role": "user", "content": attr_exemp},
		{"role": "assistant", "content": attr_exemp_ans},
	],
	[
		{"role": "user", "content": script_exemp},
		{"role": "assistant", "content": script_exemp_ans},
	],
	[
		{"role": "user", "content": neg_effect_exemp},
		{"role": "assistant", "content": neg_effect_exemp_ans},
	],
	[
		{"role": "user", "content": neg_cause_exemp},
		{"role": "assistant", "content": neg_cause_exemp_ans},
	]
]
exemplar_2i_cot = [
	[
		{"role": "user", "content": effect_exemp},
		{"role": "assistant", "content": effect_cot + "\n\nSo the answer is:\n\n" + effect_exemp_ans},
	],
	[
		{"role": "user", "content": cause_exemp},
		{"role": "assistant", "content": cause_cot + "\n\nSo the answer is:\n\n" + cause_exemp_ans},
	]
	,
	[
		{"role": "user", "content": attr_exemp},
		{"role": "assistant", "content": attr_cot + "\n\nSo the answer is:\n\n" + attr_exemp_ans},
	],
	[
		{"role": "user", "content": script_exemp},
		{"role": "assistant", "content": script_cot + "\n\nSo the answer is:\n\n" + script_exemp_ans},
	],
	[
		{"role": "user", "content": neg_effect_exemp},
		{"role": "assistant", "content": neg_effect_cot + "\n\nSo the answer is:\n\n" + neg_effect_exemp_ans},
	],
	[
		{"role": "user", "content": neg_cause_exemp},
		{"role": "assistant", "content": neg_cause_cot + "\n\nSo the answer is:\n\n" + neg_cause_exemp_ans},
	]
]

def question_template_2p(e1, r1, r2):
	return f"What event or state is {relation_prompt_new[r2]} {relation_prompt_new[r1]} {e1}?"

context = "James does not have enough money"
h1 = "James does not have enough money"
r1 = "xEffect"
r2 = "xEffect"
gt = "James can now afford to buy something he always wanted"
negs = ['James decide when to talk next', 'James is improving',
       'James is happy and affectionate', 'None of the answers are correct.']
effect_2p_cot = "Let's first infer the effect of James not having enough money. As a result of not having money, James would work hard to earn more money, which will result in the fact that James can afford things he wants. Therefore, the correct answer is A: James can now afford to buy something he always wanted."
effect_2p_exemp = question_template_2p(h1, r1, r2) + "\n" + candidate_template([gt, negs[0], negs[1], negs[2], negs[3]])
effect_2p_exemp_ans = "A"

context = "Isabella has a bowl of cereal"
h1 = "Isabella has a bowl of cereal"
r1 = "xNeed"
r2 = "xNeed"
gt = "Isabella buys some milk"
negs = ['Isabella likes boating', 'Isabella loves to eat cereal',
       'Isabella has a kitchen', 'None of the answers are correct.']
cause_2p_cot = "Let's first infer what Isabella needed to do before having a bowl of cereal. In general, Isabella needs milk when eating cereal. So before eating cereal with milk, Isabella needs to have purchased milk. Therefore, the correct answer is C: Isabella buys some milk."
cause_2p_exemp = question_template_2p(h1, r1, r2) + "\n" + candidate_template([negs[0], negs[1], gt, negs[2], negs[3]])
cause_2p_exemp_ans = "C"

exemplar_2p = [
	[
		{"role": "user", "content": effect_2p_exemp},
		{"role": "assistant", "content": effect_2p_exemp_ans},
	],
	[
		{"role": "user", "content": cause_2p_exemp},
		{"role": "assistant", "content": cause_2p_exemp_ans},
	],
]

exemplar_2p_cot = [
	[
		{"role": "user", "content": effect_2p_exemp},
		{"role": "assistant", "content": effect_2p_cot + "\n\nSo the answer is:\n\n" + effect_2p_exemp_ans},
	],
	[
		{"role": "user", "content": cause_2p_exemp},
		{"role": "assistant", "content": cause_2p_cot + "\n\nSo the answer is:\n\n" + cause_2p_exemp_ans},
	],
]

##############################
##### 3. one-shot (2i)
##############################

prompt_list = []

for i, item in df.iterrows():
	question = item['question']
	context = item['context']
	options = json.loads(item['options'])
	options_text = candidate_template(options)
	prompt = question + "\n\n" + options_text

	messages= [
		{"role": "system", "content": "Answer this commonsense reasoning question, where you are supposed to handle a multiple-chioce question answering task to select the correct answer. Select one correct answer from A to E."},
	] + exemplar_2i[1] + \
	[
		{"role": "user", "content": prompt},
	]
	prompt_list.append(messages)

bs = 5
oneshot_pred = []
for i in tqdm(range(0, len(prompt_list), bs)):
    msg_list = prompt_list[i:i+bs]
    while True:
        try:
            preds = await dispatch_openai_requests(messages_list = msg_list, max_tokens=4, model='gpt-4-1106-preview')
            break
        except Exception as e:
            print(e)
            time.sleep(10)
        # time.sleep(0.5)
    oneshot_pred.extend([p["choices"][0]["message"]["content"].strip() for p in preds])

def parse_one_shot(text):
    if text in answer_2_idx.keys():
        return text
    else:
        return parse_zs_answer(text)
    
results_one_shot_id = [answer_2_idx[parse_one_shot(text)] for text in oneshot_pred]
# evaluate_result(results_one_shot_id, ground_truth_option, list(df['type'].apply(lambda x:x[:2])) )
write_results_to_file(results_one_shot_id, "data/gpt4_pred/oneshot_pred_id_cause.json")
write_results_to_file(oneshot_pred, "data/gpt4_pred/oneshot_pred_cause.json")
evaluate_result(results_one_shot_id, ground_truth_option, list(df['type']) )




##############################
##### 3. one-shot CoT
##############################

prompt_list = []

for i, item in df.iterrows():
	question = item['question']
	context = item['context']
	options = json.loads(item['options'])
	options_text = candidate_template(options)
	prompt = question + "\n\n" + options_text

	messages= [
		{"role": "system", "content": "Answer this commonsense reasoning question, where you are supposed to handle a multiple-chioce question answering task to select the correct answer. Select one correct answer from A to E."},
	] + exemplar_2i_cot[1] + \
	[
		{"role": "user", "content": prompt},
	]
	prompt_list.append(messages)

bs = 5
oneshot_cot_pred = []
for i in tqdm(range(0, len(prompt_list), bs)):
    msg_list = prompt_list[i:i+bs]
    while True:
        try:
            preds = await dispatch_openai_requests(messages_list = msg_list, max_tokens=256, model='gpt-4-1106-preview')
            break
        except Exception as e:
            print(e)
            time.sleep(10)
        # time.sleep(0.5)
    oneshot_cot_pred.extend([p["choices"][0]["message"]["content"].strip() for p in preds])

pattern_1s = r"[ABCDE][:)]?"   
def parse_1s_cot_answer(text):

    for kw in ['answer is', 'answer would be',  "Answer:"]:
        if kw in text:
            ans_text = text.split(kw)[-1].strip()
            try:
                ans = re.findall(pattern_1s, ans_text)[0][0]
            except:
                ans = 'E'
                print(text, "<<<END>>>")
            return ans
    
    return 'E'

results_1s_cot_id = [answer_2_idx[parse_1s_cot_answer(text)] for text in oneshot_cot_pred]
# evaluate_result(results_1s_cot_id, ground_truth_option, list(df['type'].apply(lambda x:x[:2])) )
evaluate_result(results_1s_cot_id, ground_truth_option, list(df['type']) )
# evaluate_result(results_zs_id, ground_truth_option, list(df['type']) )

write_results_to_file(results_1s_cot_id, "data/gpt4_pred/oneshot_cot_pred_id_cause.json")
write_results_to_file(oneshot_cot_pred, "data/gpt4_pred/oneshot_cot_pred_cause.json")

##############################
##### 4. one-shot (2p)
##############################

prompt_list = []

for i, item in df.iterrows():
	question = item['question']
	context = item['context']
	options = json.loads(item['options'])
	options_text = candidate_template(options)
	prompt = question + "\n\n" + options_text

	messages= [
		{"role": "system", "content": "Answer this commonsense reasoning question, where you are supposed to handle a multiple-chioce question answering task to select the correct answer. Select one correct answer from A to E."},
	] + exemplar_2p[0] + \
	[
		{"role": "user", "content": prompt},
	]
	prompt_list.append(messages)

bs = 5
oneshot_pred = []
for i in tqdm(range(0, len(prompt_list), bs)):
    msg_list = prompt_list[i:i+bs]
    while True:
        try:
            preds = await dispatch_openai_requests(messages_list = msg_list, max_tokens=4)
            break
        except Exception as e:
            print(e)
            time.sleep(10)
        # time.sleep(0.5)
    oneshot_pred.extend([p["choices"][0]["message"]["content"].strip() for p in preds])

def parse_one_shot(text):
    if text in answer_2_idx.keys():
        return text
    else:
        return parse_zs_answer(text)
    
results_one_shot_id = [answer_2_idx[parse_one_shot(text)] for text in oneshot_pred]
# evaluate_result(results_one_shot_id, ground_truth_option, list(df['type'].apply(lambda x:x[:2])) )
evaluate_result(results_one_shot_id, ground_truth_option, list(df['type']) )

write_results_to_file(results_one_shot_id, "data/gpt4_pred/oneshot_pred_id_2p_effect.json")
write_results_to_file(oneshot_pred, "data/gpt4_pred/oneshot_pred_2p_effect.json")



##############################
# 2-shot 2p
##############################

prompt_list = []

for i, item in df.iterrows():
	question = item['question']
	context = item['context']
	options = json.loads(item['options'])
	options_text = candidate_template(options)
	prompt = question + "\n\n" + options_text

	messages= [
		{"role": "system", "content": "Answer this commonsense reasoning question, where you are supposed to handle a multiple-chioce question answering task to select the correct answer. Select one correct answer from A to E."},
	] + exemplar_2p[0] + exemplar_2p[1] + \
	[
		{"role": "user", "content": prompt},
	]
	prompt_list.append(messages)

bs = 5
twoshot_pred = []
for i in tqdm(range(0, len(prompt_list), bs)):
    msg_list = prompt_list[i:i+bs]
    while True:
        try:
            preds = await dispatch_openai_requests(messages_list = msg_list, max_tokens=4)
            break
        except Exception as e:
            print(e)
            time.sleep(10)
        # time.sleep(0.5)
    twoshot_pred.extend([p["choices"][0]["message"]["content"].strip() for p in preds])

def parse_one_shot(text):
    if text in answer_2_idx.keys():
        return text
    else:
        return parse_zs_answer(text)
    
results_two_shot_id = [answer_2_idx[parse_one_shot(text)] for text in twoshot_pred]

write_results_to_file(results_two_shot_id, "data/gpt4_pred/twoshot_pred_id_2p.json")
write_results_to_file(twoshot_pred, "data/gpt4_pred/twoshot_pred_2p.json")
evaluate_result(results_two_shot_id, ground_truth_option, list(df['type']) )

##############################
##### 4. one-shot (2p) CoT
##############################

prompt_list = []

for i, item in df.iterrows():
	question = item['question']
	context = item['context']
	options = json.loads(item['options'])
	options_text = candidate_template(options)
	prompt = question + "\n\n" + options_text

	messages= [
		{"role": "system", "content": "Answer this commonsense reasoning question, where you are supposed to handle a multiple-chioce question answering task to select the correct answer. Select one correct answer from A to E."},
	] + exemplar_2p_cot[0] + \
	[
		{"role": "user", "content": prompt},
	]
	prompt_list.append(messages)

bs = 5
oneshot_cot_pred = []
for i in tqdm(range(0, len(prompt_list), bs)):
    msg_list = prompt_list[i:i+bs]
    while True:
        try:
            preds = await dispatch_openai_requests(messages_list = msg_list, max_tokens=256)
            break
        except Exception as e:
            print(e)
            time.sleep(10)
        # time.sleep(0.5)
    oneshot_cot_pred.extend([p["choices"][0]["message"]["content"].strip() for p in preds])

pattern_1s = r"[ABCDE][:)]?"   
def parse_1s_cot_answer(text):

    for kw in ['answer is', 'answer would be',  "Answer:"]:
        if kw in text:
            ans_text = text.split(kw)[-1].strip()
            try:
                ans = re.findall(pattern_1s, ans_text)[0][0]
            except:
                ans = 'E'
                print(text, "<<<END>>>")
            return ans
    
    return 'E'

results_1s_cot_id = [answer_2_idx[parse_1s_cot_answer(text)] for text in oneshot_cot_pred]
# evaluate_result(results_1s_cot_id, ground_truth_option, list(df['type'].apply(lambda x:x[:2])) )
evaluate_result(results_1s_cot_id, ground_truth_option, list(df['type']) )
# evaluate_result(results_zs_id, ground_truth_option, list(df['type']) )
write_results_to_file(results_1s_cot_id, "data/gpt4_pred/oneshot_cot_pred_id_2p_effect.json")
write_results_to_file(oneshot_cot_pred, "data/gpt4_pred/oneshot_cot_pred_2p_effect.json")

##############################
### 2-shot CoT
##############################

prompt_list = []

for i, item in df.iterrows():
	question = item['question']
	context = item['context']
	options = json.loads(item['options'])
	options_text = candidate_template(options)
	prompt = question + "\n\n" + options_text

	messages= [
		{"role": "system", "content": "Answer this commonsense reasoning question, where you are supposed to handle a multiple-chioce question answering task to select the correct answer. Select one correct answer from A to E."},
	] + exemplar_2p_cot[0] + exemplar_2p_cot[1] + \
	[
		{"role": "user", "content": prompt},
	]
	prompt_list.append(messages)

bs = 5
twoshot_cot_pred = []
for i in tqdm(range(0, len(prompt_list), bs)):
    msg_list = prompt_list[i:i+bs]
    while True:
        try:
            preds = await dispatch_openai_requests(messages_list = msg_list, max_tokens=256)
            break
        except Exception as e:
            print(e)
            time.sleep(10)
        # time.sleep(0.5)
    twoshot_cot_pred.extend([p["choices"][0]["message"]["content"].strip() for p in preds])

results_2s_cot_id = [answer_2_idx[parse_1s_cot_answer(text)] for text in twoshot_cot_pred]
# evaluate_result(results_2s_cot_id, ground_truth_option, list(df['type'].apply(lambda x:x[:2])) )
evaluate_result(results_2s_cot_id, ground_truth_option, list(df['type']) )
# evaluate_result(results_zs_id, ground_truth_option, list(df['type']) )
write_results_to_file(results_2s_cot_id, "data/gpt4_pred/twoshot_cot_pred_id_2p_effect.json")
write_results_to_file(twoshot_cot_pred, "data/gpt4_pred/twoshot_cot_pred_2p_effect.json")

##############################
##### 5. six-shot (2i)
##############################

from itertools import chain
prompt_list = []

for i, item in df.iterrows():
	question = item['question']
	context = item['context']
	options = json.loads(item['options'])
	options_text = candidate_template(options)
	prompt = question + "\n\n" + options_text

	messages= [
		{"role": "system", "content": "Answer this commonsense reasoning question, where you are supposed to handle a multiple-chioce question answering task to select the correct answer. Select one correct answer from A to E."},
	] + list(chain(*exemplar_2i)) + \
	[
		{"role": "user", "content": prompt},
	]
	prompt_list.append(messages)

bs = 5
sixshot_pred = []
for i in tqdm(range(0, len(prompt_list), bs)):
    msg_list = prompt_list[i:i+bs]
    while True:
        try:
            preds = await dispatch_openai_requests(messages_list = msg_list, max_tokens=4)
            break
        except Exception as e:
            print(e)
            time.sleep(10)
        # time.sleep(0.5)
    sixshot_pred.extend([p["choices"][0]["message"]["content"].strip() for p in preds])

def parse_one_shot(text):
    if text in answer_2_idx.keys():
        return text
    else:
        return parse_zs_answer(text)
    
results_six_shot_id = [answer_2_idx[parse_one_shot(text)] for text in sixshot_pred]
# evaluate_result(results_six_shot_id, ground_truth_option, list(df['type'].apply(lambda x:x[:2])) )
evaluate_result(results_six_shot_id, ground_truth_option, list(df['type']) )


write_results_to_file(results_six_shot_id, "data/gpt4_pred/sixshot_pred_id_2i.json")
write_results_to_file(sixshot_pred, "data/gpt4_pred/sixshot_pred_2i.json")

##############################
####### six shot COT
##############################

prompt_list = []

for i, item in df.iterrows():
	question = item['question']
	context = item['context']
	options = json.loads(item['options'])
	options_text = candidate_template(options)
	prompt = question + "\n\n" + options_text

	messages= [
		{"role": "system", "content": "Answer this commonsense reasoning question, where you are supposed to handle a multiple-chioce question answering task to select the correct answer. Select one correct answer from A to E."},
	] + list(chain(*exemplar_2i_cot)) + \
	[
		{"role": "user", "content": prompt},
	]
	prompt_list.append(messages)

bs = 5
sixshot_cot_pred = []
for i in tqdm(range(0, len(prompt_list), bs)):
    msg_list = prompt_list[i:i+bs]
    while True:
        try:
            preds = await dispatch_openai_requests(messages_list = msg_list, max_tokens=256)
            break
        except Exception as e:
            print(e)
            time.sleep(10)
        # time.sleep(0.5)
    sixshot_cot_pred.extend([p["choices"][0]["message"]["content"].strip() for p in preds])

pattern_1s = r"[ABCDE][:)]?"   
def parse_1s_cot_answer(text):

    for kw in ['answer is', 'answer would be',  "Answer:"]:
        if kw in text:
            ans_text = text.split(kw)[-1].strip()
            try:
                ans = re.findall(pattern_1s, ans_text)[0][0]
            except:
                ans = 'E'
                print(text, "<<<END>>>")
            return ans
    
    return 'E'

results_6s_cot_id = [answer_2_idx[parse_1s_cot_answer(text)] for text in sixshot_cot_pred]
# evaluate_result(results_6s_cot_id, ground_truth_option, list(df['type'].apply(lambda x:x[:2])) )
evaluate_result(results_6s_cot_id, ground_truth_option, list(df['type']) )
# evaluate_result(results_zs_id, ground_truth_option, list(df['type']) )

write_results_to_file(results_6s_cot_id, "data/gpt4_pred/sixshot_cot_pred_id.json")
write_results_to_file(sixshot_cot_pred, "data/gpt4_pred/sixshot_cot_pred.json")

##############################
##### 7. 8-shot (2p+2i)
##############################

from itertools import chain
prompt_list = []

for i, item in df.iterrows():
	question = item['question']
	context = item['context']
	options = json.loads(item['options'])
	options_text = candidate_template(options)
	prompt = question + "\n\n" + options_text

	messages= [
		{"role": "system", "content": "Answer this commonsense reasoning question, where you are supposed to handle a multiple-chioce question answering task to select the correct answer. Select one correct answer from A to E."},
	] + list(chain(*exemplar_2i)) + list(chain(*exemplar_2p)) + \
	[
		{"role": "user", "content": prompt},
	]
	prompt_list.append(messages)

bs = 5
eightshot_pred = []
for i in tqdm(range(0, len(prompt_list), bs)):
    msg_list = prompt_list[i:i+bs]
    while True:
        try:
            preds = await dispatch_openai_requests(messages_list = msg_list, max_tokens=2, model='gpt-4-1106-preview')
            break
        except Exception as e:
            print(e)
            time.sleep(10)
        # time.sleep(0.5)
    eightshot_pred.extend([p["choices"][0]["message"]["content"].strip() for p in preds])

def parse_one_shot(text):
    if text in answer_2_idx.keys():
        return text
    else:
        return parse_zs_answer(text)
    
results_eight_shot_id = [answer_2_idx[parse_one_shot(text)] for text in eightshot_pred]
# evaluate_result(results_eight_shot_id, ground_truth_option, list(df['type'].apply(lambda x:x[:2])) )
write_results_to_file(results_eight_shot_id, "data/gpt4_pred/eightshot_pred_id.json", overwrite=True)
write_results_to_file(eightshot_pred, "data/gpt4_pred/eightshot_pred.json", overwrite=True)
evaluate_result(results_eight_shot_id, ground_truth_option, list(df['type']) )

##############################
####### CoT 8-shot
##############################


prompt_list = []

for i, item in df.iterrows():
	question = item['question']
	context = item['context']
	options = json.loads(item['options'])
	options_text = candidate_template(options)
	prompt = question + "\n\n" + options_text

	messages= [
		{"role": "system", "content": "Answer this commonsense reasoning question, where you are supposed to handle a multiple-chioce question answering task to select the correct answer. Select one correct answer from A to E."},
	] + list(chain(*exemplar_2p_cot)) + list(chain(*exemplar_2i_cot))  + \
	[
		{"role": "user", "content": prompt},
	]
	prompt_list.append(messages)

bs = 5
eightshot_cot_pred = []
for i in tqdm(range(0, len(prompt_list), bs)):
    msg_list = prompt_list[i:i+bs]
    while True:
        try:
            preds = await dispatch_openai_requests(messages_list = msg_list, max_tokens=256, model='gpt-4-1106-preview')
            break
        except Exception as e:
            print(e)
            time.sleep(10)
        # time.sleep(0.5)
    eightshot_cot_pred.extend([p["choices"][0]["message"]["content"].strip() for p in preds])

pattern_1s = r"[ABCDE][:)]?"   
def parse_1s_cot_answer(text):

    for kw in ['answer is', 'answer would be',  "Answer:"]:
        if kw in text:
            ans_text = text.split(kw)[-1].strip()
            try:
                ans = re.findall(pattern_1s, ans_text)[0][0]
            except:
                ans = 'E'
                print(text, "<<<END>>>")
            return ans
    
    return 'E'

results_8s_cot_id = [answer_2_idx[parse_1s_cot_answer(text)] for text in eightshot_cot_pred]
# evaluate_result(results_8s_cot_id, ground_truth_option, list(df['type'].apply(lambda x:x[:2])) )
write_results_to_file(results_8s_cot_id, "data/gpt4_pred/eightshot_cot_pred_id_update.json", overwrite=True)
write_results_to_file(eightshot_cot_pred, "data/gpt4_pred/eightshot_cot_pred_update.json", overwrite=True)
evaluate_result(results_8s_cot_id, ground_truth_option, list(df['type']) )
# evaluate_result(results_zs_id, ground_truth_option, list(df['type']) )





