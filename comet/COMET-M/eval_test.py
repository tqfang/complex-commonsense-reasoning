

from collections import defaultdict
import sys
sys.path.append("utils/comet-m")
sys.path.append("../../../utils/comet-m")
from rouge_score import rouge_scorer, scoring
import pandas as pd
import argparse
import numpy as np
import json
import os
from collections import defaultdict
import random
import torch
from tqdm import tqdm
from pathlib import Path

from nltk import bleu
from nltk.translate import meteor
from rouge import Rouge
from nltk.translate.bleu_score import SmoothingFunction
from collections import defaultdict
# from system_eval.evaluation.bert_score.bert_score import BertScore
import sys
from comet_atomic2020_bart.utils import calculate_rouge, use_task_specific_params, calculate_bleu_score, trim_batch
from comet_m.functions import beam_generations_new, get_gen_comet_m
import nltk
from typing import List
def read_jsonl_lines(input_file: str) -> List[dict]:
    with open(input_file) as f:
        lines = f.readlines()
        return [json.loads(l.strip()) for l in lines]

from evaluate import load
bertscore = load("bertscore")
# config.PRED_TARGET = "../../../../data/comet-m/test.target"


def get_llama_answer(strs, zero_shot=False):
    if zero_shot:
        answer_str = "The answer is:\n [/INST]"
        end_token = "</s>"
        
        
    else:
        answer_str = "The answer is:\n<|im_end|> \n<|im_start|> answer\n"
        end_token = "<|im_end|>"
        
        
    idx = strs.find(answer_str)
    answer = strs[idx+len(answer_str):]

    idx_end = answer.find(end_token)
    return answer[:idx_end] 

def evaluate_comet_m(pred_file, source_path, target_path, out_dir, llama=False, rm_none=False, zero_shot=False):
    # rm_none: replace none/nan as "", and keep "". (in previous ones it's not kept)
    pred_generations = read_jsonl_lines(pred_file)

    sources = [line.strip() for line in open(source_path).readlines()]
    targets = [line.strip() for line in open(target_path).readlines()]
    
    gold = defaultdict(list)
    for source, target in zip(sources, targets):
        # if source in predictions:
        curr_gold = target
        gold[source].append(curr_gold)

    pred_dataset = pd.DataFrame({"head_event":sources, "tail_event":""})
    pred_dataset = pred_dataset.drop_duplicates(['head_event'], ignore_index=True)

    if not llama:
        generations = [[get_gen_comet_m(g) for g in item['generations'] ] for item in pred_generations]
    else:
        generations = [[get_llama_answer(g, zero_shot=zero_shot) for g in item['generations'] ] for item in pred_generations]
    if rm_none:
        generations = [[g if not g.lower().strip() in ['nan', 'none'] else "" for g in item ] for item in generations]
    
    predictions = defaultdict(set)

    for source, gens in zip(pred_dataset["head_event"], generations):

        if rm_none:
            curr_preds = set([pred for pred in gens])
        else:
            curr_preds = set([pred for pred in gens if len(pred) > 0])
        if len(curr_preds) > 0:
            predictions[source] = predictions[source].union(curr_preds)

    for source in predictions:
        if len(predictions[source]) == 0:
            predictions[source] = set([''])
    # calculate

    print("len gold", len(gold), "len preds", len(predictions))

    bleu1_scores, bleu2_scores, bleu3_scores, bleu4_scores, rouge_scores, meteor_scores = [], [],[],[],[],[]
    smoothing = SmoothingFunction().method4
    rouge = Rouge()

    inputs = []
    generations = []
    references = []
    all_scores = []
    ngram_diversity_scores = []

    inf1 = defaultdict(list)
    inf2 = defaultdict(set)
    for input, curr_gold in gold.items():
        curr_predictions = list(predictions[input])
        # The refs and gold must be in the same size
        length = min(len(curr_gold), len(curr_predictions))
        if length > 0:
            hyps = curr_predictions
            refs = curr_gold
            # DIVERSITY
            # ngram_diversity_scores.append(measure_ngram_diversity(hyps))
            # ROUGE
            # 
            
            if rm_none:
                scores = [np.max([rouge.get_scores(p, g)[0]["rouge-l"]["f"] if len(p) > 0 else 0 for g in refs]) for p in hyps]
            else:
                scores = [np.max([rouge.get_scores(p, g)[0]["rouge-l"]["f"] for g in refs]) for p in hyps]

            sorted_scores = [s for s, x in sorted(zip(scores, hyps), reverse=True)][:length]
            sorted_hyps = [x for _, x in sorted(zip(scores, hyps), reverse=True)][:length]
            scores = sorted_scores
            hyps = sorted_hyps
            # print(input, hyps, refs, scores)
            inputs.append(input)
            generations.append(hyps)
            references.append(refs)
            rouge_scores.extend(list(scores))
            all_scores.append(scores)
            # BLEU
            hyps = [tuple(h.split()) for h in hyps]
            refs = [tuple(r.split()) for r in refs]
            bleu1_scores.extend([bleu(refs, pred, smoothing_function=smoothing, weights=[1.0]) for pred in hyps])
            bleu2_scores.extend([bleu(refs, pred, smoothing_function=smoothing, weights=[0.5, 0.5]) for pred in hyps])
            bleu3_scores.extend([bleu(refs, pred, smoothing_function=smoothing, weights=[0.34, 0.33, 0.33]) for pred in hyps])
            bleu4_scores.extend([bleu(refs, pred, smoothing_function=smoothing, weights=[0.25, 0.25, 0.25, 0.25]) for pred in hyps])
            # meteor_scores.extend([meteor(refs, pred) for pred in hyps])
            # Top 2 Inferences for BERT score
            if len(hyps) > 1:
                inf1[input].append(hyps[0])
                inf2[input] = predictions[input].union(hyps[1])

    # sys.path.append("../")
    # from system_eval.evaluation.bert_score.bert_score import BertScore
    # scorer = BertScore()

    # sem_sim, scores = scorer.compute_score(gold, predictions)
    # sem_sim_self, scores = scorer.compute_score(inf1, inf2)
    sem_sim = 0
    bleu2 = 100.0 * np.mean(bleu2_scores)
    bleu4 = 100.0 * np.mean(bleu4_scores)
    rougel = 100.0 * np.mean(rouge_scores)
    # ngram_diversity = 100.0 * np.mean(ngram_diversity_scores)

    gold_bertscore = [gold[source] for source in gold]
    pred_bertscore = [predictions[source] for source in gold]

    gold_flatten = []
    pred_flatten = []
    for refs, preds in zip(gold_bertscore,pred_bertscore):
        pred_flatten.extend(preds)
        for p in preds:
            gold_flatten.append(refs)


    bert_score_results = bertscore.compute(predictions=pred_flatten,
                                                references=gold_flatten,
                                                model_type="bert-base-uncased")

    print("\t".join([f"Bleu-2: {bleu2:.3f}", f"Bleu-4: {bleu4:.3f}", f"Rouge-L: {rougel:.3f}", f"BertScore: {np.mean(bert_score_results['f1']):.3f}"])) 


    with open(out_dir + "/real_results.txt", "w") as f:
        f.write( f"Rouge-L: {rougel:.3f}" + "\n")
        f.write(f"Bleu-2: {bleu2:.3f}"+ "\n")
        f.write(f"Bleu-4: {bleu4:.3f}" + "\n")
        f.write(f"BertScore: {sem_sim:.3f}" + "\n")
        # f.write(f"Average BERTScore_TopInferences: {sem_sim_self:.3f}" + "\n")

# pred_generations = read_jsonl_lines("")

# out_dir = "data/model/comet-m/2i-mix-25-e3-comet-gpt2-large-bs16-e2-s42/"
source_path = "data/comet-m/test.source"
target_path = "data/comet-m/test.target"

evaluate_comet_m(pred_file, source_path, target_path, os.path.dirname(pred_file))