import json
import pandas as pd
import numpy as np
from nlgeval.pycocoevalcap.bleu.bleu import Bleu
from nlgeval.pycocoevalcap.cider.cider import Cider
from nlgeval.pycocoevalcap.meteor.meteor import Meteor
from nlgeval.pycocoevalcap.rouge.rouge import Rouge
from evaluate import load
from eval.eval_generation import get_llama_answer

def read_jsonl_lines(input_file: str):
    with open(input_file) as f:
        lines = f.readlines()
        return [json.loads(l.strip()) for l in lines]
def get_gen(strs):
    strs = strs.split()
    st = 0
    ed = 0
    for i in range(len(strs)):
        if strs[i] == "[GEN]":
            st = i
        if strs[i] == "[EOS]":
            ed = i
            break
    return " ".join(strs[st+1:ed])


def get_paracomet_eval(pred_file_path, ground_file_path='data/paracomet/gold_set.jsonl', llama=False, zero_shot=False):

    prediction_file = read_jsonl_lines(pred_file_path)
    if llama:  
        predictions = [get_llama_answer(item['generations'][0], zero_shot=zero_shot).replace('[PAD]', '').strip() for item in prediction_file]
    else:
        predictions = [get_gen(item['generations'][0].replace('[EOS]', ' [EOS]').replace('[PAD]', '')).strip() for item in prediction_file]
        # predictions = [get_gen(item['generations'][0]).replace('[PAD]', '').strip() for item in prediction_file]
    ground = [item['rel'] for item in read_jsonl_lines(ground_file_path)]

    hyps = {idx: [strippedlines] for (idx, strippedlines) in enumerate(predictions)}
    refs = {idx: [strippedlines] for (idx, strippedlines) in enumerate(ground)}
    # print(" | ".join([str(round(score, 10)) for score in Bleu(4).compute_score(refs, hyps)[0]] + \
    #                  [str(Rouge().compute_score(refs, hyps)[0]), str(Meteor().compute_score(refs, hyps)[0]), str(Cider().compute_score(refs, hyps)[0])]))
    ret_scores = {}
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    for scorer, method in scorers:
        score, scores = scorer.compute_score(refs, hyps)
        if isinstance(method, list):
            for sc, scs, m in zip(score, scores, method):
                print("%s: %0.6f" % (m, sc))
                ret_scores[m] = sc
        else:
            print("%s: %0.6f" % (method, score))
            ret_scores[method] = score
        if isinstance(scorer, Meteor):
            scorer.close()
    # bertscore
    bertscore = load("bertscore")
    bert_score_results = bertscore.compute(predictions=predictions, 
                                            references=ground, 
                                            model_type="bert-base-uncased")
    import numpy as np
    ret_scores["bertscore"] = np.mean(bert_score_results["f1"])

    del scorers
    return ret_scores