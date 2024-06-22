import json
import os
import pandas as pd
import argparse
from transformers import AutoTokenizer, LlamaForCausalLM
from tqdm import tqdm
import sys
sys.path.append(os.getcwd())
from paracomet.utils import rel_to_question
from paracomet.utils import NL_to_rel

def read_jsonl_lines(input_file: str):
    with open(input_file) as f:
        lines = f.readlines()
        return [json.loads(l.strip()) for l in lines]
def write_items(output_file, items):
    with open(output_file, 'w') as f:
        for concept in items:
            f.write(concept + "\n")
    f.close()
def to_cuda(d):
    return {key:d[key].to("cuda") for key in d}

DEBUG=False
NUM_INST = 10


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--pred_file', default="data/paracomet/gold_set.jsonl")
    parser.add_argument('--our_dir', required=True) # "data/llama-2-7b-atomic2020_name"
    parser.add_argument('--zero_shot', action='store_true', help='whether to use zero-shot prompt')

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = LlamaForCausalLM.from_pretrained(args.model_path)
    model.to("cuda")

    pred_dataset = read_jsonl_lines(args.pred_file)

    sentences = [item['story'].replace('!" ', '. ').replace('! ', '. ').replace('? ', '. ').split('. ') for item in pred_dataset]
    target_sentences = [sents[int(item['sentID'])] for sents, item in zip(sentences, pred_dataset)]
    input_stories = [item['story'].replace(target_sent, '<TGT> '+target_sent+' <TGT>') for item, target_sent in zip(pred_dataset, target_sentences)]
    
    relations = [rel_to_question[NL_to_rel[item['prefix']]] for item in pred_dataset]

    all_prompts = []

    for rel, story, target in zip(relations, [item['story'] for item in pred_dataset], target_sentences):
        context = story
        question = rel + ' ' + target
        if not args.zero_shot:
            prompt = f"As an expert in commonsense reasoning, your task is to provide a concise response to a question based on the given context. The question focuses on studying the causes, effects, or attributes of personas related to the given context. \nContext: {context}\nQuestion: {question}\nThe answer is:\n"
            prompt = f"<|im_start|>question\n{prompt}<|im_end|>\n<|im_start|>answer\n"
        else:
            system_message = "As an expert in commonsense reasoning, your task is to provide a concise response to a question based on the given context. The question focuses on studying the causes, effects, or attributes of personas related to the given context. Answer shortly with no more than 5 words."
            prompt = f"Context: {context}\nQuestion: {question}\nThe answer is:\n"
            prompt = f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{prompt} [/INST]"

        all_prompts.append(prompt)
    

    if DEBUG:
        all_prompts = all_prompts[:NUM_INST]
        print(all_prompts)



    pred_generations = []
    for prompt in tqdm(all_prompts):
        inputs = to_cuda(tokenizer(prompt, return_tensors="pt"))
        generate_ids = model.generate(inputs["input_ids"], max_new_tokens=30, num_beams=10, num_return_sequences=1)
        pred_generations.append(
            {
            'input': prompt,
            'generations': tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
            })

    os.makedirs(args.our_dir, exist_ok=True)
    write_items(os.path.join(args.our_dir, f"pred_generations.jsonl"),
                [json.dumps(r) for r in pred_generations])


