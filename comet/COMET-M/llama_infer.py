import json
import os
import pandas as pd
import argparse
from transformers import AutoTokenizer, LlamaForCausalLM
from tqdm import tqdm

def write_items(output_file, items):
    with open(output_file, 'w') as f:
        for concept in items:
            f.write(concept + "\n")
    f.close()
def to_cuda(d):
    return {key:d[key].to("cuda") for key in d}
def get_tgt(strs):
    strs = strs.split()
    st = 0
    ed = 0
    flag = False
    for i in range(len(strs)):
        if strs[i] == "<TGT>" and flag:
            ed = i
            break
        if strs[i] == "<TGT>" and not flag:
            st = i
            flag = True
        
    return " ".join(strs[st+1:ed])

relation_to_question = {
    "HinderedBy": "What would hinder {} from happening?",
     'Causes' :"What is the result of {}?",
     'isAfter': "What happens before {}?",
     'isBefore': "What happens after {}?",
     'xReason': "What is the reason of {}?",
     'HasPrerequisite': "What is the prerequisite for {}?",
}

DEBUG=False
NUM_INST = 10


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--pred_file', default="data/comet-m/test.source")
    parser.add_argument('--our_dir', required=True) # "data/llama-2-7b-atomic2020_name"
    parser.add_argument('--zero_shot', action='store_true', help='whether to use zero-shot prompt')

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = LlamaForCausalLM.from_pretrained(args.model_path)
    model.to("cuda")

    pred_dataset = pd.DataFrame({"head_event":open(args.pred_file).readlines(), "tail_event":""})
    pred_dataset = pred_dataset.drop_duplicates(['head_event'], ignore_index=True)

    if DEBUG:
        pred_dataset = pred_dataset.head(NUM_INST)


    all_prompts = []
    for h in pred_dataset["head_event"]:
        h = h.strip()
        assert h[-5:] == "[GEN]"
        h = h[:-5].strip()
        r = h.split()[-1]

        assert r in relation_to_question
        tgt = get_tgt(h)
        context = " ".join([t for t in h.split() if t != "<TGT>"][:-1])
        question = relation_to_question[r].format(tgt)
        if not args.zero_shot:
            prompt = f"As an expert in commonsense reasoning, your task is to provide a concise response to a question based on the given context. The question focuses on studying the causes, effects, or attributes of personas related to the given context. \nContext: {context}\nQuestion: {question}\nThe answer is:\n"
            prompt = f"<|im_start|>question\n{prompt}<|im_end|>\n<|im_start|>answer\n"
        else:
            system_message = "As an expert in commonsense reasoning, your task is to provide a concise response to a question based on the given context. The question focuses on studying the causes, effects, or attributes of personas related to the given context. Answer shortly with no more than 5 words."
            prompt = f"Context: {context}\nQuestion: {question}\nThe answer is:\n"
            prompt = f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{prompt} [/INST]"
        all_prompts.append(prompt)



    pred_generations = []
    for prompt in tqdm(all_prompts):
        inputs = to_cuda(tokenizer(prompt, return_tensors="pt"))
        generate_ids = model.generate(inputs["input_ids"], max_new_tokens=30, num_beams=10, num_return_sequences=5)
        pred_generations.append(
            {
            'input': prompt,
            'generations': tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
            })

    os.makedirs(args.our_dir, exist_ok=True)
    write_items(os.path.join(args.our_dir, f"pred_generations.jsonl"),
                [json.dumps(r) for r in pred_generations])


