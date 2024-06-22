import json
from typing import List
def read_jsonl_lines(input_file: str) -> List[dict]:
    with open(input_file) as f:
        lines = f.readlines()
        return [json.loads(l.strip()) for l in lines]

pred_file = "./pred_generations.jsonl"
pred_generations = read_jsonl_lines(pred_file)

def get_llama_answer(strs):
    answer_str = "The answer is:\n<|im_end|> \n<|im_start|> answer\n"
    idx = strs.find(answer_str)
    answer = strs[idx+len(answer_str):]

    idx_end = answer.find("<|im_end|>")
    return answer[:idx_end] 
