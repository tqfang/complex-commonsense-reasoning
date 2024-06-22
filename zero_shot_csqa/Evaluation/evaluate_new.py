import argparse
import glob
import os

from tqdm import tqdm


parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--model_path", default="../Training/output/Output_ATOMIC-pseudo-wWC/deberta-v3-large_kaixin_baseline", type=str, required=True,
                    help="The train file name")
parser.add_argument("--eval_output_dir", default="./eval_results", type=str, required=True,
                    help="output of the predictions")



eval_tasks = [
    ("socialiqa", "../../../data/mcqa/eval/socialiqa_dev.jsonl"),
    ("winogrande", "../../../data/mcqa/eval/winogrande_dev.jsonl"),
    ("piqa", "../../../data/mcqa/eval/piqa_dev.jsonl"),
    ("commonsenseqa", "../../../data/mcqa/eval/commonsenseqa_dev.jsonl"),
    ("anli", "../../../data/mcqa/eval/commonsenseqa_dev.jsonl"),
    ("complex2i", "../../../data/mcqa/eval/complex2i_dev.jsonl")
]

total_models_to_eval = 0
for f in glob.glob('../Training/Output*'):
    for models in glob.glob('{}/roberta*'.format(f)):
        total_models_to_eval += 1
    for models in glob.glob('{}/deberta-v3-large*'.format(f)):
        total_models_to_eval += 1

progress_bar = tqdm(total=total_models_to_eval)

output_folders = glob.glob('../Training/Output*')
for f in output_folders:
    output_split = f.split('_')[-1]

    for models in glob.glob('{}/roberta*'.format(f)):
        training_data = models.split('_')[-1]
        if not os.path.exists("./eval_results/{}_{}_roberta-large".format(output_split, training_data)):
            for reader, dataset in eval_tasks:
                os.system(
                    """python evaluate_RoBERTa.py --lm {} --dataset_file {} --out_dir {} --device 5 --reader {}""".format(
                        models, dataset, "./eval_results/{}_{}_roberta-large".format(output_split, training_data),
                        reader))
        progress_bar.update(1)

    for models in glob.glob('{}/deberta-v3-large*'.format(f)):
        training_data = models.split('_')[-1]
        if not os.path.exists("./eval_results/{}_{}_deberta-v3-large".format(output_split, training_data)):
            for reader, dataset in eval_tasks:
                if reader != 'piqa':
                    os.system(
                        """python evaluate_DeBERTa.py --lm {} --dataset_file {} --out_dir {} --device 5 --reader {}""".format(
                            models, dataset,
                            "./eval_results/{}_{}_deberta-v3-large".format(output_split, training_data),
                            reader))
                else:
                    os.system(
                        """python evaluate_DeBERTa_60MSL.py --lm {} --dataset_file {} --out_dir {} --device 5 --reader {}""".format(
                            models, dataset,
                            "./eval_results/{}_{}_deberta-v3-large".format(output_split, training_data),
                            reader))
        progress_bar.update(1)
