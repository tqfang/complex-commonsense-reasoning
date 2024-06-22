# checkpoints

## CAR+COM2: output/Output_ATOMIC-pseudo-wWC/car_2i/deberta-v3-large_car_2i_name_100k_seed_101_5e-6
## HyKAS+COM2: output/Output_ATOMIC-pseudo-wWC/deberta-v3-large_2i_atm_half_sample_name_5e-6


# COM2 eval
out_path=./eval_results/;\
lm_path=output/Output_ATOMIC-pseudo-wWC/car_2i/deberta-v3-large_car_2i_name_100k_seed_101_5e-6;\
python evaluate_DeBERTa.py --lm $lm_path \
 --dataset_file ../../../data/mcqa/eval/cqa_atomic_v1.0_name_mcqa.jsonl \
 --out_dir $out_path \
 --device 0 --reader cqa --overwrite_output_dir --max_length 150

out_path=./eval_results/;\
lm_path=output/Output_ATOMIC-pseudo-wWC/car_2i/deberta-v3-large_car_2i_name_100k_seed_101_5e-6;\
python evaluate_DeBERTa.py --lm $lm_path \
 --dataset_file ../../../data/mcqa/eval/commonsenseqa_dev.jsonl \
 --out_dir $out_path \
 --device 0 --reader commonsenseqa --overwrite_output_dir

# other evaluation files:

# socialiqa_dev.jsonl, winogrande_dev.jsonl, anli_dev.jsonl, piqa_dev.jsonl, 
# which can be downloaded at https://github.com/HKUST-KnowComp/CAR/blob/main/tasks/


