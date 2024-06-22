cd /nlpdata1/home/tfang/projects/complex_reasoning_atomic/experiments/MCQA/Training

${CONDA} run -n myenv pip install overrides
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256'

CUDA_VISIBLE_DEVICES=0 ${CONDA} run -n myenv python run_pretrain.py \
--model_type deberta-mlm \
--model_name_or_path "microsoft/deberta-v3-large" \
--task_name atomic \
--output_dir output/Output_ATOMIC-pseudo-wWC/car_2i/deberta-v3-large_car_2i_name_100k_seed_42 \
--train_file ../../../data/mcqa/atomic/train_atmc_2i_100k_name.jsonl \
--dev_file ../../../data/mcqa/atomic/dev_atmc_SyntheticQA_10k.jsonl \
--max_seq_length 128 \
--do_train --do_eval \
--per_gpu_train_batch_size 2 \
--gradient_accumulation_steps 16 \
--learning_rate 7e-6 \
--num_train_epochs 1 \
--warmup_proportion 0.05 \
--evaluate_during_training \
--per_gpu_eval_batch_size 32 \
--save_steps 200 \
--margin 1.0 \
--eval_output_dir ./eval_results --do_ext_eval --seed 42 