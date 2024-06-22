cd /nlpdata1/home/tfang/projects/complex_reasoning_atomic/experiments/MCQA/Training

${CONDA} run -n myenv pip install overrides

CUDA_VISIBLE_DEVICES=0 ${CONDA} run -n myenv python run_pretrain.py \
--model_type deberta-mlm \
--model_name_or_path "microsoft/deberta-v3-large" \
--task_name atomic \
--output_dir output/Output_ATOMIC-pseudo-wWC/deberta-v3-large_2i_atm_half_sample_name_5e-6 \
--train_file ../../../data/mcqa/atomic/train_atm_n_2i_half_sample_name.jsonl \
--dev_file ../../../data/mcqa/atomic/dev_random_10k.jsonl \
--max_seq_length 128 \
--do_train --do_eval \
--per_gpu_train_batch_size 2 \
--gradient_accumulation_steps 16 \
--learning_rate 5e-6 \
--num_train_epochs 1 \
--warmup_proportion 0.05 \
--evaluate_during_training \
--per_gpu_eval_batch_size 16 \
--save_steps 500 \
--margin 1.0 \
--eval_output_dir ./eval_results --do_ext_eval