CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes=1 --main_process_port=12547 run_leco.py \
-- test_set datasets/gsm8k.jsonl\
-- prompts prompts/gsm8k/complex_step.txt\
-- output_dir Your_output_dir\
-- ckpt_dir Your_model_path\
