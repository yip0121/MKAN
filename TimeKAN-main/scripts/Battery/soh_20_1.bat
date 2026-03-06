@echo off
set CUDA_VISIBLE_DEVICES=0
set model_name=TimeKAN

python -u run.py ^
  --task_name long_term_forecast ^
  --is_training 1 ^
  --root_path ./dataset/battery/ ^
  --data_path battery_36Ah_70W_65W_1551.xlsx ^
  --model_id battery_soh_20_1 ^
  --model %model_name% ^
  --data battery_soh ^
  --features S ^
  --target soh ^
  --seq_len 20 ^
  --label_len 0 ^
  --pred_len 1 ^
  --e_layers 2 ^
  --enc_in 1 ^
  --dec_in 1 ^
  --c_out 1 ^
  --des Exp ^
  --itr 1 ^
  --d_model 16 ^
  --d_ff 32 ^
  --batch_size 32 ^
  --learning_rate 0.001 ^
  --train_epochs 20 ^
  --num_workers 0
