#!/usr/bin/env bash
set -euo pipefail

ROOT_PATH="./dataset/battery/"
DATA_PATH="battery_36Ah_70W_65W_1551.xlsx"
TARGET_COL_IDX=3
COMMON_ARGS=(
  --task_name long_term_forecast
  --is_training 1
  --data battery_soh
  --root_path "$ROOT_PATH"
  --data_path "$DATA_PATH"
  --target_col_idx "$TARGET_COL_IDX"
  --seq_len 20
  --pred_len 1
  --label_len 0
  --train_ratio 0.7
  --val_ratio 0.1
  --itr 1
  --train_epochs 20
  --batch_size 32
  --learning_rate 0.001
  --quantiles 0.95,0.5,0.05
  --enable_bayes_opt
  --bayes_trials 10
  --bayes_train_epochs 6
  --bayes_refit
)

python run.py "${COMMON_ARGS[@]}" --model CNNLSTM --model_id battery_cnnlstm
python run.py "${COMMON_ARGS[@]}" --model iTransformer --model_id battery_itransformer
python run.py "${COMMON_ARGS[@]}" --model TimeMixer --model_id battery_timemixer
