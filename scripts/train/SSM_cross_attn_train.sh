python -u run.py \
  --train \
  --model SSM_cross_attn \
  --num_frames 5 \
  --d_model 256 \
  --d_ff 512 \
  --d_state 256 \
  --n_heads 4 \
  --num_layers 1 \
  --train_data_path "data/distance_data" \
  --files_num 35 \
  --test_datafile_path "data/test_distance_data.txt" \
  --checkpoints_path "checkpoints" \
  --batch_size 100 \
  --num_epochs 2000 \
  --test_frequency 3 \
  --learning_rate 0.0001 \

