python -u run.py \
  --train \
  --model transformer \
  --num_frames 5 \
  --d_model 128 \
  --n_heads 4 \
  --d_ff 256 \
  --num_layers 4 \
  --train_data_path "data/distance_data" \
  --files_num 22 \
  --test_datafile_path "data/test_distance_data.txt" \
  --checkpoints_path "checkpoints" \
  --batch_size 100 \
  --num_epochs 37 \
  --test_frequency 3 \
  --learning_rate 0.0001 \

