python -u run.py \
  --train \
  --model bp \
  --num_frames 1 \
  --hidden_size 150 \
  --num_layers 5 \
  --nodes_per_layer "256,256,256,256,256" \
  --train_data_path "data/distance_data" \
  --files_num 1 \
  --test_datafile_path "data/test_distance_data_small.txt" \
  --checkpoints_path "checkpoints" \
  --batch_size 100 \
  --num_epochs 10 \
  --test_frequency 3 \
  --learning_rate 0.0001 \
