clear
python -u run.py \
  --optimize \
  --model lstm \
  --checkpoint "checkpoints/lstm/best_lstm_model_13.38.pt" \
  --num_frames 3 \
  --hidden_size 150 \
  --num_layers 1 \
  --test_datafile_path "data/test_distance_data_small.txt" \
  --distance_weight 1 \
  --position_weight 10 \
  --maxiter 1 \
  --robot_operation 0 \
  