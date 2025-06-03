python -u run.py \
  --eval \
  --model lstm \
  --num_frames 3 \
  --hidden_size 150 \
  --num_layers 1 \
  --test_datafile_path "data/test_distance_data_small.txt" \
  --checkpoint "checkpoints/best_lstm_model.pt" \
  --batch_size 100 \
