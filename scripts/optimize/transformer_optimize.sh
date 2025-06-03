clear
python -u run.py \
  --optimize \
  --model transformer \
  --checkpoint "checkpoints/transformer/best_transformer_model_5frames_128_4_512_4layers_8.3.pt" \
  --num_frames 5 \
  --d_model 128 \
  --n_heads 4 \
  --d_ff 512 \
  --num_layers 4 \
  --test_datafile_path "data/test_distance_data_small.txt" \
  --distance_weight 1 \
  --position_weight 100 \
  --maxiter 1 \
  --robot_operation 0 \
  