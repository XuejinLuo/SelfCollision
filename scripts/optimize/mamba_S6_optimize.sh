clear
python -u run.py \
  --optimize \
  --model mamba_S6 \
  --checkpoint "checkpoints/mamba_S6/best_mamba_S6_model_5_64_128_256_7.7.pt" \
  --num_frames 5 \
  --d_model 64 \
  --d_ff 128 \
  --d_state 256 \
  --test_datafile_path "data/test_distance_data_small.txt" \
  --distance_weight 1 \
  --position_weight 10 \
  --maxiter 1 \
  --robot_operation 0 \
  