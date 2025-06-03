clear
python -u run.py \
  --optimize \
  --model SSM_cross_attn \
  --checkpoint "checkpoints/SSM_cross_attn/checkpoints\SSM_cross_attn\best_cross_attn_model_S6_5_256_512_256_4_1_7.7.pt" \
  --num_frames 5 \
  --d_model 256 \
  --d_ff 512 \
  --d_state 256 \
  --n_heads 4 \
  --num_layers 1 \
  --test_datafile_path "data/test_distance_data_small.txt" \
  --distance_weight 1 \
  --position_weight 100 \
  --maxiter 1 \
  --robot_operation 0 \
  