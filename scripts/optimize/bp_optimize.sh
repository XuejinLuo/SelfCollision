clear
python -u run.py \
  --optimize \
  --model bp \
  --checkpoint "checkpoints/bp/best_bp_model_jsdf.pt" \
  --num_frames 1 \
  --hidden_size 150 \
  --num_layers 5 \
  --nodes_per_layer "256,256,256,256,256" \
  --test_datafile_path "data/test_distance_data_small.txt" \
  --distance_weight 1 \
  --position_weight 10 \
  --maxiter 1 \
  --robot_operation 0 \
  