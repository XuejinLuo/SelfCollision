python -u run.py \
  --eval \
  --model bp \
  --num_frames 1 \
  --hidden_size 150 \
  --num_layers 5 \
  --nodes_per_layer "256,256,256,256,256" \
  --test_datafile_path "data/test_distance_data_small.txt" \
  --checkpoint "checkpoints/best_bp_model.pt" \
  --batch_size 100 \
