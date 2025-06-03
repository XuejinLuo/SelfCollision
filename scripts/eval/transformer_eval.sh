clear
python -u run.py \
  --eval \
  --model transformer \
  --num_frames 5 \
  --d_model 128 \
  --n_heads 4 \
  --d_ff 512 \
  --num_layers 3 \
  --test_datafile_path "data/test_distance_data.txt" \
  --checkpoint "checkpoints/transformer/best_transformer_model_5steps_3layer_11.7.pt" \
  --batch_size 100 \
