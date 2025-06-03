# main.py
import argparse
import random
import torch
import numpy as np
from optimize.optimize_manager import optimize_manager
from train.manager import manager
from test.test_manager import test_manager

if __name__ == "__main__":
    fix_seed = 2025
    torch.manual_seed(fix_seed)
    random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='self-collision detection')

    # basic config
    parser.add_argument('--train', action='store_true', help='status for training')
    parser.add_argument('--eval', action='store_true', help='status for evaluation')
    parser.add_argument('--test', action='store_true', help='status for test')
    parser.add_argument('--optimize', action='store_true', help='optimize test if True')
    parser.add_argument('--model', type=str, default='bp',
                    help='model name, options: [bp, lstm, mamba, transformer]')
    
    # data loader
    parser.add_argument('--train_data_path', type=str, default='data/distance_data', help='train path of the data file')
    parser.add_argument('--test_datafile_path', type=str, default='data/test_distance_data.txt', help='test data file')
    parser.add_argument('--files_num', type=int, default=None, help='data_files num')
    parser.add_argument('--checkpoints_path', type=str, default='checkpoints', help='location of model checkpoints')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/lstm/best_lstm_model_13.38.pt', help='model checkpoint')
    parser.add_argument('--batch_size', type=int, default=100, help='data batch size')
    parser.add_argument('--print_data_shape', action='store_true', help='Print the DataLoader dataset shape if True')
    parser.add_argument('--time_step', type=int, default=10, help='input data time step')

    # model define
    parser.add_argument('--output_num', type=int, default=1, help='model output num')
    parser.add_argument('--input_num', type=int, default=14, help='model input num')
    parser.add_argument('--num_frames', type=int, default=10, help='model input frames num')
    parser.add_argument('--num_epochs', type=int, default=2000, help='training epochs')
    parser.add_argument('--test_frequency', type=int, default=3, help='training test frequency')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--continue_train', action='store_true', help='continue training if True')

    parser.add_argument('--d_model', type=int, default=128, help='dimension of the model for transformer or mamba')
    parser.add_argument('--n_heads', type=int, default=4, help='number of attention heads for transformer')
    parser.add_argument('--d_ff', type=int, default=512, help='dimension of the feedforward layer for transformer')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers for transformer or lstm or bp')
    parser.add_argument('--hidden_size', type=int, default=150, help='hidden size for lstm')
    parser.add_argument('--nodes_per_layer', type=str, default='150', help='nodes per layer for bp, comma-separated values')
    parser.add_argument('--d_state', type=int, default=16, help='state expansion factor for mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='local convolution width for mamba')
    parser.add_argument('--expand', type=int, default=2, help='block expansion factor for mamba')

    # optimizer
    parser.add_argument('--distance_weight', type=float, default=1, help='self-collision distance weight')
    parser.add_argument('--position_weight', type=float, default=1, help='robot position weight')
    parser.add_argument('--maxiter', type=int, default=1, help='optimize solver maxiter')
    parser.add_argument('--robot_operation', type=int, default=0, help='choose which robot operation. (0: dual arm, 1: robot1, 2: robot2)')

    args = parser.parse_args()

    if args.optimize:
        print("Running optimization test...")
        optimize_manager(args)
    
    elif args.train or args.eval:
        if args.train:
            print("Starting training...")
        else:
            print("Starting evaluation...")
        manager(args)
    
    elif args.test:
        print("Testing manager function...")
        test_manager(args)

    else:
        print("Please specify an action: --train, --test, or --optimize.")