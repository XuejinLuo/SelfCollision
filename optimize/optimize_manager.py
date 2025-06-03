import os
import torch
import torch.optim as optim
from models.model_basic import ModelBasic
from data_loader import data_loader
from optimize.optimize_solver import OptimizeSolver

def optimize_manager(args):
    model_basic = ModelBasic(args)
    model = model_basic.model
    # print(model)
    model_file = args.checkpoint
    if os.path.exists(model_file):
        if os.path.isfile(model_file):
            model.load_state_dict(torch.load(model_file))
            print(f'Model loaded from {model_file}')
        else:
            print(f'No model file found at {model_file}')
    else:
        print(f'Checkpoint path {model_file} does not exist.')

    my_data_loader = data_loader(args)
    test_data_loader = my_data_loader.test_data_loader
    if args.print_data_shape:
        my_data_loader.print_data_shape()

    solver = OptimizeSolver(args, model, test_data_loader)
    solver.solve()

    print("Optimization completed for all samples.")

    