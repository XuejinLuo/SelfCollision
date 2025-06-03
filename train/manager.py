import os
import torch
import torch.optim as optim
from models.model_basic import ModelBasic
from data_loader import data_loader
from train.train_process import ModelTrainer
from train.eval_process import ModelEval

def manager(args):
    model_basic = ModelBasic(args)
    model = model_basic.model
    # print(model)

    if args.eval or args.continue_train:
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
    if args.train:
        trainer = ModelTrainer(args, model, my_data_loader)
        trainer.train()

    if args.eval:
        evaluator = ModelEval(args, model, my_data_loader)
        evaluator.eval()