import numpy as np
import torch
import time
import os
import torch.nn as nn
from utils.utils import bcolors
from datetime import datetime

class ModelEval(object):
    def __init__(self, args, model, my_data_loader):
        self.args = args
        self.model = model
        self.modelname = args.model
        self.test_data_loader = my_data_loader.test_data_loader
        self.criterion = nn.MSELoss()

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"output/{args.model}/{current_time}_{args.files_num}_{args.num_frames}frames"
        os.makedirs(self.output_dir, exist_ok=True)

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            total_test_loss = 0
            total_errors = []
            with open(f"{self.output_dir}/eval_file.txt", "w") as f:
                for x_test_batch, y_test_batch in self.test_data_loader:
                    x_test_batch = x_test_batch.to("cuda")
                    y_test_batch = y_test_batch.to("cuda")
                    test_outputs = self.model.forward(x_test_batch)
                    test_loss = self.criterion(test_outputs, y_test_batch)
                    total_test_loss += test_loss.item()
                    errors = (test_outputs - y_test_batch).abs()
                    total_errors.append(errors.mean().item())

                    for i in range(test_outputs.size(0)):
                        pred_value = test_outputs[i].cpu().numpy()
                        true_value = y_test_batch[i].cpu().numpy()
                        f.write(f"Pred: {pred_value}, True: {true_value}\n")

            avg_test_loss = total_test_loss / len(self.test_data_loader)
            avg_total_error = sum(total_errors) / len(total_errors)
            
            print(f"Test Loss: {avg_test_loss:.4f}")
            print(f"Total Error: {avg_total_error:.4f}")
            with open(f"{self.output_dir}/avg_total_error.txt", "a") as f:
                f.write(f"avg: {avg_total_error:.4f} total: {sum(total_errors)}\n")

            start_time = time.time()
            test_outputs = self.model.forward(x_test_batch[0].unsqueeze(0))
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Forward pass time: {elapsed_time:.6f} seconds")