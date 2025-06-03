import torch
import os
from tqdm import tqdm
from datetime import datetime
import torch.nn as nn
from utils.utils import bcolors

class ModelTrainer(object):
    def __init__(self, args, model, my_data_loader):
        self.args = args
        self.model = model
        self.modelname = args.model
        self.train_data_loader = my_data_loader.train_data_loader
        self.test_data_loader = my_data_loader.test_data_loader
        self.num_epochs = args.num_epochs
        self.min_test_loss = float('inf')
        self.test_frequency = args.test_frequency
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"output/{args.model}/{current_time}_{args.files_num}_{args.num_frames}frames_{args.continue_train}"
        os.makedirs(self.output_dir, exist_ok=True)
        self.checkpoints_path = args.checkpoints_path

    def train(self):
        # Training loop
        print("begin training")
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0
            for x_batch, y_batch in tqdm(self.train_data_loader, desc=f'Epoch {epoch+1}/{self.num_epochs}', unit='batch'):
                x_batch, y_batch = x_batch.to("cuda"), y_batch.to("cuda")
                self.optimizer.zero_grad()
                y_pred = self.model(x_batch)
                loss = self.criterion(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {epoch_loss/len(self.train_data_loader)}")

            if (epoch + 1) % self.test_frequency == 0:
                self.model.eval()
                with torch.no_grad():
                    total_test_loss = 0
                    total_errors = []
                    with open(f"{self.output_dir}/test_predictions_epoch_{epoch+1}.txt", "w") as f:
                        for x_test_batch, y_test_batch in self.test_data_loader:
                            x_test_batch = x_test_batch.to("cuda")
                            y_test_batch = y_test_batch.to("cuda")
                            test_outputs = self.model.forward(x_test_batch)
                            test_loss = self.criterion(test_outputs, y_test_batch)
                            total_test_loss += test_loss.item()
                            errors = (test_outputs - y_test_batch).abs()
                            total_errors.append(errors.mean().item())

                            for i in range(test_outputs.size(0)):  # 遍历每个测试数据
                                pred_value = test_outputs[i].cpu().numpy()
                                true_value = y_test_batch[i].cpu().numpy()
                                f.write(f"\nPred: {pred_value}, True: {true_value}\n")

                    avg_test_loss = total_test_loss / len(self.test_data_loader)
                    avg_total_error = sum(total_errors) / len(total_errors)
                    
                    print(f"Test Loss after Epoch {epoch+1}: {avg_test_loss:.4f}")
                    print(f"{bcolors.WARNING}Total Error after Epoch {epoch+1}: {avg_total_error:.4f}{bcolors.ENDC}")
                    with open(f"{self.output_dir}/avg_total_error.txt", "a") as f:
                        f.write(f"Epoch {epoch+1}: avg: {avg_total_error:.4f} total: {sum(total_errors)}\n")

                    if avg_total_error < self.min_test_loss:
                        self.min_test_loss = avg_total_error
                        torch.save(self.model.state_dict(), f"{self.checkpoints_path}/best_{self.modelname}_model.pt")
                        print(f"{bcolors.OKGREEN}Model saved at epoch {epoch+1} with test loss: {self.min_test_loss:.4f}{bcolors.ENDC}")

                self.model.train()


    def train_online(self):
        # Training loop
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0
    
            for x_batch, y_batch in tqdm(self.train_data_loader, desc=f'Epoch {epoch+1}/{self.num_epochs}', unit='batch'):
                x_batch, y_batch = x_batch.to("cuda"), y_batch.to("cuda")
                seq_len = x_batch.size(1)
                batch_loss = 0
                for t in range(seq_len):
                    if t == 0:
                        self.model.prev_features = None
                    current_frame = x_batch[:, t:t + 1, :]
                    current_y = y_batch[:, t:t + 1]
                    y_pred = self.model(current_frame)
                    if t > 4:
                        self.optimizer.zero_grad()
                        loss = self.criterion(y_pred, current_y)
                        loss.backward()
                        if self.model.prev_features is not None:
                            self.model.prev_features = self.model.prev_features.detach()
                        self.optimizer.step()
                        batch_loss = loss.item()
                epoch_loss += batch_loss

            print(f'Epoch {epoch + 1}/{self.num_epochs}, Loss: {epoch_loss / len(self.train_data_loader)}')
            if (epoch + 1) % self.test_frequency == 0:
                self.model.eval()
                with torch.no_grad():
                    total_test_loss = 0
                    total_errors = []
                    with open(f"{self.output_dir}/test_predictions_epoch_{epoch + 1}.txt", "w") as f:
                        for x_test_batch, y_test_batch in self.test_data_loader:
                            x_test_batch = x_test_batch.to("cuda")
                            y_test_batch = y_test_batch.to("cuda")
                            seq_len = x_test_batch.size(1)

                            for t in range(seq_len):
                                if t == 0:
                                    self.model.prev_features = None
                                current_frame = x_test_batch[:, t:t + 1, :]
                                current_y = y_test_batch[:, t:t + 1]
                                test_output = self.model(current_frame)
                                if test_output is not None:
                                    test_loss = self.criterion(test_output, current_y)
                                if t == seq_len - 1:
                                    for i in range(test_output.size(0)):
                                        pred_value = test_output[i].cpu().numpy()
                                        true_value = current_y[i].cpu().numpy()
                                        f.write(f"Pred: {pred_value}, True: {true_value}\n")
                            total_test_loss += test_loss.item()
                            errors = (test_output - current_y).abs()
                            total_errors.append(errors.mean().item())

                    avg_test_loss = total_test_loss / len(self.test_data_loader)
                    avg_total_error = sum(total_errors) / len(total_errors)

                    print(f"Test Loss after Epoch {epoch+1}: {avg_test_loss:.4f}")
                    print(f"{bcolors.WARNING}Total Error after Epoch {epoch+1}: {avg_total_error:.4f}{bcolors.ENDC}")
                    with open(f"{self.output_dir}/avg_total_error.txt", "a") as f:
                        f.write(f"Epoch {epoch + 1}: avg: {avg_test_loss:.6f}\n")

                    if avg_total_error < self.min_test_loss:
                        self.min_test_loss = avg_total_error
                        torch.save(self.model.state_dict(), f"{self.checkpoints_path}/best_{self.modelname}_model.pt")
                        print(f"{bcolors.OKGREEN}Model saved at epoch {epoch+1} with test loss: {self.min_test_loss:.4f}{bcolors.ENDC}")

                self.model.train()