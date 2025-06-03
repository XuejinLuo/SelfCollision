import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class data_loader(object):
    def __init__(self, args):
        self.is_mamba = args.model.lower() in ["mamba", "smamba"]
        self.is_bp = args.model.lower() in ["bp"]
        self.is_online = args.model.lower() in ["SSM_cross_attn"]
        self.train_data_directory = args.train_data_path
        self.test_data_directory = args.test_datafile_path
        self.files_num = args.files_num
        self.num_frames = args.num_frames
        self.batch_size = args.batch_size
        if args.train:
            self.train_data_loader = self.load_all_data(self.train_data_directory, args.time_step)
        self.test_data_loader = self.load_test_data(args.test_datafile_path, args.time_step)

    def read_data_from_file(self, file_path):
        data_distance_raw = pd.read_csv(file_path, header=None, delimiter=r'\s+')
        return data_distance_raw.values
    
    def process_data(self, file_path, num_rows_per_group):
        data_all_raw = self.read_data_from_file(file_path)
        num_groups = data_all_raw.shape[0] // num_rows_per_group
        groups = []

        for i in range(num_groups):
            start_index = i * num_rows_per_group + (num_rows_per_group - self.num_frames)
            end_index = (i + 1) * num_rows_per_group
            group_data = data_all_raw[start_index:end_index, :]
            groups.append(group_data)
        processed_inputs = []
        processed_outputs = []
        for group in groups:
            
            inputs = group[:, 1:]
            inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(device)
            inputs_tensor = inputs_tensor.view(-1, self.num_frames, 14)  # convert to (N, num_frames, 14)
            if self.is_bp:
                inputs_tensor = inputs_tensor.squeeze(1) # bp output needs to be (N, 1)
            
            # mamba needs the output to be in the shape of (N, 1, 14)
            if self.is_mamba:
                outputs = group[:, 0]
                outputs_tensor = torch.tensor(outputs, dtype=torch.float32).to(device)
                outputs_tensor = outputs_tensor.unsqueeze(1).unsqueeze(2)
                outputs_tensor = outputs_tensor.permute(1, 0, 2)
                outputs_tensor = outputs_tensor.repeat(1, 1, 14)
            # if self.is_online:
            #     outputs = group[:, 0]
            #     outputs_tensor = torch.tensor(outputs, dtype=torch.float32).to(device)
            #     outputs_tensor = outputs_tensor.unsqueeze(0)
            else:
                outputs = group[-1, 0]
                outputs_tensor = torch.tensor([outputs], dtype=torch.float32).to(device)
                outputs_tensor = outputs_tensor.unsqueeze(1)

            processed_inputs.append(inputs_tensor)
            processed_outputs.append(outputs_tensor)

        file_inputs_tensor = torch.cat(processed_inputs, dim=0)
        file_outputs_tensor = torch.cat(processed_outputs, dim=0)
        dataset = TensorDataset(file_inputs_tensor, file_outputs_tensor)

        return dataset
            
    def get_file_paths(self, directory, prefix="distance_data_file_", suffix=".txt"):
        file_paths = []
        for filename in os.listdir(directory):
            if filename.startswith(prefix) and filename.endswith(suffix):
                file_paths.append(os.path.join(directory, filename))
        file_paths.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))  # Sort by the number in the filename
        print("Loading file:")
        for file_path in file_paths[:self.files_num]:
            print(os.path.basename(file_path))
        return file_paths[:self.files_num]  # only load the first `files_num` files

    def load_all_data(self, directory, num_rows_per_group):
        file_paths = self.get_file_paths(directory)
        all_datasets = []
        for file_path in tqdm(file_paths, desc="Processing files", unit="file", ncols=100):
            dataset = self.process_data(file_path, num_rows_per_group)
            all_datasets.append(dataset)

        all_data_loader = DataLoader(torch.utils.data.ConcatDataset(all_datasets), batch_size=self.batch_size, shuffle=True)

        return all_data_loader
    
    def load_test_data(self, file_path, num_rows_per_group):
        dataset = self.process_data(file_path, num_rows_per_group=num_rows_per_group)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
    
    def print_data_shape(self):
        for batch in self.test_data_loader:
            inputs, outputs = batch
            print("Inputs:", inputs[0])
            print("Outputs:", outputs[0])
            print("Inputs shape:", inputs.shape)
            print("Outputs shape:", outputs.shape)
            break
