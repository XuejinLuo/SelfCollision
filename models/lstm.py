# lstm.py
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, batch_first=True):
        super(LSTMModel, self).__init__()
        # 定义LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=batch_first)
        # 定义线性层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 通过LSTM层，获取LSTM的输出
        lstm_out, (hn, cn) = self.lstm(x)
        # 获取最后一个时间步的输出
        last_out = lstm_out[:, -1, :]  # 获取最后一个时间步的输出
        # 通过全连接层进行预测
        predictions = self.fc(last_out)
        return predictions

def create_model(args):
    model = LSTMModel(
        input_size=args.input_num,
        hidden_size=args.hidden_size,
        output_size=args.output_num,
        num_layers=args.num_layers
    )
    return model