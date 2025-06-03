# transformer_model.py
import torch
import torch.nn as nn
import torch.optim as optim

class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.q_linear(query).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attention_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads and pass through final linear layer
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_linear(attention_output)
        
        return output, attn_weights


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.attn = SelfAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, _ = self.attn(x, x, x, mask)
        x = self.layer_norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + self.dropout(ffn_output))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TimeSeriesTransformer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_layers, n_features, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.embedding = nn.Linear(n_features, d_model)  # 输入特征的维度映射到模型的维度
        self.encoder = TransformerEncoder(d_model, n_heads, d_ff, num_layers, dropout)
        self.fc_out = nn.Linear(d_model, 1)  # 输出一个连续值（第10帧的距离）

    def forward(self, x):
        x = self.embedding(x)  # 输入通过嵌入层
        enc_output = self.encoder(x)  # 经过 Transformer 编码器
        # 取最后一个时间步的输出，作为对第10帧的预测
        out = self.fc_out(enc_output[:, -1, :])  
        return out

def create_model(args):
    model = TimeSeriesTransformer(
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        n_features=args.input_num
    )
    return model