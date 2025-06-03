import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange

class RMSNorm(nn.Module):
    def __init__(self,
                d_model: int,
                eps: float = 1e-5,
                device: str ='cuda'):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device))


    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
    

class S6(nn.Module):
    def __init__(self, batch_size, seq_len, d_model, state_size, device):
        super(S6, self).__init__()

        self.fc1 = nn.Linear(d_model, d_model, device=device)
        self.fc2 = nn.Linear(d_model, state_size, device=device)
        self.fc3 = nn.Linear(d_model, state_size, device=device)

        self.seq_len = seq_len
        self.d_model = d_model
        self.state_size = state_size


        self.A = nn.Parameter(F.normalize(torch.ones(d_model, state_size, device=device), p=2, dim=-1))
        nn.init.xavier_uniform_(self.A)

        self.B = torch.zeros(batch_size, self.seq_len, self.state_size, device=device)
        self.C = torch.zeros(batch_size, self.seq_len, self.state_size, device=device)

        self.delta = torch.zeros(batch_size, self.seq_len, self.d_model, device=device)
        self.dA = torch.zeros(batch_size, self.seq_len, self.d_model, self.state_size, device=device)
        self.dB = torch.zeros(batch_size, self.seq_len, self.d_model, self.state_size, device=device)

        # h [batch_size, seq_len, d_model, state_size]
        self.h = torch.zeros(batch_size, self.seq_len, self.d_model, self.state_size, device=device)
        self.y = torch.zeros(batch_size, self.seq_len, self.d_model, device=device)


    def discretization(self):

        self.dB = torch.einsum("bld,bln->bldn", self.delta, self.B)

        self.dA = torch.exp(torch.einsum("bld,dn->bldn", self.delta, self.A))


        return self.dA, self.dB

    def forward(self, x):
        # Algorithm 2 MAMBA paper
        self.B = self.fc2(x)
        self.C = self.fc3(x)
        self.delta = F.softplus(self.fc1(x))

        self.discretization()
 
        # h [batch_size, seq_len, d_model, state_size]
        h = torch.zeros(x.size(0), self.seq_len, self.d_model, self.state_size, device=x.device)
        y = torch.zeros_like(x)

        h =  torch.einsum('bldn,bldn->bldn', self.dA, h) + rearrange(x, "b l d -> b l d 1") * self.dB

        # y [batch_size, seq_len, d_model]
        y = torch.einsum('bln,bldn->bld', self.C, h)

        return y

class MambaBlock(nn.Module):
    def __init__(self, batch_size, seq_len, d_model, d_ff, state_size, device):
        super(MambaBlock, self).__init__()

        self.inp_proj = nn.Linear(d_model, d_ff, device=device)
        self.out_proj = nn.Linear(d_ff, d_model, device=device)

        # For residual skip connection
        self.D = nn.Linear(d_model, d_ff, device=device)

        # Set _no_weight_decay attribute on bias
        self.out_proj.bias._no_weight_decay = True

        # Initialize bias to a small constant value
        nn.init.constant_(self.out_proj.bias, 1.0)

        self.S6 = S6(batch_size, seq_len, d_ff, state_size, device)

        # Add 1D convolution with kernel size 3
        self.conv = nn.Conv1d(seq_len, seq_len, kernel_size=3, padding=1, device=device)

        # Add linear layer for conv output
        self.conv_linear = nn.Linear(d_ff, d_ff, device=device)

        # rmsnorm
        self.norm = RMSNorm(d_model, device=device)

    def forward(self, x):
        """
    x_proj.shape = torch.Size([batch_size, seq_len, d_ff])
    x_conv.shape = torch.Size([batch_size, seq_len, d_ff])
    x_conv_act.shape = torch.Size([batch_size, seq_len, d_ff])
    """
        # Refer to Figure 3 in the MAMBA paper

        x = self.norm(x)

        x_proj = self.inp_proj(x)

        # Add 1D convolution with kernel size 3
        x_conv = self.conv(x_proj)

        x_conv_act = F.silu(x_conv)

        # Add linear layer for conv output
        x_conv_out = self.conv_linear(x_conv_act)

        x_ssm = self.S6(x_conv_out)
        x_act = F.silu(x_ssm)  # Swish activation can be implemented as x * sigmoid(x)

        # residual skip connection with nonlinearity introduced by multiplication
        x_residual = F.silu(self.D(x))

        x_combined = x_act * x_residual

        x_out = self.out_proj(x_combined)

        return x_out
    

class Mamba(nn.Module):
    def __init__(self, batch_size, seq_len, input_dim, d_model, d_ff, state_size, device):
        super(Mamba, self).__init__()
        self.input_layer = nn.Linear(input_dim, d_model, device=device)
        self.mamba_block1 = MambaBlock(batch_size, seq_len, d_model, d_ff, state_size, device)
        self.mamba_block2 = MambaBlock(batch_size, seq_len, d_model, d_ff, state_size, device)
        self.mamba_block3 = MambaBlock(batch_size, seq_len, d_model, d_ff, state_size, device)
        self.final_layer = nn.Linear(d_model, 1, device=device)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.mamba_block1(x)
        x = self.mamba_block2(x)
        x = self.mamba_block3(x)
        x = x[:, -1, :]  # Take the last time step for prediction
        prediction = self.final_layer(x)
        return prediction
    
def create_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Mamba(
        batch_size=args.batch_size,
        seq_len=args.num_frames,
        input_dim=args.input_num,
        d_model=args.d_model,
        d_ff=args.d_ff,
        state_size=args.d_state,
        device=device
    )
    return model