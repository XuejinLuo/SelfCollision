import os
import torch
from models import SSM_cross_attn, bp, lstm, mamba_S6, transformer, mamba, mamba_transformer, attn_mambaS6, mamba_transformer_posemb

class ModelBasic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'bp': bp,
            'lstm': lstm,
            'transformer': transformer,
            'mamba': mamba,
            'mamba_S6': mamba_S6,
            'mamba_transformer': mamba_transformer,
            'attn_mambaS6': attn_mambaS6,
            'SSM_cross_attn': SSM_cross_attn, 
            'mamba_transformer_posemb': mamba_transformer_posemb
        }
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        model_class = self.model_dict[self.args.model]
        return model_class.create_model(self.args)

