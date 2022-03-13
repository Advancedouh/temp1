import torch
from torch import nn
from torch.nn import functional as F

class MLPModel(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size, dropout):
        super().__init__()
        if not hasattr(hidden_size, '__len__'):
            hidden_size = [hidden_size]

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

        hidden_size_all = [embed_size] + hidden_size + [output_size]
        self.model = nn.ModuleList([
            nn.Linear(hidden_size_all[i], hidden_size_all[i+1]) for i in range(len(hidden_size_all) - 1)
        ])
    
    def pool(self, input_embed, input_mask):
        #input_embed:[batchsize, max_len, embed_size]
        #input_mask:[batchsize, max_len]
        pooled_input = (input_embed * input_mask.unsqueeze(-1)).sum(1) / input_mask.sum(-1).unsqueeze(-1)
        return pooled_input

    def forward(self, input_embed, input_mask=None):
        #如果没输入mask，就代表已经池化过
        #如果输入mask，则先进行池化
        if input_mask is not None:
            input_embed = self.pool(input_embed, input_mask)
        for fc_layer_except_last in self.model[:-1]:
            input_embed = fc_layer_except_last(input_embed)
            input_embed = self.act(self.dropout(input_embed))
        
        return self.model[-1](input_embed)

        

