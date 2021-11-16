'''
Descripttion: 
version: 
Author: Andrew
Date: 2021-07-26 22:02:18
LastEditors: Andrew
LastEditTime: 2021-10-08 21:47:31
'''
import torch
import torch.nn as nn
import logging
from .encode_layer import PromptEncodeLayer
logger = logging.getLogger(__name__)

class PromptEncoder(torch.nn.Module):
    def __init__(self, spell_length, hidden_size, ):
        super().__init__()
        self.spell_length = spell_length
        self.hidden_size = hidden_size
        self.n_layers = 2
        # embedding
        self.embedding = torch.nn.Embedding(self.spell_length, self.hidden_size)
        # LSTM
        # self.encode_layers = torch.nn.LSTM(input_size=self.hidden_size,
        #                                hidden_size=self.hidden_size // 2,
        #                                num_layers=self.n_layers,
        #                                dropout=0.5,
        #                                bidirectional=True,
        #                                batch_first=True)
        
        # Transformer
        self.encode_layers = nn.ModuleList([PromptEncodeLayer(d_model=hidden_size) 
                                                    for _ in range(self.n_layers)])

        self.mlp_head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.hidden_size))
        logger.info("init prompt encoder...")

    def forward(self, seq_indices):
        input_embeds = self.embedding(seq_indices).unsqueeze(0)
        # LSTM
        # input_embeds = self.encode_layers(input_embeds)[0]

        # Transformer
        prompt_att_mask = torch.ones([input_embeds.size(0), input_embeds.size(1), input_embeds.size(1)]).bool().to(input_embeds.device)

        for layer in self.encode_layers:
            input_embeds = layer(input_embeds, prompt_att_mask)[0]
        
        output_embeds = self.mlp_head(input_embeds).squeeze()
        return output_embeds
