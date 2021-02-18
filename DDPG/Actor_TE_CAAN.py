import copy
from typing import Optional, Any
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Softmax
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
import pandas as pd
import gc
import numpy as np
from torch.nn.modules.activation import MultiheadAttention  # 빨간줄 뜨지만, 실행됨.
import matplotlib.pyplot as plt
import warnings
import torch.optim as optim

# Turn off warnings
warnings.filterwarnings(action='ignore')

torch.set_default_tensor_type('torch.DoubleTensor')  # 'torch.DoubleTensor'


class Transformer(Module):
    """
        *** 클래스들에 대한 정의 ***
        1. Transformer : 아래 네 가지 클래스(TE, TEL, TD, TDL)을 이용한 전체 프로세스(NN) 실행 및 결과 출력
        2. TransformerEncoder : Transformer와 TE_Layer의 어댑터 역할
        3. TE_Layer : TE의 network을 구성하는 클래스
        4. CAAN : Transformer와 CAAN_Layer의 어댑터 역할
        5. CAAN_Layer : CAAN의 network을 구성하는 클래스

        Args:
        d_model: the number of expected features in the encoder/CAAN inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_CAAN_layers: the number of sub-CAAN-layers in the CAAN (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of encoder/CAAN intermediate layer, relu or gelu (default=relu).
        custom_encoder: custom encoder (default=None).
        custom_CAAN: custom CAAN (default=None).
    """

    # Initiate TE-CAAN network
    def __init__(self, d_model: int = 24, nhead: int = 4, nhead2: int = 1,
                 num_encoder_layers: int = 1,
                 num_CAAN_layers: int = 1, dim_feedforward: int = 2, dropout: float = 0.1,
                 column_num: int = 22, threshold: int = 0, activation: str = "relu", lr = 0.0001,
                 custom_encoder: Optional[Any] = None, custom_CAAN: Optional[Any] = None) -> None:
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.threshold = threshold
        self.column_num = column_num
        self.embeddingLayer = Linear(self.column_num, self.d_model)
        self.period = 12
        self.optimizer = optim.Adam(self.parameters(), lr =lr, weight_decay = 0.01)
        self.lr = lr
        # Rewards slot for Sharpe Ratio calculation
        self.rewards = []

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(d_model, nhead)
        encoder_norm = LayerNorm(self.d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # Cross Asset Attention Network
        CAAN_layer = CAAN_Layer(d_model, nhead2, dim_feedforward, dropout, activation,
                                column_num=self.column_num)
        self.CAAN = CAAN(CAAN_layer, num_CAAN_layers)

        #Reset Parameters
        self._reset_parameters()

        # Forward of Transformer
    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None, key_table=pd.Series()) -> Tensor:

        src_input = src.permute([1, 0, 2])  # [Num_firm, Time, Factors] --> [Time, Num_firm, Factors]

        src_input = torch.as_tensor(src_input, dtype=torch.double, device='cpu')

        src_input = self.embeddingLayer(src_input) # [Time, Num_firm, Factors] -> [Time, Num_firm, embed_dim]



        memory = self.encoder(src_input, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        #clean memory
        del src
        del src_input
        gc.collect()
        torch.cuda.empty_cache()

        # Run through https://pytorch.org/docs/stable/_modules/torch/nn/modules/sparse.html#EmbeddingCAAN network
        memory = memory.permute([1, 0, 2])  # [Time, Num_Firm, Features] ==> [Num_firms, Time, Features]
        output = self.CAAN(memory, memory_mask=memory_mask, memory_key_padding_mask=memory_key_padding_mask,
                              threshold=self.threshold, key_table=key_table)

        #clean memory
        del memory
        gc.collect()
        torch.cuda.empty_cache()

        return output


    #Method used to reset parameters
    def _reset_parameters(self):
        'Initiate parameters in the transformer model'

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class TransformerEncoder(Module):
    'Transformer Encoder'
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)
        return output


class CAAN(Module):
    'CAAN'
    __constants__ = ['norm']

    def __init__(self, CAAN_layer, num_layers):
        super(CAAN, self).__init__()
        self.layers = _get_clones(CAAN_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, memory: Tensor, threshold: int,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                key_table=pd.Series()) -> Tensor:
        output = memory

        for mod in self.layers:
            output = mod(memory, threshold=threshold, key_table=key_table)
        return output


class TransformerEncoderLayer(Module):
    'Transformer Encoder Layer'

    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.activation = _get_activation_fn(activation)


    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        src2 = self.self_attn(src, src, src)[0] #, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]  # [period, company_num, column_num]

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class CAAN_Layer(Module):
    def __init__(self, d_model, nhead2, dim_feedforward=2, dropout=0.1, activation="relu", column_num = None):
        super(CAAN_Layer, self).__init__()

        self.d_model = d_model
        self.self_attn = MultiheadAttention(d_model, nhead2, dropout=dropout)
        self.period = 12
        self.column_num = column_num
        self.dim_feedforward = dim_feedforward
        self.num_hidden_node1 = self.d_model * self.period  # 1 year
        self.num_hidden_node2 = self.d_model * self.period // 2  # 6 months
        self.linear_w1 = Linear(self.d_model * self.period, self.num_hidden_node1)
        self.linear_w2 = Linear(self.d_model * self.period, self.num_hidden_node2)
        self.linear_w3 = Linear(self.num_hidden_node2, 1)
        self.tanh = torch.tanh
        self.relu = torch.relu
        self.dropout = Dropout(dropout)


    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(CAAN_Layer, self).__setstate__(state)


    def forward(self, memory: Tensor, threshold: int = 0, key_table=pd.Series(),
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        tgt = self.self_attn(memory, memory, memory)[0]
        # Self_attention Output Shape: [Num_Firm, Time, Features]   FYI:[Sequence, Batch, embedding dim]
        # tgt = self.norm1(tgt)
        tgt = tgt.reshape(tgt.shape[0], -1)  # Output Shape: [Num_Firm, Time * Features]

        #Producing Winner Score
        # Producing Winner Score
        tgt = self.linear_w1(tgt)
        tgt = self.relu(tgt)
        tgt = self.dropout(tgt)

        tgt = self.linear_w2(tgt)
        tgt = self.relu(tgt)
        tgt = self.dropout(tgt)

        winner_score = self.linear_w3(tgt)
        winner_score = self.tanh(winner_score)

        # sort winner score in descending order
        winner_score, indices = torch.sort(winner_score, descending=True, dim=0)
        _stocks = indices[:threshold].cpu().numpy().squeeze()  # Stocks to Long
        key_table_np = key_table.to_numpy()

        stocks = key_table_np[_stocks]  # Stocks to long in the key_table

        softmax = Softmax(dim=0)  # softmax

        _action = winner_score[:threshold]
        action = softmax(_action)

        # 지워야함. bc를 보기 위함.
        #print(action)


        return action, stocks, _stocks


def _get_clones(module, N):  # 여기서 module이 우리 전체 model의 현재 상태를 나타내는가? 그래야 한다…!
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))