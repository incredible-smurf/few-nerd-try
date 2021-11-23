import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from transformers.models.bart.light_ner_prompt_bart import *




class LightEncoder(nn.Module):
    def __init__(self,encoder) -> None:
        super().__init__()
        self.encoder=encoder

    #返回2d mask
    def seq_len_to_mask(self,seq_len, max_len=None):
        r"""

        将一个表示sequence length的一维数组转换为二维的mask，不包含的位置为0。
        转变 1-d seq_len到2-d mask.

        .. code-block::
        
            >>> seq_len = torch.arange(2, 16)
            >>> mask = seq_len_to_mask(seq_len)
            >>> print(mask.size())
            torch.Size([14, 15])
            >>> seq_len = np.arange(2, 16)
            >>> mask = seq_len_to_mask(seq_len)
            >>> print(mask.shape)
            (14, 15)
            >>> seq_len = torch.arange(2, 16)
            >>> mask = seq_len_to_mask(seq_len, max_len=100)
            >>>print(mask.size())
            torch.Size([14, 100])

        :param np.ndarray,torch.LongTensor seq_len: shape将是(B,)
        :param int max_len: 将长度pad到这个长度。默认(None)使用的是seq_len中最长的长度。但在nn.DataParallel的场景下可能不同卡的seq_len会有
            区别，所以需要传入一个max_len使得mask的长度是pad到该长度。
        :return: np.ndarray, torch.Tensor 。shape将是(B, max_length)， 元素类似为bool或torch.uint8
        """
        if isinstance(seq_len, np.ndarray):
            assert len(np.shape(seq_len)) == 1, f"seq_len can only have one dimension, got {len(np.shape(seq_len))}."
            max_len = int(max_len) if max_len else int(seq_len.max())
            broad_cast_seq_len = np.tile(np.arange(max_len), (len(seq_len), 1))
            mask = broad_cast_seq_len < seq_len.reshape(-1, 1)

        elif isinstance(seq_len, torch.Tensor):
            assert seq_len.dim() == 1, f"seq_len can only have one dimension, got {seq_len.dim() == 1}."
            batch_size = seq_len.size(0)
            max_len = int(max_len) if max_len else seq_len.max().long()
            broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
            mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))
        else:
            raise TypeError("Only support 1-d numpy.ndarray or 1-d torch.Tensor.")

        return mask

    """
        :param torch.LongTensor tokens: bsz x max_len, encoder的输入
        :param torch.LongTensor seq_len: bsz , batch中每一个句子的实际长度
        :return:
    """
    def forward(self,src_tokens, src_seq_len=None):
        if(src_seq_len==None):
            src_seq_len=torch.ones(src_tokens.size(1))
        mask = self.seq_len_to_mask(src_seq_len, max_len=src_tokens.size(1))
        dict = self.encoder(input_ids=src_tokens, attention_mask=mask, return_dict=True,
                                 output_hidden_states=True)
        encoder_outputs = dict.last_hidden_state
        hidden_states = dict.hidden_states
        return encoder_outputs, mask, hidden_states


class Lightdncoder(nn.Module):
    def __init__(self,decoder) -> None:
        self.decoder=decoder
    
if __name__=='__main__':
    model = BartModel.from_pretrained('facebook/bart-large')
    tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-large')
    st='i am winner'
    inputs = tokenizer(st, return_tensors="pt")
    print(st,tokenizer(st, return_tensors="pt"))
    outputs = model(**inputs)
    print(outputs)
