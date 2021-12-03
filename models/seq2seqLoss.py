import torch.nn.functional as F
from data_process.data_utils import seq_len_to_mask
import torch

def Seq2SeqLoss(tgt_tokens, tgt_seq_len, pred):
    tgt_seq_len = tgt_seq_len - 1
    
    if(pred.shape[1]==tgt_tokens.shape[1]-1):#model.train
        mask = seq_len_to_mask(tgt_seq_len, max_len=tgt_tokens.size(1) - 1).eq(0)
        tgt_tokens = tgt_tokens[:, 1:].masked_fill(mask, -100)
    else :#model.eval
        mask = seq_len_to_mask(tgt_seq_len, max_len=tgt_tokens.size(1)).eq(0)
        tgt_tokens = tgt_tokens[:, :].masked_fill(mask, -100)
    loss = F.cross_entropy(target=tgt_tokens, input=pred.transpose(1, 2))
    return loss

