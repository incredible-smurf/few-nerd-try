import torch.nn.functional as F
from data_process.data_utils import seq_len_to_mask
import torch
import warnings

#pred bz seq_len logit
#tgt_tokens bz seq_len
def Seq2SeqLoss(tgt_tokens, tgt_seq_len, pred):
    tgt_seq_len = tgt_seq_len - 1
    
    if(pred.shape[1]==tgt_tokens.shape[1]-1):#model.train
        mask = seq_len_to_mask(tgt_seq_len, max_len=tgt_tokens.size(1) - 1).eq(0)
        tgt_tokens = tgt_tokens[:, 1:].masked_fill(mask, -100)
    else :#model.eval
        mask = seq_len_to_mask(tgt_seq_len, max_len=tgt_tokens.size(1)).eq(0)
        tgt_tokens = tgt_tokens[:, :].masked_fill(mask, -100)
        warnings.warn("calulate loss when in eval")
    bz=tgt_tokens.shape[0]
    seq_len = tgt_tokens.shape[1]
    loss = F.cross_entropy(target=tgt_tokens.reshape(bz*seq_len), input=pred.reshape(bz*seq_len,-1),ignore_index=-100)
    return loss

