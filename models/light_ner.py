import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from transformer.models.bart.light_ner_prompt_bart import *
from fastNLP.modules import Seq2SeqEncoder, Seq2SeqDecoder, State
from fastNLP.models import Seq2SeqModel




class LightSeq2SeqModel(Seq2SeqModel):
    def __init__(self, encoder: Seq2SeqEncoder, decoder: Seq2SeqDecoder):
        super().__init__(encoder, decoder)

    
    def forward(self, src_tokens, tgt_tokens, src_seq_len=None, tgt_seq_len=None):
        """

        :param torch.LongTensor src_tokens: source的token
        :param torch.LongTensor tgt_tokens: target的token
        :param torch.LongTensor src_seq_len: src的长度
        :param torch.LongTensor tgt_seq_len: target的长度，默认用不上
        :return: {'pred': torch.Tensor}, 其中pred的shape为bsz x max_len x vocab_size
        """
        state = self.prepare_state(src_tokens, src_seq_len)
        decoder_output = self.decoder(tgt_tokens, state)
        if isinstance(decoder_output, torch.Tensor):
            return {'pred': decoder_output}
        elif isinstance(decoder_output, (tuple, list)):
            return {'pred': decoder_output[0]}
        else:
            raise TypeError(f"Unsupported return type from Decoder:{type(self.decoder)}")
    
    def prepare_state(self, src_tokens, src_seq_len=None):
        """
        调用encoder获取state，会把encoder的encoder_output, encoder_mask直接传入到decoder.init_state中初始化一个state

        :param src_tokens:
        :param src_seq_len:
        :return:
        """
        encoder_output, encoder_mask = self.encoder(src_tokens, src_seq_len)
        state = self.decoder.init_state(encoder_output, encoder_mask)
        return state
    
    @classmethod
    def build_model(cls, bart_model, tokenizer, label_ids, decoder_type=None,
                    use_encoder_mlp=False):
        model = BartModel.from_pretrained(bart_model)
        num_tokens, _ = model.encoder.embed_tokens.weight.shape
        model.resize_token_embeddings(len(tokenizer.unique_no_split_tokens)+num_tokens)
        encoder = model.encoder
        decoder = model.decoder

        _tokenizer = BartTokenizer.from_pretrained(bart_model)
        for token in tokenizer.unique_no_split_tokens:
            if token[:2] == '<<':  # 特殊字符
                index = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
                if len(index)>1:
                    raise RuntimeError(f"{token} wrong split")
                else:
                    index = index[0]
                assert index>=num_tokens, (index, num_tokens, token)
                indexes = _tokenizer.convert_tokens_to_ids(_tokenizer.tokenize(token[2:-2]))
                embed = model.encoder.embed_tokens.weight.data[indexes[0]]
                for i in indexes[1:]:
                    embed += model.decoder.embed_tokens.weight.data[i]
                embed /= len(indexes)
                model.decoder.embed_tokens.weight.data[index] = embed

        encoder =LightEncoder(encoder)
        decoder = Lightdecoder(decoder)

        return cls(encoder=encoder, decoder=decoder)
    



class LightEncoder(Seq2SeqEncoder):
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


class Lightdecoder(Seq2SeqDecoder):
    def __init__(self,decoder, pad_token_id,use_encoder_mlp=True) -> None:
        self.decoder=decoder
        super().__init__()
        self.decoder = decoder
        causal_mask = torch.zeros(512, 512).fill_(float('-inf'))
        causal_mask = causal_mask.triu(diagonal=1)
        self.register_buffer('causal_masks', causal_mask.float())
        self.pad_token_id = pad_token_id  # 加上一个
        hidden_size = decoder.embed_tokens.weight.size(1)
        if use_encoder_mlp:
            self.encoder_mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                             nn.Dropout(0.3),
                                             nn.ReLU(),
                                             nn.Linear(hidden_size, hidden_size))

    def forward(self,tokens, state):
        encoder_outputs = state.encoder_output
        encoder_pad_mask = state.encoder_mask

        if(self.training):
            decoder_pad_mask = tokens.eq(self.pad_token_id)
            dict = self.decoder(input_ids=tokens,
                            encoder_hidden_states=encoder_outputs,
                            encoder_padding_mask=encoder_pad_mask,
                            decoder_padding_mask=decoder_pad_mask,
                            decoder_causal_mask=self.causal_masks[:tokens.size(1), :tokens.size(1)],
                            return_dict=True)
                        
        return 1
        
        


    
if __name__=='__main__':
    model = BartModel.from_pretrained('facebook/bart-large')
    tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-large')
    st='i am winner'
    inputs = tokenizer(st, return_tensors="pt")
    print(st,tokenizer(st, return_tensors="pt"))
    outputs = model(**inputs)
    print(outputs)
