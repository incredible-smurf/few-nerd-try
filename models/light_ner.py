import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from transformers.models.bart.light_ner_prompt_bart import *
import torch.nn.functional as F
from data_process.data_utils import seq_len_to_mask

class LightSeq2SeqModel(torch.nn.Module):
    def __init__(self, Bartmodel, args):
        super().__init__()
        num_tokens, _ = Bartmodel.encoder.embed_tokens.weight.shape
        Bartmodel.resize_token_embeddings(200+num_tokens)
        self.encoder = LightEncoder(Bartmodel.encoder)
        self.decoder = LightDecoder(Bartmodel.decoder, args.pad_id, args.N+4)
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, src_tokens, tgt_tokens, src_seq_len=None, tgt_seq_len=None, label_id=None):
        """
        :param torch.LongTensor src_tokens: source的token
        :param torch.LongTensor tgt_tokens: target的token
        :param torch.LongTensor src_seq_len: src的长度
        :param torch.LongTensor tgt_seq_len: target的长度，默认用不上
        :param torch.LongTensor label_id:词表中实体类型词对应的位置
        :return: {'pred': torch.Tensor}, 其中pred的shape为bsz x max_len x vocab_size
        """
        state = self.prepare_state(src_tokens, src_seq_len)

        state['label_id'] = [torch.tensor(i) for i in range(3)]
        state['label_id'] += label_id
        # swtich whole to torch.Longtensor
        state['label_id'] = torch.LongTensor(
            [int(state['label_id'][i]) for i in range(len(state['label_id']))]).to(self.dummy_param.device)
        state['src_tokens'] = src_tokens

        decoder_output = self.decoder(tgt_tokens, state)
        if isinstance(decoder_output, torch.Tensor):
            return {'pred': decoder_output}
        elif isinstance(decoder_output, (tuple, list)):
            return {'pred': decoder_output[0]}
        else:
            raise TypeError(
                f"Unsupported return type from Decoder:{type(self.decoder)}")

    def prepare_state(self, src_tokens, src_seq_len=None,first=None):
        """
        调用encoder获取state，会把encoder的encoder_output, encoder_mask直接传入到decoder.init_state中初始化一个state

        :param src_tokens:
        :param src_seq_len:
        :return:
        """
        encoder_output, encoder_mask, encoder_hidden_state = self.encoder(
            src_tokens, src_seq_len)
        state = self.decoder.init_state(encoder_output, encoder_mask)
        state['num_samples'] = encoder_output.size(0)
        return state


class LightEncoder(torch.nn.Module):
    def __init__(self, encoder) -> None:
        super().__init__()
        self.encoder = encoder

    """
        :param torch.LongTensor tokens: bsz x max_len, encoder的输入
        :param torch.LongTensor seq_len: bsz , batch中每一个句子的实际长度
        :return:
    """
    def forward(self, src_tokens, src_seq_len=None):
        if(src_seq_len == None):
            src_seq_len = torch.ones(src_tokens.size(1))
        mask = seq_len_to_mask(src_seq_len, max_len=src_tokens.size(1))
        dict = self.encoder(input_ids=src_tokens, attention_mask=mask, return_dict=True,
                            output_hidden_states=True)
        encoder_outputs = dict.last_hidden_state
        hidden_states = dict.hidden_states
        return encoder_outputs, mask, hidden_states


class LightDecoder(torch.nn.Module):
    def __init__(self, decoder, pad_token_id, src_start_id, use_encoder_mlp=True,avg_feature=True) -> None:
        super().__init__()
        self.decoder = decoder
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
        self.dropout_layer = nn.Dropout(0.3)
        self.src_start_index = src_start_id
        self.avg_feature = avg_feature

    def init_state(self, encoder_output, encoder_mask):
        state = {'encoder_output': encoder_output,
                 'encoder_mask': encoder_mask,
                 'past_key_values':None}
        return state

    def forward(self, tokens, state):
        encoder_outputs = state['encoder_output']
        encoder_pad_mask = state['encoder_mask']

        tgt_pad_mask = tokens.eq(self.pad_token_id)
        mapping_token_mask = tokens.lt(self.src_start_index)
        mapped_tokens = tokens.masked_fill(tokens.ge(self.src_start_index), 0)
        tag_mapped_tokens = state['label_id'][mapped_tokens]

        src_tokens_index = tokens - self.src_start_index
        src_tokens_index = src_tokens_index.masked_fill(
            src_tokens_index.lt(0), 0)
        src_tokens = state['src_tokens']

        word_mapped_tokens = src_tokens.gather(index=src_tokens_index, dim=1)
        tokens = torch.where(mapping_token_mask,
                             tag_mapped_tokens, word_mapped_tokens)
        tokens = tokens.masked_fill(tgt_pad_mask, self.pad_token_id)

        if(self.training):
            tokens = tokens[:, :-1]
            decoder_pad_mask = ~tokens.eq(self.pad_token_id)
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_attention_mask=encoder_pad_mask,
                                attention_mask=decoder_pad_mask,
                                return_dict=True)
        else:
            past_key_values = state['past_key_values']
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_attention_mask=encoder_pad_mask,
                                attention_mask=None,
                                past_key_values=past_key_values,
                                use_cache=True,
                                return_dict=True)

        hidden_state = dict.last_hidden_state
        hidden_state = self.dropout_layer(hidden_state)#[bz,token_len,hidden]
        if not self.training:
            state['past_key_values'] = dict.past_key_values
        
        logits = hidden_state.new_full((hidden_state.size(0), hidden_state.size(1), self.src_start_index+src_tokens.size(-1)),
                                       fill_value=-1e24)
        
        eos_scores = F.linear(hidden_state, self.dropout_layer(self.decoder.embed_tokens.weight[:3]))  # bsz x max_len x 3   #eos bos pad
        tag_scores = F.linear(hidden_state, self.dropout_layer(self.decoder.embed_tokens.weight[state['label_id'][4:]]))  # bsz x max_len x num_class
        src_outputs = state['encoder_output']

        if hasattr(self, 'encoder_mlp'):
            src_outputs = self.encoder_mlp(src_outputs)
        
        mask = state['encoder_mask'].eq(0)
        mask = mask.unsqueeze(1)

        input_embed = self.dropout_layer(self.decoder.embed_tokens(src_tokens))


        if self.avg_feature:  # 先把feature合并一下
            src_outputs = (src_outputs + input_embed)/2
        word_scores = torch.einsum('blh,bnh->bln', hidden_state, src_outputs)  # bsz x max_len x max_word_len
        if not self.avg_feature:
            gen_scores = torch.einsum('blh,bnh->bln', hidden_state, input_embed)  # bsz x max_len x max_word_len
            word_scores = (gen_scores + word_scores)/2
        mask = mask.__or__(src_tokens.eq(2).cumsum(dim=1).ge(1).unsqueeze(1))# 2 represent eos_id
        word_scores = word_scores.masked_fill(mask, -1e32)

        logits[:, :, :3] = eos_scores
        logits[:, :, 4:self.src_start_index] = tag_scores
        logits[:, :, self.src_start_index:] = word_scores

        return logits

    def decode(self, tokens, state):
        return_tensor = self(tokens, state)[:, -1] #get last word logits
        return return_tensor



if __name__ == '__main__':
    model = BartModel.from_pretrained('facebook/bart-large')
    tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-large')
    st = 'i am winner'
    inputs = tokenizer(st, return_tensors="pt")
    print(st, tokenizer(st, return_tensors="pt"))
    outputs = model(**inputs)
    print(outputs)
