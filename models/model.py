import copy
import math
import random

from torch._C import Size
from transformers.training_args import TrainingArguments
import warnings
from typing import Optional, Tuple
import logging

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from transformers import BartPretrainedModel, BartConfig, BartModel, BartTokenizer
from transformers.modeling_outputs import Seq2SeqLMOutput
from .prompt_encoder import PromptEncoder


logger = logging.getLogger(__name__)
max_output = 0

def shift_tokens_right(input_ids: torch.Tensor, shift_step: int, pad_token_id: int=0):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, shift_step:] = input_ids[:, :-shift_step].clone()
    shifted_input_ids[:, :shift_step] = pad_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

def shift_tokens_left(input_ids: torch.Tensor, shift_step: int, pad_token_id: int=0):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, :-shift_step] = input_ids[:, shift_step:].clone()
    shifted_input_ids[:, -shift_step:] = pad_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

class BartForFewShotLearning(BartPretrainedModel):
    base_model_prefix = "model"
    # _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]
    
    def __init__(self,Tokenizer, config: BartConfig, args: TrainingArguments):
        super().__init__(config)
        self.args = args
        self.model = BartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.tokenizer = Tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.unk_token_id

        self.embeddings = self.get_input_embeddings()

        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # For p-tuning
        
        self.entity_pseudo_token_id = self.tokenizer.get_vocab()[self.args.entity_pseudo_token]
        self.relation_pseudo_token_id = self.tokenizer.get_vocab()[self.args.relation_pseudo_token]
        
        self.template = self.args.template
        self.ner_template = self.args.ner_template
        self.spell_length = max([sum(self.template), sum(self.ner_template)])
        self.prompt_encoder = PromptEncoder(self.spell_length, self.embeddings.embedding_dim)

        self.prompt_loss_weight = 1.
        self.contrasive_loss_weight = 3.
        self.init_weights()

    def get_input_embeddings(self):
        return self.model.shared

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def embed_input(self, sentences, pseudo_token_id=None, template=None):
        sentences_for_embedding = sentences.clone()
        sentences_for_embedding[(sentences == pseudo_token_id)] = self.tokenizer.unk_token_id

        raw_embeds = self.embeddings(sentences_for_embedding)

        if pseudo_token_id is not None:
            bz = sentences.shape[0]
            task_spell_length = self.spell_length
            if template is not None:
                task_spell_length = sum(template)

            blocked_indices = (sentences == pseudo_token_id).nonzero().reshape((bz, task_spell_length, 2))[:, :, 1]  # bz
            
            replace_embeds = self.prompt_encoder(torch.LongTensor(list(range(task_spell_length))).to(sentences.device))

            for bidx in range(bz):
                for i in range(task_spell_length):
                    raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]

        return raw_embeds

    def get_hidden_states(self, inputs, decoder_inputs, prompt_token=None, task_template=None, return_dict=None):
        inputs_embeds = self.embed_input(inputs, prompt_token, task_template)
        decoder_inputs_embeds=self.embed_input(decoder_inputs, prompt_token, task_template)

        attention_mask = (inputs != self.pad_token_id).bool()
        decoder_attention_mask = (decoder_inputs != self.pad_token_id).bool()

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_attention_mask=decoder_attention_mask,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]

        global max_output
        if hidden_states.shape[1] > max_output:
            max_output = hidden_states.shape[1]
            logger.info('max length of output sequence changes to [%d] ' % int(max_output))
        
        return hidden_states, outputs

    def get_sentence_representation(self, 
                                    inputs, 
                                    decoder_inputs, 
                                    prompt_token=None, 
                                    task_template=None, 
                                    return_dict=None):
        
        hidden_states, outputs = self.get_hidden_states(inputs, decoder_inputs, prompt_token, task_template, return_dict)

        eos_mask = decoder_inputs.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[
            :, -1, :
        ]

        return sentence_representation

    def get_lm_prompt_loss(self, 
                            inputs, # 35 79
                            decoder_inputs, #35 79
                            lm_labels, # 30 71
                            prompt_labels, 
                            prompt_token, 
                            task_template, 
                            return_dict=None, 
                            origin_sentence=None):
        hidden_states, outputs = self.get_hidden_states(inputs, decoder_inputs, prompt_token, task_template, return_dict)
        lm_logits = self.lm_head(hidden_states) + self.final_logits_bias# 30 95 hidden

        true_mask = torch.LongTensor([[1] * (origin_sentence.size(1)-1)] * origin_sentence.size(0)).bool()# 30 70
        extent_mask = torch.LongTensor([[0] * (decoder_inputs.size(1)-(origin_sentence.size(1)-1))] * origin_sentence.size(0)).bool()# 30 25
        lm_loss_mask = torch.cat((true_mask, extent_mask), -1) # 30 95 

        loss_fct = CrossEntropyLoss()
        lm_loss = loss_fct(lm_logits[lm_loss_mask, :].view(-1, self.config.vocab_size), lm_labels.view(-1))

        prompt_mask = shift_tokens_left(decoder_inputs.eq(self.tokenizer.eos_token_id), 2)
        prompt_representation = lm_logits[prompt_mask, :].view(lm_logits.size(0), -1, lm_logits.size(-1))[
                :, -1, :
        ]
        prompt_loss = loss_fct(prompt_representation.view(-1, self.config.vocab_size), prompt_labels.view(-1))
        return lm_loss, prompt_loss, lm_logits, outputs
    
    def get_lm_loss(self, 
                    inputs, 
                    decoder_inputs, 
                    labels, 
                    return_dict=None):
        hidden_states, outputs = self.get_hidden_states(inputs, decoder_inputs, return_dict=return_dict)
        lm_logits = self.lm_head(hidden_states) + self.final_logits_bias

        lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        return lm_loss, lm_logits, outputs

    def get_contrasive_task_loss(self, 
                            anchor_inputs, 
                            anchor_decoder_inputs, 
                            pos_inputs, 
                            pos_decoder_inputs, 
                            neg_inputs, 
                            neg_decoder_inputs, 
                            prompt_token, 
                            task_template):

        anchor_representation = self.get_sentence_representation(anchor_inputs, anchor_decoder_inputs)
        pos_representation = self.get_sentence_representation(pos_inputs, pos_decoder_inputs, prompt_token, task_template)
        neg_representation = self.get_sentence_representation(neg_inputs, neg_decoder_inputs, prompt_token, task_template)
        return F.triplet_margin_loss(anchor_representation, pos_representation, neg_representation)

    def forward(
        self,
        origin_encoder_inputs=None,
        mask_encoder_inputs=None,
        origin_decoder_inputs=None,
        prompt_encoder_inputs=None,
        prompt_decoder_inputs=None,
        pos_encoder_inputs=None,
        pos_decoder_inputs=None,
        neg_encoder_inputs=None,
        neg_decoder_inputs=None,
        lm_labels=None,
        prompt_labels=None,
        prompt_task=None,
        return_dict=None,
    ):
        # print(origin_encoder_inputs.size())
        # print(mask_encoder_inputs.size())
        # print(origin_decoder_inputs.size())
        # print(prompt_encoder_inputs.size())
        # print(prompt_decoder_inputs.size())
        # print(pos_encoder_inputs.size())
        # print(pos_decoder_inputs.size())
        # print(neg_encoder_inputs.size())
        # print(neg_decoder_inputs.size())
        # print(lm_labels.size())
        # print(prompt_labels)
        # print(prompt_task)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.args.p_tuning:
            if prompt_task == 're':
                prompt_token = self.relation_pseudo_token_id
                task_template = self.template
            elif prompt_task == 'ner':
                prompt_token = self.entity_pseudo_token_id
                task_template = self.ner_template
            
            lm_loss, prompt_loss, lm_logits, outputs = self.get_lm_prompt_loss(prompt_encoder_inputs, 
                                                                                prompt_decoder_inputs, 
                                                                                lm_labels, 
                                                                                prompt_labels, 
                                                                                prompt_token, 
                                                                                task_template, 
                                                                                return_dict, 
                                                                                origin_decoder_inputs)

        else:
            lm_loss, lm_logits, outputs = self.get_lm_loss(mask_encoder_inputs, 
                                                            origin_decoder_inputs, 
                                                            lm_labels, 
                                                            return_dict)
        
        if self.args.p_tuning and self.args.contrasive:
            contrasive_loss = self.get_contrasive_task_loss(origin_encoder_inputs, 
                                                            origin_decoder_inputs, 
                                                            pos_encoder_inputs, 
                                                            pos_decoder_inputs, 
                                                            neg_encoder_inputs, 
                                                            neg_decoder_inputs, 
                                                            prompt_token, 
                                                            task_template)

        loss = None
        if self.args.p_tuning:
            if self.args.contrasive:
                loss = lm_loss + self.prompt_loss_weight * prompt_loss + self.contrasive_loss_weight * contrasive_loss
                # print(lm_loss)
                # print(prompt_loss)
                # print(contrasive_loss)
            else:
                loss = lm_loss + self.prompt_loss_weight * prompt_loss
        else:
            loss = lm_loss

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return((loss) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss= loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past
