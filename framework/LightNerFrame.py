import torch
import numpy as np
from torch import optim
from torch.nn import parameter
from transformers.optimization import get_linear_schedule_with_warmup
from data_process.data_utils import data2device
from models.seq2seqLoss import *
from models.seq2seq_generator import SequenceGeneratorModel

class LightNerFrame:
    def __init__(self, model, tokenizer, args) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.device = torch.device(args.device)

    def __get_model_paras(self):
        parameters = []
        params = {'lr': self.args.bart_lr, 'weight_decay': 1e-2}
        params['params'] = [param for name,
                            param in self.model.named_parameters() if not ('prompt' in name)]

        params = {'lr': self.args.prompt_lr, 'weight_decay': 1e-2}
        params['params'] = [param for name,
                            param in self.model.named_parameters() if ('prompt' in name)]
        parameters.append(params)
        return parameters

    def eval(self, eval_dataloader=None):
        self.model.eval()
        pass

    def train(self, train_dataloader, eval_dataloader=None, epoch=100, batch_size_each_epoch=1000):
        parameters = self.__get_model_paras()
        self.model.train()
        if self.args.optimizer == 'AdamW':
            optimizer = optim.AdamW(parameters)

        if self.args.loss_func == 'Seq2SeqLoss':
            loss_func = Seq2SeqLoss
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=10, num_training_steps=batch_size_each_epoch)
        train_dataloader = iter(train_dataloader)

        for train_iter in range(epoch):

            for __ in range(batch_size_each_epoch):
                support, query = next(train_dataloader)
                query['encoder_input_id'] = query['encoder_input_id'].squeeze(
                    0)
                query['decoder_input_id'] = query['decoder_input_id'].squeeze(
                    0)
                query['encoder_input_length'] = query['encoder_input_length'].squeeze(
                    0)
                query['decoder_input_length'] = query['decoder_input_length'].squeeze(
                    0)
                support['encoder_input_id'] = support['encoder_input_id'].squeeze(
                    0)
                support['decoder_input_id'] = support['decoder_input_id'].squeeze(
                    0)
                support['encoder_input_length'] = support['encoder_input_length'].squeeze(
                    0)
                support['decoder_input_length'] = support['decoder_input_length'].squeeze(
                    0)
                data2device(self.device,support)
                data2device(self.device,query)

                pred = self.model(support['encoder_input_id'],
                                  support['decoder_input_id'],
                                  support['encoder_input_length'],
                                  support['decoder_input_length'],
                                  support['label_vocab_id'])

                loss = loss_func(support['decoder_input_id'],support['decoder_input_length'],pred['pred'])
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if eval_dataloader != None:
                    if train_iter % 10 == 0:
                        self.eval(eval_dataloader)
                        self.model.train()

    """
    给定source的内容，输出generate的内容

    :param torch.LongTensor src_tokens: bsz x max_len
    :param torch.LongTensor src_seq_len: bsz
    :param torch.LongTensor label_id: args.N
    :return:
    """
    def predict(self,src_tokens, src_seq_len=None,label_id=None):
        if src_seq_len.size(0)==1:
            src_seq_len=src_seq_len.squeeze(0)
        if src_tokens.size(0)==1:
            src_tokens=src_tokens.squeeze(0)

        generate_model = SequenceGeneratorModel(self.model,
        self.tokenizer.bos_token_id,
        self.tokenizer.eos_token_id,)
        self.model.eval()
        return generate_model.predict(src_tokens,src_seq_len=src_seq_len,label_id=label_id)
                

