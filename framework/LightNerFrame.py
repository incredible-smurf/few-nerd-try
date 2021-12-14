from pickle import NONE
import torch
import numpy as np
from torch import optim
from torch.nn import parameter
from transformers.optimization import get_linear_schedule_with_warmup
from data_process.data_utils import data2device
from models.seq2seqLoss import *
from models.seq2seq_generator import SequenceGeneratorModel
from tqdm import tqdm

class LightNerFrame:
    def __init__(self, model, tokenizer, args,metrics=None,train_dataloader=None,eval_dataloader=None,writer=None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.device = torch.device(args.device)
        self.metrics=metrics
        self.train_dataloader=train_dataloader
        self.eval_dataloader=eval_dataloader
        self.writer=writer

    def __get_model_optim_paras(self):
        parameters = []
        params = {'lr': self.args.bart_lr, 'weight_decay': 1e-2}
        params['params'] = [param for name,
                            param in self.model.named_parameters() if not ('prompt' in name)]

        params = {'lr': self.args.prompt_lr, 'weight_decay': 1e-2}
        params['params'] = [param for name,
                            param in self.model.named_parameters() if ('prompt' in name)]
        parameters.append(params)
        return parameters

    def evaluate(self,iters=None,check_times=10):
        
        parameters = self.__get_model_optim_paras()
        tmp_model_state=self.model.state_dict().copy()
        eval_dataloader = iter(self.eval_dataloader)
        if self.args.optimizer == 'AdamW':
            optimizer = optim.AdamW(parameters)

        if self.args.loss_func == 'Seq2SeqLoss':
            loss_func = Seq2SeqLoss

        fn,tp,fp,loss_item=0,0,0,0
        for __ in range(check_times):
            for name, param in tmp_model_state.items():
                self.model.state_dict()[name]=param
            self.model.train()
            support, query = next(eval_dataloader)
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
            loss_item+=loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            with torch.no_grad():
                self.model.eval()
                pred = self.model.predict(query['encoder_input_id'],
                                    query['label_vocab_id'],
                                    src_seq_len=query['encoder_input_length']
                                    )
                #print("===========")
                #print(pred['pred'],query['decoder_input_id'])
                self.metrics.evaluate(query['span'],pred['pred'],query['decoder_input_id'])
        result = self.metrics.get_metric()
        loss_item/=check_times
        for name, param in tmp_model_state.items():
            self.model.state_dict()[name]=param
        if iters!=None:
            assert isinstance(iters,int)
        if self.writer!=None and iters !=None:
            self.writer.add_scalars('Eval',{'loss_item':loss_item,'f':result['f'],'rec':result['rec'],'pre':result['pre']},iters)

        

    def train(self, epoch=100, batch_size_each_epoch=1000,save_path=None):
        parameters = self.__get_model_optim_paras()
        self.model.train()
        self.model.to(self.device)
        if self.args.optimizer == 'AdamW':
            optimizer = optim.AdamW(parameters)

        if self.args.loss_func == 'Seq2SeqLoss':
            loss_func = Seq2SeqLoss
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=10, num_training_steps=batch_size_each_epoch)
        train_dataloader = iter(self.train_dataloader)

        min_loss=99999
        for train_iter in tqdm(range(epoch)):
            fn,tp,fp,loss_item=0,0,0,0
            for __ in range(batch_size_each_epoch):
                self.model.train()
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
                #需要过拟合
                for ____ in range(1):
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
                loss_item +=loss.item()
                #fn,tp,fp =self.metrics.evluate(support['span'],pred,support['decoder_input_id'])
                with torch.no_grad():
                    self.model.eval()
                    pred = self.model.predict(query['encoder_input_id'],
                                        query['label_vocab_id'],
                                        src_seq_len=query['encoder_input_length']
                                        )
                
                    self.metrics.evaluate(query['span'],pred['pred'],query['decoder_input_id'])

                
            result = self.metrics.get_metric()

            tqdm.write("f {}, loss {} ".format(result['f'], loss_item))
            tqdm.write("pred format now {}".format(pred['pred']))
            if self.eval_dataloader != None:
                if train_iter % self.args.eval_per_train_epoch == 0:
                    self.evaluate(iters=train_iter)
                    self.model.train()

            if self.writer!=None:
                self.writer.add_scalars('Train',{'loss_item':loss_item,'f':result['f'],'rec':result['rec'],'pre':result['pre']},train_iter)
            if save_path!=None:
                torch.save(self.model,save_path)
                if min_loss>loss_item:
                    min_loss=loss_item
                    torch.save(self.model,save_path+'tmp_best')



    #for debug
    def train_de(self, epoch=1, batch_size_each_epoch=1):
        parameters = self.__get_model_optim_paras()
        self.model.train()
        self.model.to(self.device)
        if self.args.optimizer == 'AdamW':
            optimizer = optim.AdamW(self.model.parameters(),lr=self.args.bart_lr)

        if self.args.loss_func == 'Seq2SeqLoss':
            loss_func = Seq2SeqLoss
        
        #scheduler = get_linear_schedule_with_warmup(
            #optimizer, num_warmup_steps=10, num_training_steps=batch_size_each_epoch)
        train_dataloader = iter(self.train_dataloader)

        for train_iter in tqdm(range(epoch)):
            fn,tp,fp,loss_item=0,0,0,0
            self.model.train()
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
            for j in range(500):
                self.model.train()
                pred = self.model(support['encoder_input_id'],
                                support['decoder_input_id'],
                                support['encoder_input_length'],
                                support['decoder_input_length'],
                                support['label_vocab_id'])

                loss = loss_func(support['decoder_input_id'],support['decoder_input_length'],pred['pred'])
                loss.backward()
                optimizer.step()
                #scheduler.step()
                optimizer.zero_grad()
                loss_item +=loss.item()
                #fn,tp,fp =self.metrics.evluate(support['span'],pred,support['decoder_input_id'])
                with torch.no_grad():
                    self.model.eval()
                    pred = self.model.predict(support['encoder_input_id'],
                                        support['label_vocab_id'],
                                        src_seq_len=support['encoder_input_length']
                                        )
                
                    result=self.metrics.evaluate(support['span'],pred['pred'],support['decoder_input_id'])
                    fn+=result[0]
                    tp+=result[1]
                    fp+=result[2]
                print('loss',loss.item(),'epoch',j)
                if j %10==0:
                    print('pred',pred['pred'])
                
            fn/=batch_size_each_epoch
            tp/=batch_size_each_epoch
            fp/=batch_size_each_epoch
            loss_item/=batch_size_each_epoch
            tqdm.write("fn {}, loss {} ".format(fn, loss_item))
            tqdm.write("pred format now {}".format(pred['pred']))
            if self.eval_dataloader != None:
                if train_iter % self.args.eval_per_train_epoch == 0:
                    self.evaluate(iters=train_iter)
                    self.model.train()

            if self.writer!=None:
                self.writer.add_scalars('Train',{'loss_item':loss_item,'fn':fn,'fp':fp,'tp':tp},train_iter)


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
        self.model.eval()
        return self.model.predict(src_tokens,src_seq_len=src_seq_len,label_id=label_id)
                

