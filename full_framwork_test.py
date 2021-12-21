import torch
#light ner  bart
#normal bert 
#from transformers.models.bart.modeling_bart import *
from transformers.models.bart.configuration_bart import LightBartConfig
from framework.LightNerFrame import LightNerFrame,NerPretrainFramework
from models.light_ner import NerSeq2SeqModel,LightEncoder,LightDecoder
from models.seq2seq_generator import SequenceGeneratorModel

from metrics.seq2seqmetrics import Seq2SeqSpanMetric
from data_process.few_ner_dataset_for_bart import FewShotNERDataset
from data_process.conll03_dataset_for_bart import NormalConll03Dataset

from transformers import BartTokenizer
import torch
import torch.utils.data as data
import argparse 
import time
import os
import json
from torch.utils.tensorboard import SummaryWriter

def get_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    # data
    parser.add_argument("--N", default=4, type=str, 
                        help="N ways" )
    parser.add_argument("--K", default=10, type=str, 
                        help="K shots" )
    parser.add_argument("--Q", default=1, type=str, 
                        help="Query set size" )

    parser.add_argument("--model_name_or_path", default=None, type=str, 
                        help="Path to pre-trained model")
    parser.add_argument("--dataset_choice", default='conll03',choices=['conll03','few_nerd'], type=str, 
                        help="The dataset to be trained", )

    parser.add_argument("--optimizer", default='AdamW', type=str, 
                        help="the optimizer", )
    parser.add_argument("--loss_func", default='Seq2SeqLoss', type=str, 
                        help="the optimizer", )

    #pretrain_setting
    parser.add_argument("--need_pretrian", default=True, type=bool, 
                        help="whether start pretrain or not", )
    parser.add_argument("--pretrian_epoch", default=80, type=int, 
                        help="how many epoch for pretrain", ) 
    parser.add_argument("--pretrian_batch_size", default=40, type=int, 
                        help="how many batch for pretrain", ) 
    parser.add_argument("--only_pretrian", default=True, type=bool, 
                        help="Only pretrain the model,when pretrain is over, the program will exit", )                   
    parser.add_argument("--pretrian_dataset", default='conll03', type=str, choices=['conll03'],
                        help="choose dataset for pretrain", )    
    parser.add_argument("--pretrian_eval_time", default=10, type=int,
                        help="every X epoch evaluate one times", ) 
    
    # prompt learning
    parser.add_argument("--p_tuning", type=bool, default=False)
    parser.add_argument("--entity_pseudo_token", type=str, default='[PROMPT]')
    parser.add_argument("--entity_split_token", type=str, default='[ENTITY_SPLIT]')
    
    parser.add_argument("--template", type=str, default="(9)")
    parser.add_argument("--decoder_template", type=str, default="(3)")
    parser.add_argument("--max_length", default=100, type=int, 
                        help="the max length of tokens.", )
    parser.add_argument("--prompt_model", default='promptv2',choices=['lightner','promptv2','normal'], type=str, 
                        help="choice of prompt_language_model", )
    


    #few shot train/dev setting
    """ parser.add_argument("--bsz_per_device", default=3, type=int, 
                        help="train/dev batch size per device", ) """
    parser.add_argument("--epoch", default=100, type=int, 
                        help="the number of training epochs.", )
    parser.add_argument("--batch_size_each_epoch", default=100, type=int, 
                        help="the batch of few shot set per epoch.", )
    parser.add_argument("--max_steps", default=1000000, type=int, 
                        help="the number of training steps. \
                            If set to a positive number, \
                            the total number of training steps to perform. \
                            and it will override any value given in num_train_epochs", )
    parser.add_argument("--eval_per_train_epoch", default=10, type=int, 
                        help="when train function finish this epoch, then start a eval epoch", )
    parser.add_argument("--tag_rate", default=2, type=int, 
                        help="workers number of dataloader", )  #由于实体标签很难被生成  所以增加tag_score的比例使之更易被生成

    parser.add_argument('--if_tensorboard', action='store_true', default=True)


    parser.add_argument("--bart_lr", default=2e-5, type=float, 
                        help="learning rate", )
    parser.add_argument("--prompt_lr", default=1e-4, type=float, 
                        help="learning rate", )
    parser.add_argument("--device", default='cpu', type=str, 
                        help="training device", )
    parser.add_argument("--num_workers", default=0, type=int, 
                        help="workers number of dataloader", )
    #metrics
    parser.add_argument("--metrics", default='seq2seqMetrics', type=str, 
                        help="evalate metircs", )


    args = parser.parse_args()

    args.template = eval(args.template) if type(args.template) is not tuple else args.template
    args.decoder_template=eval(args.decoder_template)

    return args


args=get_args()
args.pytorch_model_path='./pretrained_model/bart-large-facebook/model'
args.prompt_len=8
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
args.pad_id=tokenizer.pad_token_id





save_path='./checkpoint/'+str(time.time())
if not os.path.exists(save_path):
    os.makedirs(save_path)

with open(os.path.join(save_path,'args.json'),'w') as f:
    json.dump(args.__dict__, f, indent=2)

if args.if_tensorboard:
    tensor_board_path = os.path.join(save_path,'runs') 
    writer = SummaryWriter(tensor_board_path)
else:
    writer=None



#load the model 
configs = LightBartConfig.from_pretrained('facebook/bart-large')
configs.prompt_lenth = args.prompt_len
if args.prompt_model == 'promptv2':
    from transformers.models.bart.ptuningv2_prompt_bart import *
elif args.prompt_model =='lightner':
    from transformers.models.bart.light_ner_prompt_bart import *
elif args.prompt_model == 'normal':
    from transformers.models.bart.modeling_bart import *
else:raise NotImplementedError

if args.model_name_or_path!=None:
    model = BartModel(configs)
    checkpoint = torch.load(args.pytorch_model_path)
    model.load_state_dict(checkpoint,strict=False)

    model = NerSeq2SeqModel(model,args)
    model = SequenceGeneratorModel(model,tokenizer.bos_token_id,eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id)
else :
    model = BartModel(configs)
    checkpoint = torch.load('./pretrained_model/bart-large-facebook/model')
    model.load_state_dict(checkpoint,strict=False)

    model = NerSeq2SeqModel(model,args)
    model = SequenceGeneratorModel(model,tokenizer.bos_token_id,eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id)

# pretrain part
if args.need_pretrian:
    pretrain_frame= NerPretrainFramework(model,args,writer)
    if args.pretrian_dataset =='conll03':
        train_dataset =NormalConll03Dataset(
        './data/conll03/train.txt',tokenizer=tokenizer)
        train_dataloader = data.DataLoader(train_dataset,num_workers=args.num_workers,batch_size=args.pretrian_batch_size)
        eval_dataset =NormalConll03Dataset(
        './data/conll03/eval.txt',tokenizer=tokenizer)
        eval_dataloader = data.DataLoader(eval_dataset,num_workers=args.num_workers,batch_size=args.pretrian_batch_size)
    else:raise NotImplementedError

    print('========start pretrain===========')
    pretrain_frame.pretrain(train_dataloader,eval_dataloader,os.path.join(save_path,'pretrain'))


elif not args.need_pretrian or not args.only_pretrian:
    #lignt ner model
    #----data
    print('start training')
    if args.dataset_choice =='few_nerd':
        train_dataset =FewShotNERDataset(
                './data/few_ner/intra/train.txt', tokenizer, args.N, args.K, args.Q, args.max_length,args=args)
        train_dataloader = data.DataLoader(train_dataset,num_workers=args.num_workers)
        eval_dataset =FewShotNERDataset(
                './data/few_ner/intra/dev.txt', tokenizer, args.N, args.K, args.Q, args.max_length,args=args)
        eval_dataloader=data.DataLoader(eval_dataset,num_workers=args.num_workers)
    elif args.dataset_choice =='conll03':
        train_dataset =FewShotNERDataset(
                './data/conll03/train.txt', tokenizer, args.N, args.K, args.Q, args.max_length,args=args)
        train_dataloader = data.DataLoader(train_dataset,num_workers=args.num_workers)
        eval_dataset =FewShotNERDataset(
                './data/conll03/dev.txt', tokenizer, args.N, args.K, args.Q, args.max_length,args=args)
        eval_dataloader=data.DataLoader(eval_dataset,num_workers=args.num_workers)
    else :raise NotImplementedError
        
    if args.metrics == 'seq2seqMetrics':
        metrics = Seq2SeqSpanMetric(tokenizer.eos_token_id,args.N)


    Framework = LightNerFrame(model,tokenizer,args,metrics=metrics,train_dataloader=train_dataloader,eval_dataloader=eval_dataloader,writer=writer)
    Framework.train(epoch=args.epoch,batch_size_each_epoch=args.batch_size_each_epoch,save_path=os.path.join(save_path,'model'))
    print("train finished")




    torch.save(model,os.path.join(save_path,'models.pth'))
writer.close()
