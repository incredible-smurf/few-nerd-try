import torch
from transformers.models.bart.light_ner_prompt_bart import *
from transformers.models.bart.configuration_bart import LightBartConfig
from framework.LightNerFrame import LightNerFrame
from models.light_ner import LightSeq2SeqModel,LightEncoder,LightDecoder
from models.seq2seq_generator import SequenceGeneratorModel

from metrics.seq2seqmetrics import Seq2SeqSpanMetric
from data_process.few_ner_dataset_for_bart import FewShotNERDataset
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
    parser.add_argument("--N", default=10, type=str, 
                        help="N ways" )
    parser.add_argument("--K", default=5, type=str, 
                        help="K shots" )
    parser.add_argument("--Q", default=1, type=str, 
                        help="Query set size" )

    parser.add_argument("--model_name_or_path", default="facebook/bart-base", type=str, 
                        help="Path to pre-trained model checkpoints or shortcut name selected in the list: " )
    parser.add_argument("--output_dir", default='outputs/p_tuning/', type=str, 
                        help="The output directory where the model predictions and checkpoints will be written.", )
    parser.add_argument("--data_dir", default='dataset/wiki_NER/processed', type=str, 
                        help="The input data dir.", )

    parser.add_argument("--optimizer", default='AdamW', type=str, 
                        help="the optimizer", )
    parser.add_argument("--loss_func", default='Seq2SeqLoss', type=str, 
                        help="the optimizer", )
    
    # prompt learning
    parser.add_argument("--p_tuning", type=bool, default=True)
    parser.add_argument("--entity_pseudo_token", type=str, default='[PROMPT]')
    parser.add_argument("--entity_split_token", type=str, default='[ENTITY_SPLIT]')
    
    parser.add_argument("--template", type=str, default="(9)")
    parser.add_argument("--decoder_template", type=str, default="(3)")
    parser.add_argument("--max_length", default=150, type=int, 
                        help="the max length of tokens.", )
    # contractive learning
    #parser.add_argument("--contrasive", type=bool, default=False)

    # train/dev setting
    """ parser.add_argument("--bsz_per_device", default=3, type=int, 
                        help="train/dev batch size per device", ) """
    parser.add_argument("--epoch", default=100, type=int, 
                        help="the number of training epochs.", )
    parser.add_argument("--batch_size_each_epoch", default=1000, type=int, 
                        help="the batch of few shot set per epoch.", )
    parser.add_argument("--max_steps", default=1000000, type=int, 
                        help="the number of training steps. \
                            If set to a positive number, \
                            the total number of training steps to perform. \
                            and it will override any value given in num_train_epochs", )
    parser.add_argument("--eval_per_train_epoch", default=10, type=int, 
                        help="when train function finish this epoch, then start a eval epoch", )
    parser.add_argument('--if_tensorboard', action='store_true', default=True)


    parser.add_argument("--bart_lr", default=1e-5, type=float, 
                        help="learning rate", )
    parser.add_argument("--prompt_lr", default=1e-5, type=float, 
                        help="learning rate", )
    parser.add_argument("--device", default='cpu', type=str, 
                        help="training device", )
    #parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--num_workers", default=5, type=int, 
                        help="the batch of few shot set per epoch.", )
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



#lignt ner model
configs = LightBartConfig.from_pretrained('facebook/bart-large')
configs.prompt_lenth = args.prompt_len

#----data
train_dataset =FewShotNERDataset(
        './data/few_ner/intra/train.txt', tokenizer, args.N, args.K, args.Q, args.max_length,args=args)
train_dataloader = data.DataLoader(train_dataset,num_workers=args.num_workers)
eval_dataset =FewShotNERDataset(
        './data/few_ner/intra/dev.txt', tokenizer, args.N, args.K, args.Q, args.max_length,args=args)
eval_dataloader=data.DataLoader(eval_dataset,num_workers=args.num_workers)

model = BartModel(configs)
checkpoint = torch.load(args.pytorch_model_path)
model.load_state_dict(checkpoint,strict=False)

model = LightSeq2SeqModel(model,args)
model = SequenceGeneratorModel(model,tokenizer.bos_token_id,eos_token_id=tokenizer.eos_token_id)


if args.metrics == 'seq2seqMetrics':
    metrics = Seq2SeqSpanMetric(tokenizer.eos_token_id,args.N)


Framework = LightNerFrame(model,tokenizer,args,metrics=metrics,train_dataloader=train_dataloader,eval_dataloader=eval_dataloader,writer=writer)
Framework.train(epoch=args.epoch,batch_size_each_epoch=args.batch_size_each_epoch)
print("train finished")




torch.save(model,os.path.join(save_path,'models.pth'))
writer.close()


tmp=next(iter(train_dataloader))[0]
Framework.predict(tmp['encoder_input_id'],label_id= tmp['label_vocab_id'],src_seq_len=tmp['encoder_input_length'])

