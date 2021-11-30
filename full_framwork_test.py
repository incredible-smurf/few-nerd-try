import torch
from transformers.models.bart.light_ner_prompt_bart import *
from transformers.models.bart.configuration_bart import LightBartConfig
from framework.LightNerFrame import LightNerFrame
from models.light_ner import LightSeq2SeqModel,LightEncoder,LightDecoder


from data_process.few_ner_dataset_for_bart import FewShotNERDataset
from transformers import BartTokenizer
import torch
import torch.utils.data as data
import argparse 

def get_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    # data
    parser.add_argument("--N", default=3, type=str, 
                        help="N ways" )
    parser.add_argument("--K", default=5, type=str, 
                        help="K shots" )
    parser.add_argument("--Q", default=2, type=str, 
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
    
    # contractive learning
    #parser.add_argument("--contrasive", type=bool, default=False)

    # train/dev settting
    """ parser.add_argument("--bsz_per_device", default=3, type=int, 
                        help="train/dev batch size per device", ) """
    parser.add_argument("--epoch", default=50, type=int, 
                        help="the number of training epochs.", )
    parser.add_argument("--max_steps", default=1000000, type=int, 
                        help="the number of training steps. \
                            If set to a positive number, \
                            the total number of training steps to perform. \
                            and it will override any value given in num_train_epochs", )
    parser.add_argument("--bart_lr", default=1e-5, type=float, 
                        help="learning rate", )
    parser.add_argument("--prompt_lr", default=1e-5, type=float, 
                        help="learning rate", )
    parser.add_argument("--device", default='cpu', type=str, 
                        help="training device", )
    #parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    
    args = parser.parse_args()

    args.template = eval(args.template) if type(args.template) is not tuple else args.template
    args.decoder_template=eval(args.decoder_template)

    return args


args=get_args()
args.pytorch_model_path='./pretrained_model/bart-large-facebook/model'
args.prompt_len=8
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
args.pad_id=tokenizer.pad_token_id



#lignt ner model
configs = LightBartConfig.from_pretrained('facebook/bart-large')
configs.prompt_lenth = args.prompt_len

#----data
train_dataset =FewShotNERDataset(
        './data/few_ner/intra/train.txt', tokenizer, args.N, args.K, args.Q, 122,args=args)
dataloader = data.DataLoader(train_dataset)


model = BartModel(configs)
checkpoint = torch.load(args.pytorch_model_path)
model.load_state_dict(checkpoint,strict=False)

model = LightSeq2SeqModel(model,args)



Framework =LightNerFrame (model,tokenizer,args)
Framework.train(dataloader,epoch=1,batch_size_each_epoch=1)
print("train finished")
Framework.predict(torch.tensor([[1,412,2213]]))








