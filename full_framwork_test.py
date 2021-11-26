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
    parser.add_argument("--model_name_or_path", default="facebook/bart-base", type=str, 
                        help="Path to pre-trained model checkpoints or shortcut name selected in the list: " )
    parser.add_argument("--output_dir", default='outputs/p_tuning/', type=str, 
                        help="The output directory where the model predictions and checkpoints will be written.", )
    parser.add_argument("--data_dir", default='dataset/wiki_NER/processed', type=str, 
                        help="The input data dir.", )
    # prompt learning
    parser.add_argument("--p_tuning", type=bool, default=True)
    parser.add_argument("--entity_pseudo_token", type=str, default='[PROMPT]')
    parser.add_argument("--entity_split_token", type=str, default='[ENTITY_SPLIT]')
    
    parser.add_argument("--template", type=str, default="(9)")
    parser.add_argument("--decoder_template", type=str, default="(3)")
    
    # contractive learning
    parser.add_argument("--contrasive", type=bool, default=False)

    # train/dev settting
    parser.add_argument("--bsz_per_device", default=3, type=int, 
                        help="train/dev batch size per device", )
    parser.add_argument("--epoch", default=50, type=int, 
                        help="the number of training epochs.", )
    parser.add_argument("--max_steps", default=1000000, type=int, 
                        help="the number of training steps. \
                            If set to a positive number, \
                            the total number of training steps to perform. \
                            and it will override any value given in num_train_epochs", )
    parser.add_argument("--lr", default=1e-5, type=float, 
                        help="learning rate", )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    
    args = parser.parse_args()

    args.template = eval(args.template) if type(args.template) is not tuple else args.template
    args.decoder_template=eval(args.decoder_template)

    return args


args=get_args()
args.pytorch_model_path='./pretrained_model/bart-large-facebook/model'
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

#lignt ner model
configs = LightBartConfig.from_pretrained('facebook/bart-large')
configs.prompt_lenth = 8
model = BartModel(configs)
checkpoint = torch.load(args.pytorch_model_path)
model.load_state_dict(checkpoint,strict=False)

model = LightSeq2SeqModel(LightEncoder(model.encoder),LightDecoder(model.decoder,tokenizer.pad_token_id))


train_dataset =FewShotNERDataset(
        './data/few_ner/intra/train.txt', tokenizer, 3, 2, 2, 122,args=args)

dataloader = data.DataLoader(train_dataset)


Framework =LightNerFrame (model,tokenizer,args)










inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
prompt_len=8

#此处要把prompt长度加上
#由于prompt在尾部，padding的词mask也要为1
inputs['attention_mask']=torch.cat((torch.tensor([[1]*prompt_len]),inputs['attention_mask']),dim=1)
print(inputs['attention_mask'].shape)
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state