import torch
from transformers.models.bart.light_ner_prompt_bart import *
from transformers.models.bart.configuration_bart import LightBartConfig

from transformers import BartTokenizer
import torch

configs= LightBartConfig.from_pretrained('facebook/bart-large')
configs.prompt_lenth = 8
pytorch_model_path='./pretrained_model/bart-large-facebook/model'

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
model = BartModel(configs)


checkpoint = torch.load(pytorch_model_path)

model.load_state_dict(checkpoint,strict=False)
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
prompt_len=8

#此处要把prompt长度加上
#由于prompt在尾部，padding的词mask也要为1
inputs['attention_mask']=torch.cat((torch.tensor([[1]*prompt_len]),inputs['attention_mask']),dim=1)
print(inputs['attention_mask'].shape)
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state