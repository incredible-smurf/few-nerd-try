from data_process.few_ner_dataset_for_bart_n_gram import FewShotNERDataset,data_collator

from models.model import BartForFewShotLearning
from transformers import BartConfig,BartTokenizer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer,AdamW
import torch.nn.functional as F
import argparse
import torch
import torch.nn
from sklearn.metrics import accuracy_score
import time

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
    parser.add_argument("--relation_pseudo_token", type=str, default='[PROMPT]')
    parser.add_argument("--entity_split_token", type=str, default='[ENTITY_SPLIT]')
    
    parser.add_argument("--template", type=str, default="(9)")
    
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

    assert type(args.template) is tuple
    return args

class simple_model(torch.nn.Module):
    def __init__(self,tokenizer,config, args,device ):
        super().__init__()
        self.a1=BartForFewShotLearning( tokenizer,config=config, args = args)
        #self.a2=torch.nn.Linear(5,2)
        self.tokenizer = tokenizer
        self.yes_id=(tokenizer.get_vocab()['yes'])
        self.no_id=(tokenizer.get_vocab()['no'])
        self.N=5
        self.device=device

    def forward(self,set):
        output=self.a1(
                origin_decoder_inputs=set['origin_decode_tensors'].to(self.device),
                prompt_encoder_inputs=torch.tensor(set['encoder_tensors']).to(self.device),
                prompt_decoder_inputs=torch.tensor(set['decoder_tensors']).to(self.device),
                lm_labels=torch.tensor(set['lm_labels']).to(self.device),
                prompt_labels=torch.tensor(set['prompt_labels']).to(self.device),
                prompt_task='ner'
        )
        yes_logits= output.logits[:,:,self.yes_id]

        tmp_yes=torch.ones((yes_logits.shape[0]))
        assert len(set['mask_loc'][0])==yes_logits.shape[0]
        for i in range(len(set['mask_loc'][0])):
            tmp_yes[i]=yes_logits[i][set['mask_loc'][0][i]]
            
        yes_logits=tmp_yes

        yes_logits=yes_logits.reshape(-1,self.N)
        yes_logits=F.softmax(yes_logits,dim=1)
        label= torch.argmax(yes_logits,dim=1)
        
        return label,yes_logits
        
        
    def load(self,path='./checkpoint/checkpoint-200000/pytorch_model.bin'):
        self.a1.load_state_dict(torch.load(path,map_location=self.device)) 





def test():
    args= get_args()
    device=torch.device('cpu')
    train_args = Seq2SeqTrainingArguments(output_dir=args.output_dir,
                                do_train=True,
                                do_eval=True,
                                evaluation_strategy="steps",
                                per_device_train_batch_size=args.bsz_per_device,
                                per_device_eval_batch_size=args.bsz_per_device,
                                learning_rate=args.lr,
                                num_train_epochs=args.epoch,
                                max_steps=args.max_steps,
                                warmup_ratio=0.05,
                                local_rank=args.local_rank,
                                log_level='info',
                                log_level_replica='warning',
                                log_on_each_node=True,
                                logging_steps=500,
                                eval_steps=500,
                                save_steps=10000,
                                save_total_limit=20,
                                p_tuning = args.p_tuning,
                                contrasive = args.contrasive,
                                entity_pseudo_token = args.entity_pseudo_token,
                                relation_pseudo_token = args.relation_pseudo_token,
                                template = args.template,
                                ner_template = args.ner_template,
                                label_names = ['lm_labels', 'prompt_labels']
                                )

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    
    tokenizer.add_special_tokens({'additional_special_tokens': [args.entity_pseudo_token,args.relation_pseudo_token]})
    

    dataset = FewShotNERDataset(
            './data/few_ner/intra/dev.txt', tokenizer, 5, 1, 1, 122,args=args)
    config = BartConfig.from_pretrained(args.model_name_or_path)
    #model = torch.load(r'.\checkpoint\checkpoint-40000\pytorch_model.bin',map_location=torch.device('cpu'))
    #model = BartForFewShotLearning( tokenizer,config=config, args = train_args)
    model =simple_model(tokenizer,config=config, args = train_args,device=device).to(device)
    dataloader = iter(torch.utils.data.DataLoader(dataset,batch_size=1,collate_fn=data_collator))

    optimizer = AdamW(model.parameters(),lr= 1e-05)
    loss_func=torch.nn.CrossEntropyLoss()

    loss_list=[]
    acc_list=[]
    
    for epoch in range(1000):
        #model.load_state_dict(torch.load(r'.\checkpoint\checkpoint-40000\pytorch_model.bin',map_location=device)) 
        model.load()
        model.train()
        support , query = next(dataloader)
        print(support)
        for i in range(5):
            with torch.no_grad():
                output_label,logits=model(
                support
                )
            label = support['label'][0]-1
            loss=loss_func(logits,label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        model.eval()
        output_label,logits=model(
            query
        )
        label =query['label'][0]-1
        with torch.no_grad():
            loss=loss_func(logits,label)
        loss_list.append(loss.item())
        acc_list.append(accuracy_score(label.cpu().tolist(),output_label.cpu().tolist()))
    with open('./results/ner/'+str(time.time())+'.txt','w') as f:
        print('acc',acc_list,file=f)
        print('loss',loss_list,file=f)
        


        
        
        

        




if __name__ == '__main__':
    test()
    #model = BartForFewShotLearning( tokenizer,config=config, args = train_args).load_state_dict(r'.\checkpoint\checkpoint-40000\pytorch_model.bin') 
    