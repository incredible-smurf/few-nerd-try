from urllib.parse import quote_from_bytes
import torch.utils.data as data
import torch
import random
import os
import numpy as np
from transformers import BertTokenizer,BartTokenizer
from torch.nn.utils.rnn import pad_sequence
import argparse
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

#entity classification
def get_class_name(rawtag):
    # get (finegrained) class name
    if rawtag.startswith('B-') or rawtag.startswith('I-'):
        return rawtag[2:]
    else:
        return rawtag


class Sample():
    def __init__(self, filelines):
        filelines = [line.split('\t') for line in filelines]
        self.words, self.tags = zip(*filelines)
        self.words = [word.lower() for word in self.words]
        # strip B-, I-
        self.normalized_tags = list(map(get_class_name, self.tags))
        self.class_count = {}

    def __count_entities__(self):
        current_tag = self.normalized_tags[0]
        for tag in self.normalized_tags[1:]:
            if tag == current_tag:
                continue
            else:
                if current_tag != 'O':
                    if current_tag in self.class_count:
                        self.class_count[current_tag] += 1
                    else:
                        self.class_count[current_tag] = 1
                current_tag = tag
        if current_tag != 'O':
            if current_tag in self.class_count:
                self.class_count[current_tag] += 1
            else:
                self.class_count[current_tag] = 1

    def get_class_count(self):
        if self.class_count:
            return self.class_count
        else:
            self.__count_entities__()
            return self.class_count

    def get_tag_class(self):
        # strip 'B' 'I'
        tag_class = list(set(self.normalized_tags))
        if 'O' in tag_class:
            tag_class.remove('O')
        return tag_class

    def valid(self, target_classes):
        return (set(self.get_class_count().keys()).intersection(set(target_classes))) and not (set(self.get_class_count().keys()).difference(set(target_classes)))

    def __str__(self):
        newlines = zip(self.words, self.tags)
        return '\n'.join(['\t'.join(line) for line in newlines])


class FewshotSampler:
    '''
    sample one support set and one query set
    '''

    def __init__(self, N, K, Q, samples, classes=None, random_state=0):
        '''
        N: int, how many types in each set
        K: int, how many instances for each type in support set
        Q: int, how many instances for each type in query set
        samples: List[Sample], Sample class must have `get_class_count` attribute
        classes[Optional]: List[any], all unique classes in samples. If not given, the classes will be got from samples.get_class_count()
        random_state[Optional]: int, the random seed
        '''
        self.K = K
        self.N = N
        self.Q = Q
        self.samples = samples
        self.__check__()  # check if samples have correct types
        if classes:
            self.classes = classes
        else:
            self.classes = self.__get_all_classes__()
        random.seed(random_state)

    def __get_all_classes__(self):
        classes = []
        for sample in self.samples:
            classes += list(sample.get_class_count().keys())
        return list(set(classes))

    def __check__(self):
        for idx, sample in enumerate(self.samples):
            if not hasattr(sample, 'get_class_count'):
                print(
                    '[ERROR] samples in self.samples expected to have `get_class_count` attribute, but self.samples[{idx}] does not')
                raise ValueError

    def __additem__(self, index, set_class):
        class_count = self.samples[index].get_class_count()
        for class_name in class_count:
            if class_name in set_class:
                set_class[class_name] += class_count[class_name]
            else:
                set_class[class_name] = class_count[class_name]

    def __valid_sample__(self, sample, set_class, target_classes):
        threshold = 2 * set_class['k']
        class_count = sample.get_class_count()
        if not class_count:
            return False
        isvalid = False
        for class_name in class_count:
            if class_name not in target_classes:
                return False
            if class_count[class_name] + set_class.get(class_name, 0) > threshold:
                return False
            if set_class.get(class_name, 0) < set_class['k']:
                isvalid = True
        return isvalid

    def __finish__(self, set_class):
        if len(set_class) < self.N+1:
            return False
        for k in set_class:
            if set_class[k] < set_class['k']:
                return False
        return True

    def __get_candidates__(self, target_classes):
        return [idx for idx, sample in enumerate(self.samples) if sample.valid(target_classes)]

    def __next__(self):
        '''
        randomly sample one support set and one query set
        return:
        target_classes: List[any]
        support_idx: List[int], sample index in support set in samples list
        support_idx: List[int], sample index in query set in samples list
        '''
        support_class = {'k': self.K}
        support_idx = []
        query_class = {'k': self.Q}
        query_idx = []
        target_classes = random.sample(self.classes, self.N)
        candidates = self.__get_candidates__(target_classes)
        while not candidates:
            target_classes = random.sample(self.classes, self.N)
            candidates = self.__get_candidates__(target_classes)

        # greedy search for support set
        while not self.__finish__(support_class):
            index = random.choice(candidates)
            if index not in support_idx:
                if self.__valid_sample__(self.samples[index], support_class, target_classes):
                    self.__additem__(index, support_class)
                    support_idx.append(index)
        # same for query set
        while not self.__finish__(query_class):
            index = random.choice(candidates)
            if index not in query_idx and index not in support_idx:
                if self.__valid_sample__(self.samples[index], query_class, target_classes):
                    self.__additem__(index, query_class)
                    query_idx.append(index)
        return target_classes, support_idx, query_idx

    def __iter__(self):
        return self


class FewShotNERDataset(data.Dataset):
    """
    Fewshot NER Dataset
    """

    def __init__(self, filepath, tokenizer, N, K, Q, max_length, Ptuing=True, ignore_label_id=-1, encoder_name='bart',args=None):
        if not os.path.exists(filepath):
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.class2sampleid = {}
        self.N = N
        self.K = K
        self.Q = Q
        self.tokenizer = tokenizer
        self.samples, self.classes = self.__load_data_from_file__(filepath)
        self.max_length = max_length
        self.sampler = FewshotSampler(
            N, K, Q, self.samples, classes=self.classes)
        self.ignore_label_id = ignore_label_id


        self.encoder_name = encoder_name
        self.Ptuing = Ptuing

        if (self.Ptuing):
            P_template_format = args.ner_template
            self.tokenizer.add_special_tokens({'additional_special_tokens': [args.entity_pseudo_token]})
            self.P_template_len = sum(P_template_format)
            self.template_max_len = sum(P_template_format)
            self.entity_pseudo_token_id = self.tokenizer.get_vocab()[args.entity_pseudo_token]
            self.entity_pseudo_token=args.entity_pseudo_token
            self.ner_template = args.ner_template
        else:
            self.template_max_len = 15  # need edit
        assert self.template_max_len < max_length-2

        self.MASK = self.tokenizer.mask_token_id
        if self.encoder_name == 'bert':
            self.SEP = self.tokenizer.convert_tokens_to_ids(['[SEP]'])
            self.CLS = self.tokenizer.convert_tokens_to_ids(['[CLS]'])
            self.PAD = self.tokenizer.convert_tokens_to_ids(['[PAD]'])
        else:
            self.SEP = self.tokenizer.convert_tokens_to_ids(['</s>'])
            self.CLS = self.tokenizer.convert_tokens_to_ids(['<s>'])

    def __get_question(self, question_name, entity):
        if self.Ptuing:
            question_name = question_name.split('-')
            prompt_token=self.entity_pseudo_token
            prompt = [[prompt_token] * self.ner_template[0]
                        + entity    # entity
                        + [prompt_token] * self.ner_template[1]
                        + question_name   # entity type
                        + [prompt_token] * self.ner_template[2]
                        + ["?"]
                    ]
            #raise NotImplementedError
        else:
            relation_id = self.__relation_name2ID__(question_name)
            prompt = self.id2question[str(relation_id)]
            prompt = self.encoder.tokenize_question(
                prompt, entity)
        return prompt

    def _get_entity_list (self,tokens,labels):
        entity_list = []
        entity_label_list = []
        assert len(tokens)==len(labels)
        entity_st_ptr=-1
        entity_ed_ptr=-1
        entity_label=-1
        for i in range(len(tokens)):
            if(labels[i]!=0):
                if(entity_st_ptr==-1):
                    entity_st_ptr=i
                    entity_ed_ptr=i
                    entity_label=labels[i]
                else:
                    entity_ed_ptr+=1
            else:
                if(entity_st_ptr!=-1):
                    entity_list.append([entity_st_ptr,entity_ed_ptr])
                    entity_label_list.append(entity_label)
                    entity_st_ptr=-1
                    entity_ed_ptr=-1
                    entity_label=-1
        if(entity_st_ptr!=-1):
            entity_list.append([entity_st_ptr,entity_ed_ptr])
            entity_label_list.append(entity_label)
        del entity_st_ptr,entity_ed_ptr,entity_label,i

        return entity_list,entity_label_list

    def __mask__raw_token(self,raw,entity):
        masked_tokens=raw.copy()
        lenth=len(entity)
        for i in range(len(masked_tokens)-lenth):
            mask=True
            for j in range(lenth):
                if(entity[j]!=masked_tokens[i+j]):
                    mask=False
                    break
            if(mask):
                for j in range(lenth):
                    masked_tokens[i+j]='<mask>'
        return masked_tokens
                
    def __contact_question__for__bart(self, target_classes, raw_tokens, labels):
        assert 'bart' in self.encoder_name.lower()
        entity_list,entity_label_list=self._get_entity_list(raw_tokens,labels)
        return_dic={}
        encoder_tensor_list=[]
        decoder_tensor_list=[]
        label_tensor_list = []
        span_list = []
        lmlabel_list=[]
        e_label_list = []
        origin_decode_tensor_list =[]
        origin_encode_tensor_list=[]
        mask_loc=[]
        
        for i in range(len(entity_list)):
            probably_entity = raw_tokens[entity_list[i][0]:entity_list[i][1]+1]
            span_label = entity_label_list[i]
            tmp_encode=[]
            tmp_decode=[]
            tmp_lmlabel=[]
            tmp_elabel=[]
            tmp_ori=[]
            tmp_mask_loc=[]
            tmp_ori_en=[]
            index=0
            span_list.append(entity_list[i])
            label_tensor_list.append(span_label)
            for class_name in target_classes:
                question_tokens = self.tokenizer.convert_tokens_to_ids(self.__get_question(
                        class_name, probably_entity)[0])
                
                #!masked_raw_token = self.__mask__raw_token(raw_tokens,probably_entity)
                masked_raw_token=raw_tokens

                encoder_inputs = self.tokenizer.encode(masked_raw_token, add_special_tokens=False, max_length=self.max_length - 3-self.template_max_len) + [self.tokenizer.eos_token_id]
                lm_label = self.tokenizer.encode(raw_tokens, add_special_tokens=False, max_length=self.max_length - 3-self.template_max_len) + [self.tokenizer.eos_token_id]
                ori_encoder_inputs= [self.tokenizer.bos_token_id] + self.tokenizer.encode(raw_tokens, add_special_tokens=False, max_length=self.max_length - 3-self.template_max_len)
                ori_decoder_inputs= [self.tokenizer.bos_token_id] + self.tokenizer.encode(raw_tokens, add_special_tokens=False, max_length=self.max_length - 3-self.template_max_len)+ [self.tokenizer.eos_token_id]
                encoder_inputs = encoder_inputs + question_tokens + [self.MASK] + [self.tokenizer.eos_token_id]
                
                
                if index == span_label:
                    decoder_inputs= ori_decoder_inputs + question_tokens + self.tokenizer.convert_tokens_to_ids(['yes'])+ [self.tokenizer.eos_token_id]
                    tmp_elabel.append(self.tokenizer.convert_tokens_to_ids(['yes']))
                else:
                    decoder_inputs= ori_decoder_inputs + question_tokens + self.tokenizer.convert_tokens_to_ids(['no'])+ [self.tokenizer.eos_token_id]
                    tmp_elabel.append(self.tokenizer.convert_tokens_to_ids(['no']))
                index+=1

                #print(self.tokenizer.convert_tokens_to_ids(['yes']))
                assert len(decoder_inputs)==len(ori_decoder_inputs)+len(question_tokens)+2
                #print(decoder_inputs[len(ori_decoder_inputs)+len(question_tokens)],self.tokenizer.convert_tokens_to_ids(['no']),self.tokenizer.convert_tokens_to_ids(['yes']))
                
                #assert -1==0
                tmp_mask_loc.append(len(ori_decoder_inputs)+len(question_tokens))
                tmp_lmlabel.append(lm_label)
                tmp_encode.append(encoder_inputs)
                tmp_decode.append(decoder_inputs)
                tmp_ori.append(ori_decoder_inputs)
                tmp_ori_en.append(ori_encoder_inputs)
            origin_decode_tensor_list+=(tmp_ori)
            e_label_list+=(tmp_elabel)
            lmlabel_list+=(tmp_lmlabel)
            encoder_tensor_list+=(tmp_encode)
            decoder_tensor_list+=(tmp_decode)
            origin_encode_tensor_list+=tmp_ori_en
            mask_loc+=tmp_mask_loc
        return_dic['encoder_tensor_list']=encoder_tensor_list
        return_dic['decoder_tensor_list']=decoder_tensor_list
        return_dic['span_list']=span_list
        return_dic['label_tensor_list']=label_tensor_list
        return_dic['lmlabel_list']=lmlabel_list
        return_dic['e_label_list']=e_label_list
        return_dic['origin_decode_tensor_list']=origin_decode_tensor_list
        return_dic['origin_encode_tensor_list']=origin_encode_tensor_list
        return_dic['mask_loc']=mask_loc
        return return_dic


    def __contact_question__for__bert(self, target_classes, raw_tokens, labels):
        assert 'bert' in self.encoder_name.lower()
        entity_list,entity_label_list=self._get_entity_list(raw_tokens,labels)
        return_dic={}
        masked_encoder_tensor_list=[]
        encoder_tensor_word = []
        span_list = []
        label_tensor_list = []
        masked_loc_list=[]
        att_mask_list=[]

        for i in range(len(entity_list)):
            probably_entity = raw_tokens[entity_list[i][0]:entity_list[i][1]+1]
            span_label = entity_label_list[i]
            index=0
            span_list.append(entity_list[i])
            label_tensor_list.append(span_label)
            tmp_masked_encode=[]
            tmp_unmasked=[]
            tmp_loc=[]
            tmp_att=[]
            for class_name in target_classes:
                question_tokens = self.tokenizer.convert_tokens_to_ids(self.__get_question(
                        class_name, probably_entity)[0])
                ori_encoder_input= self.CLS + self.tokenizer.encode(raw_tokens, add_special_tokens=False, max_length=self.max_length - 3-self.template_max_len) + self.SEP +question_tokens
                if index == labels[i]:
                    lm_out = ori_encoder_input+self.tokenizer.convert_tokens_to_ids(['yes'])
                else:
                    lm_out = ori_encoder_input+self.tokenizer.convert_tokens_to_ids(['no'])
                
                index+=1
                masked_encoder_input= ori_encoder_input + self.MASK
                att_mask = [1]*len(masked_encoder_input)
                masked_loc = len(ori_encoder_input)
                tmp_masked_encode.append(masked_encoder_input)
                tmp_unmasked.append(lm_out)
                tmp_loc.append(masked_loc)
                tmp_att.append(att_mask)
            att_mask_list+=tmp_att
            masked_encoder_tensor_list+=tmp_masked_encode
            encoder_tensor_word+=tmp_unmasked
            masked_loc_list+=tmp_loc
        
        return_dic['att_mask_list']=att_mask_list
        return_dic['masked_encoder_tensor_list']=masked_encoder_tensor_list
        return_dic['e_label_list']=encoder_tensor_word
        return_dic['masked_loc']=masked_loc_list
        return_dic['span_list']=span_list
        return_dic['span_label']=label_tensor_list
        return return_dic






    # every word of raw_tokens probably is an entity, so if wanna get full template, we'll need to create N*K*(token_length)*n_gram filled
    # teamplate (such a big data)
    # assume it is a classification task 
    def __contact_question__for__bert(self, target_classes, raw_tokens, labels):
        # (n_gram_total_length,target_classes,seq_length)
        word_tensor_list = []
        mask_tensor_list = []
        seg_tensor_list = []
        label_tensor_list = []  # (n_gram_total_length,label_size)
        span_list = []
        
        entity_list,entity_label_list=self._get_entity_list(raw_tokens,labels)
        
        for i in range(len(entity_list)):
                tmp_mask = []
                tmp_word = []  # (target_classes,seq_length)
                tmp_seg = []
                probably_entity = raw_tokens[entity_list[i][0]:entity_list[i][1]+1]  
                span_label = entity_label_list[i]
                
                label_tensor_list.append(span_label)
                span_list.append(entity_list[i])
                
                index=0
                for class_name in target_classes:
                    
                    question_tokens = self.__get_question(
                        class_name, probably_entity)

                    
                    inserted_qustion_tokens = self.CLS + \
                        self.tokenizer.convert_tokens_to_ids(question_tokens) + self.SEP + \
                        self.tokenizer.convert_tokens_to_ids(
                            raw_tokens) + self.SEP
                    
                        
                    
                    word_tensor = torch.ones((self.max_length)).long()
                    sentence_len = min(
                        self.max_length, len(inserted_qustion_tokens))
                    for ii in range(sentence_len):
                        word_tensor[ii] = inserted_qustion_tokens[ii]

                    mask_tensor = torch.zeros((self.max_length)).long()
                    mask_tensor[:min(self.max_length, len(
                        inserted_qustion_tokens))] = 1
                    seg_tensor = torch.ones((self.max_length)).long()
                    seg_tensor[:min(self.max_length, len(
                        question_tokens) + 1)] = 0

                    tmp_word.append(word_tensor.numpy().tolist())
                    tmp_mask.append(mask_tensor.numpy().tolist())
                    tmp_seg.append(seg_tensor.numpy().tolist())
                    index+=1
                    
                word_tensor_list.append(tmp_word)
                mask_tensor_list.append(tmp_mask)
                seg_tensor_list.append(tmp_seg)

        return word_tensor_list, mask_tensor_list, seg_tensor_list, label_tensor_list, span_list

    def __insert_sample__(self, index, sample_classes):
        for item in sample_classes:
            if item in self.class2sampleid:
                self.class2sampleid[item].append(index)
            else:
                self.class2sampleid[item] = [index]

    def __load_data_from_file__(self, filepath):
        samples = []
        classes = []
        with open(filepath, 'r', encoding='utf-8')as f:
            lines = f.readlines()
        samplelines = []
        index = 0
        for line in lines:
            line = line.strip()
            if line:
                samplelines.append(line)
            else:
                sample = Sample(samplelines)
                samples.append(sample)
                sample_classes = sample.get_tag_class()
                self.__insert_sample__(index, sample_classes)
                classes += sample_classes
                samplelines = []
                index += 1
        classes = list(set(classes))
        return samples, classes

    def __get_token_label_list__(self, sample):
        tokens = []
        labels = []
        for word, tag in zip(sample.words, sample.normalized_tags):
            word_tokens = self.tokenizer.tokenize(word)
            if word_tokens:
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                # word_labels = [self.tag2label[tag]] + [self.ignore_label_id] * (len(word_tokens) - 1)
                word_labels = [self.tag2label[tag]] * len(word_tokens)
                labels.extend(word_labels)
        return tokens, labels


    def __getraw__(self, tokens, labels, target_classes):
        # get tokenized word list, attention mask
        tokens_list = []
        origin_labels_list = []

        while len(tokens) > self.max_length - 3-self.template_max_len:
            tokens_list.append(
                tokens[:self.max_length-3-self.template_max_len])
            tokens = tokens[self.max_length-3-self.template_max_len:]
            origin_labels_list.append(
                labels[:self.max_length-3-self.template_max_len])
            labels = labels[self.max_length-3-self.template_max_len:]
        if tokens:
            tokens_list.append(tokens)
            origin_labels_list.append(labels)

        # (token_split_length,n_gram_total_length,target_classes_num,seq_length)
        word_list = []
        mask_list = []
        seg_list = []
        span_list = []
        label_list = []
        raw_list = []

        for i, tokens in enumerate(tokens_list):
            # token -> ids
            # word_tensor_list()
            word_tensor_list, mask_tensor_list, seg_tensor_list, label_tensor_list, span_tensor_list = self.__contact_question__for__bert(
                target_classes, tokens, origin_labels_list[i])
            # print(word_tensor_list,labels_list,'112',span_tensor_list)
            assert(len(span_tensor_list) == len(label_tensor_list))
            raw_list+=(tokens)
            word_list+=(word_tensor_list)
            mask_list+=(mask_tensor_list)
            seg_list+=(seg_tensor_list)
            span_list+=(span_tensor_list)
            label_list+=(label_tensor_list)

        return word_list, mask_list, seg_list, label_list, span_list, raw_list

    def __additem__(self, index, d, word, mask, seg, label, span, raw_text):
        d['index'].append(index)
        d['word'].append(word)
        d['mask'].append(mask)
        d['label'].append(label)
        d['seg'].append(seg)
        d['span'].append(span)
        d['raw_text'].append(raw_text)

    def __populate__(self, idx_list, target_classes, savelabeldic=False):
        '''
        populate samples into data dict
        set savelabeldic=True if you want to save label2tag dict
        'index': sample_index
        'word': tokenized word ids
        'mask': attention mask in BERT
        'label': NER labels
        'seg': 0 for template, 1 for real text
        'raw text':original tokens
        '''
        dataset = {'index': [], 'word': [], 'mask': [], 'raw_text': [],
                   'label': [], 'seg': [], 'span': []}
        
        #contact to matrix to acclerate train_speed
        if "bert" in self.encoder_name:
            word_matrix=[]
            seg_matrix=[]
            mask_matrix=[]
            label_matrix=[]

            for idx in idx_list:
                tokens, labels = self.__get_token_label_list__(self.samples[idx])
                word, mask, seg, label, span, raw = self.__getraw__(
                    tokens, labels, target_classes)
                word_matrix+=(word)
                seg_matrix+=(seg)
                mask_matrix+=(mask)
                label_matrix+=label
                self.__additem__(idx, dataset, word, mask, seg, label, span, raw)


            dataset['word_mat']=torch.tensor(word_matrix).long()
            dataset['seg_mat']=torch.tensor(seg_matrix).long()
            dataset['mask_mat']=torch.tensor(mask_matrix).long()
            dataset['label_mat']=torch.tensor(label_matrix).long()
            dataset['label2tag'] = [self.label2tag]
            return dataset

        elif "bart" in self.encoder_name:
            return_dic={}
            for idx in idx_list:
                tokens, labels = self.__get_token_label_list__(self.samples[idx])
                sample=self.__contact_question__for__bart(target_classes, tokens, labels)
                for key in sample:
                    if(key not in return_dic):return_dic[key]=[]
                    return_dic[key]+=sample[key]
            return return_dic



    def __getitem__(self, index):
        target_classes, support_idx, query_idx = self.sampler.__next__()
        # add 'O' and make sure 'O' is labeled 0
        distinct_tags = ['O'] + target_classes
        self.tag2label = {tag: idx for idx, tag in enumerate(distinct_tags)}
        self.label2tag = {idx: tag for idx, tag in enumerate(distinct_tags)}
        support_set = self.__populate__(support_idx, target_classes)
        query_set = self.__populate__(
            query_idx, target_classes)
        return support_set, query_set

    def __len__(self):
        return 1000000000

def batch_convert_ids_to_tensors(batch_token_ids: List[List],pad_id=BartTokenizer.from_pretrained('facebook/bart-base').pad_token_id) -> torch.Tensor:
    bz = len(batch_token_ids)
    batch_tensors = [torch.LongTensor(batch_token_ids[i]).squeeze(0) for i in range(bz)]
    batch_tensors = pad_sequence(batch_tensors, True, padding_value=pad_id).long()
    return batch_tensors



def data_collator(data):
    batch_support={'encoder_tensors':[],'decoder_tensors':[],'origin_decode_tensors':[],"lm_labels":[],"span_list":[],'label':[],'prompt_labels':[]}
    batch_query={'encoder_tensors':[],'decoder_tensors':[],'origin_decode_tensors':[],"lm_labels":[],"span_list":[],'label':[],'prompt_labels':[]}
    support_set,query_set = zip(*data)
    assert (len(support_set)==len(query_set))
    for i in range(len(support_set)):
        features=support_set[i]

    
        assert len(features["encoder_tensor_list"])==len(features["decoder_tensor_list"])
        assert len(features["encoder_tensor_list"])==len(features["lmlabel_list"])
        

        #pad
        batch_support['encoder_tensors']+= features["encoder_tensor_list"]
        batch_support['decoder_tensors']+= features["decoder_tensor_list"]
        batch_support['origin_decode_tensors']+= features["origin_decode_tensor_list"]
        batch_support['lm_labels']+=features["lmlabel_list"]

        batch_support['span_list']+=torch.tensor([features['span_list']])
        batch_support['label']+=torch.LongTensor([features["label_tensor_list"]])
        batch_support['mask_loc']=torch.tensor([features['mask_loc']])
        batch_support['prompt_labels'] = torch.LongTensor([features["e_label_list"]])
        
    for i in range(len(query_set)):
        features=query_set[i]
        batch_query['encoder_tensors']+= features["encoder_tensor_list"]
        batch_query['decoder_tensors']+= features["decoder_tensor_list"]
        batch_query['origin_decode_tensors']+= features["origin_decode_tensor_list"]
        batch_query['lm_labels']+=features["lmlabel_list"]

        batch_query['span_list']+=torch.tensor([features['span_list']])
        batch_query['label']+=torch.LongTensor([features["label_tensor_list"]])
        
        batch_query['prompt_labels'] = torch.LongTensor([features["e_label_list"]])
        batch_query['mask_loc']=torch.tensor([features['mask_loc']])

    padding_dic = ['encoder_tensors','decoder_tensors','origin_decode_tensors','lm_labels']
    for i in padding_dic:
        batch_support[i]=batch_convert_ids_to_tensors(batch_support[i])
        batch_query[i]=batch_convert_ids_to_tensors(batch_query[i])
    
    lens=torch.tensor(batch_support['encoder_tensors']).shape[-1]
    return batch_support,batch_query
    






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
    parser.add_argument("--entity_pseudo_token", type=str, default='[PROMPT_E]')
    
    parser.add_argument("--template", type=str, default="(3, 3, 3, 3)")
    parser.add_argument("--ner_template", type=str, default="(4, 4, 4)")
    
    # contractive learning
    parser.add_argument("--contrasive", type=bool, default=True)

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
    args.ner_template = eval(args.ner_template) if type(args.ner_template) is not tuple else args.ner_template

    assert type(args.template) is tuple
    assert type(args.ner_template) is tuple
    return args



if __name__=="__main__":
    # n_gram dataset
    N=5
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    dataset = FewShotNERDataset(
        './data/few_ner/intra/dev.txt', tokenizer, N, 1, 1, 122,args=get_args())
    loader = iter(torch.utils.data.DataLoader(dataset,batch_size=2,collate_fn=data_collator))
    print(next(loader)[0])
    for i in range(10000):
        t = dataset[i][0]
        print(t)
    #print(t['seg'].shape)
