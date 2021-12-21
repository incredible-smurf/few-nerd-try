from json import decoder
from urllib.parse import quote_from_bytes
import torch.utils.data as data
import torch
import random
import os
import numpy as np
from transformers import BertTokenizer, BartTokenizer, file_utils
from torch.nn.utils.rnn import pad_sequence
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import pickle
from transformers.utils.dummy_pt_objects import NoRepeatNGramLogitsProcessor
from data_process.data_utils import expand_tag_words
#from data_utils import expand_tag_words

# entity classification


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


class FewShotConll03Dataset(data.Dataset):
    """
    Fewshot NER Dataset
    """
    def __init__(self, filepath, tokenizer, N, K, Q, max_length, ignore_label_id=-1, encoder_name='bart', args=None):
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

        self.use_bart_augment = False

        self.encoder_name = encoder_name
        self.Ptuing = args.p_tuning
        self.args=args

        if (self.Ptuing):
            P_template_format = args.template
            self.tokenizer.add_special_tokens({'additional_special_tokens': [
                                              args.entity_pseudo_token, args.entity_split_token]})
            self.P_template_len = P_template_format
            self.template_max_len = P_template_format
            self.decoder_template = args.decoder_template
            self.entity_pseudo_token_id = self.tokenizer.get_vocab()[
                args.entity_pseudo_token]
            self.entity_split_token_id = self.tokenizer.get_vocab()[
                args.entity_split_token]
            self.entity_split_token = args.entity_split_token
            self.entity_pseudo_token = args.entity_pseudo_token
            self.ner_template = args.template
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

    def _get_encoder_ptuning_template(self, template_lenth):
        if self.Ptuing:
            prompt_token = self.entity_pseudo_token
            prompt = [prompt_token] * template_lenth

            #raise NotImplementedError

        return prompt

    def _get_entity_list(self, tokens, labels):
        entity_list = []
        entity_label_list = []
        assert len(tokens) == len(labels)
        entity_st_ptr = -1
        entity_ed_ptr = -1
        entity_label = -1
        for i in range(len(tokens)):
            if(labels[i] != 0):
                if(entity_st_ptr == -1):
                    entity_st_ptr = i
                    entity_ed_ptr = i
                    entity_label = labels[i]
                else:
                    entity_ed_ptr += 1
            else:
                if(entity_st_ptr != -1):
                    entity_list.append([entity_st_ptr, entity_ed_ptr])
                    entity_label_list.append(entity_label)
                    entity_st_ptr = -1
                    entity_ed_ptr = -1
                    entity_label = -1
        if(entity_st_ptr != -1):
            entity_list.append([entity_st_ptr, entity_ed_ptr])
            entity_label_list.append(entity_label)
        del entity_st_ptr, entity_ed_ptr, entity_label, i

        return entity_list, entity_label_list

    # decoder sentence =[bos_id,entity_1,[prompt_token],entity_type,[entity_spilt_token],entity2,...,[eos_token]]
    def __get_decoder_output_sentence_id(self, target_classes, entity_list, entity_label_list, raw_tokens):
        decoder_input_sentence_id = [self.tokenizer.bos_token_id]
        for i in range(len(entity_list)):
            for j in range(entity_list[i][0], entity_list[i][1]):
                decoder_input_sentence_id += self.tokenizer.convert_tokens_to_ids([
                                                                                  raw_tokens[j]])

            decoder_input_sentence_id += [
                self.entity_pseudo_token_id]*self.decoder_template

            tt = 'location-road/railway/highway/transit'
            entity_type = target_classes[entity_label_list[i]-1].split('/')[0]
            entity_type = entity_type.split('-')

            decoder_input_sentence_id += self.tokenizer.convert_tokens_to_ids(
                entity_type)

            if(i != len(entity_list)-1):
                decoder_input_sentence_id += self.entity_split_token_id
        decoder_input_sentence_id += [self.tokenizer.eos_token_id]
        return decoder_input_sentence_id

    def get_tokenizer_label_id(self, label2tag):

        vocab = self.tokenizer.get_vocab()
        label_id = []

        for i in label2tag:
            entity_type_name = label2tag[i]
            # insert [entity_type] into vocab
            special_token = '['+entity_type_name+']'
            if special_token not in vocab:
                self.tokenizer.add_tokens(special_token)
                vocab = self.tokenizer.get_vocab()
            label_id.append(vocab[special_token])

        assert len(label2tag) == len(label_id)
        return label_id

    def __contact_template_for_bart(self, target_classes, raw_tokens, labels):
        raw_tokens, labels=expand_tag_words(self.tokenizer,raw_tokens, labels)
        raw_tokens = raw_tokens[:self.args.max_length-3]
        labels =labels [:self.args.max_length-3]
        assert len(raw_tokens) == len(labels)
        assert 'bart' in self.encoder_name.lower()
        entity_list, entity_label_list = self._get_entity_list(
            raw_tokens, labels)
        return_dic = {}
        if self.Ptuing:
            prompt = self._get_encoder_ptuning_template(self.P_template_len)
        ori_input_token_id = self.tokenizer.convert_tokens_to_ids(raw_tokens)

        if self.use_bart_augment:
            pass

        if self.Ptuing:
            inserted_input_tokens = self.tokenizer.convert_tokens_to_ids(
                prompt + raw_tokens)
            return_encoder_sentence_id = [
                self.tokenizer.bos_token_id] + inserted_input_tokens + [self.tokenizer.eos_token_id]

        else:
            return_encoder_sentence_id = [
                self.tokenizer.bos_token_id] + ori_input_token_id + [self.tokenizer.eos_token_id]

        #decoder_input_sentence_id = self.__get_decoder_output_sentence_id(target_classes,entity_list,entity_label_list,raw_tokens)

        self.target_shift = self.N + 4  # +4 <eos> & <bos>& <pad> & O tag
        if self.Ptuing:
            self.target_shift += len(prompt)

        target_span = []
        decoder_input_sentence_id = [self.tokenizer.bos_token_id]

        for i in range(len(entity_list)):
            assert len(entity_list[i])==2

            entity_list[i][0]+=1 # bos 
            entity_list[i][1]+=1

            target_span .append(entity_list[i])
            target_span[i].append(entity_label_list[i])
            decoder_input_sentence_id.append(
                entity_list[i][0]+self.target_shift)  # start
            decoder_input_sentence_id.append(
                entity_list[i][1]+self.target_shift)  # end
            decoder_input_sentence_id.append(
                entity_label_list[i]+3)  # entity_type

        decoder_input_sentence_id += [self.tokenizer.eos_token_id]  # <eos>

        return_dic['span'] = target_span
        return_dic['origin_input_id'] = ori_input_token_id
        return_dic['encoder_input_id'] = return_encoder_sentence_id
        return_dic['encoder_input_length'] = len(return_encoder_sentence_id)
        return_dic['decoder_input_id'] = decoder_input_sentence_id
        return_dic['decoder_input_length'] = len(decoder_input_sentence_id)

        
        return_dic['raw_tokens']=raw_tokens
        return_dic['raw_labels']=labels
        return return_dic

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

        """ while len(tokens) > self.max_length - 3-self.template_max_len:
            tokens_list.append(
                tokens[:self.max_length-3-self.template_max_len])
            tokens = tokens[self.max_length-3-self.template_max_len:]
            origin_labels_list.append(
                labels[:self.max_length-3-self.template_max_len])
            labels = labels[self.max_length-3-self.template_max_len:]
        if tokens:
            tokens_list.append(tokens)
            origin_labels_list.append(labels) """

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
            word_tensor_list, mask_tensor_list, seg_tensor_list, label_tensor_list, span_tensor_list = self.__contact_template__for__bart(
                target_classes, tokens, origin_labels_list[i])
            # print(word_tensor_list,labels_list,'112',span_tensor_list)
            assert(len(span_tensor_list) == len(label_tensor_list))
            raw_list += (tokens)
            word_list += (word_tensor_list)
            mask_list += (mask_tensor_list)
            seg_list += (seg_tensor_list)
            span_list += (span_tensor_list)
            label_list += (label_tensor_list)

        return word_list, mask_list, seg_list, label_list, span_list, raw_list

    def __additem__(self, index, d, word, mask, seg, label, span, raw_text):
        d['index'].append(index)
        d['word'].append(word)
        d['mask'].append(mask)
        d['label'].append(label)
        d['seg'].append(seg)
        d['span'].append(span)
        d['raw_text'].append(raw_text)

    def __populate__(self, idx_list, target_classes):
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

        # contact to matrix to acclerate train_speed
        if "bert" in self.encoder_name:
            word_matrix = []
            seg_matrix = []
            mask_matrix = []
            label_matrix = []

            for idx in idx_list:
                tokens, labels = self.__get_token_label_list__(
                    self.samples[idx])
                word, mask, seg, label, span, raw = self.__getraw__(
                    tokens, labels, target_classes)
                word_matrix += (word)
                seg_matrix += (seg)
                mask_matrix += (mask)
                label_matrix += label
                self.__additem__(idx, dataset, word, mask,
                                 seg, label, span, raw)

            dataset['word_mat'] = torch.tensor(word_matrix).long()
            dataset['seg_mat'] = torch.tensor(seg_matrix).long()
            dataset['mask_mat'] = torch.tensor(mask_matrix).long()
            dataset['label_mat'] = torch.tensor(label_matrix).long()
            dataset['label2tag'] = [self.label2tag]
            return dataset

        elif "bart" in self.encoder_name:
            return_dic = {}
            for idx in idx_list:
                tokens, labels = self.__get_token_label_list__(
                    self.samples[idx])
                sample = self.__contact_template_for_bart(
                    target_classes, tokens, labels)

                for key in sample:
                    if(key not in return_dic):
                        return_dic[key] = []
                    return_dic[key].append(sample[key])
            
            #pad
            return_dic['encoder_input_id'] =batch_convert_ids_to_tensors (return_dic['encoder_input_id'],self.tokenizer.pad_token_id)
            return_dic['decoder_input_id'] =batch_convert_ids_to_tensors (return_dic['decoder_input_id'],self.tokenizer.pad_token_id)
            return_dic['encoder_input_length']=torch.tensor(return_dic['encoder_input_length'])
            return_dic['decoder_input_length']=torch.tensor(return_dic['decoder_input_length'])
            return_dic['label2tag'] = [self.label2tag]# don' t forget to +3 then
            return_dic['tag2label'] = [self.tag2label]
            return_dic['label_vocab_id'] = self.get_tokenizer_label_id(
                    self.label2tag)
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


def batch_convert_ids_to_tensors(batch_token_ids: List[List], pad_id=BartTokenizer.from_pretrained('facebook/bart-base').pad_token_id) -> torch.Tensor:
    bz = len(batch_token_ids)
    batch_tensors = [torch.LongTensor(
        batch_token_ids[i]) for i in range(bz)]
    batch_tensors = pad_sequence(
        batch_tensors, True, padding_value=pad_id).long()
    return batch_tensors


def bart_data_collator(data):
    batch_support = {}
    batch_query = {}
    support_set, query_set = zip(*data)
    assert (len(support_set) == len(query_set))
    pad_id = BartTokenizer.from_pretrained('facebook/bart-base').pad_token_id

    for i in range(len(support_set)):
        one_support = support_set[i]
        for key in one_support:
            if key not in batch_support:
                batch_support[key] = []
        for key in one_support:
            batch_support[key].append(one_support[key])

    del one_support
    for i in range(len(query_set)):
        one_query = query_set[i]
        for key in one_query:
            if key not in batch_support:
                batch_support[key] = []
        for key in one_query:
            batch_support[key].append(one_query[key])

    batch_support['encoder_input_id'] = batch_convert_ids_to_tensors(
        batch_support['encoder_input_id'], 
        pad_id)#bart pad
    batch_support['query_input_id'] = batch_convert_ids_to_tensors(
        batch_support['query_input_id'], 
        pad_id)#bart pad
    batch_query['encoder_input_id'] = batch_convert_ids_to_tensors(
        batch_support['encoder_input_id'], 
        pad_id)#bart pad
    batch_query['query_input_id'] = batch_convert_ids_to_tensors(
        batch_support['query_input_id'], 
        pad_id)#bart pad

    for i in batch_support:
        try:
            batch_support[i] =torch.tensor(batch_support[i])
            batch_query[i] = torch.tensor(batch_query[i],pad_id=pad_id)
        # do nothing
        except Exception:
            tmp = None

    return batch_support, batch_query

def bart_pretrain_data_collator(data):
    batch_data = {}
    data_dic = data
    pad_id = BartTokenizer.from_pretrained('facebook/bart-base').pad_token_id

    for i in range(len(data_dic)):
        piece_data = data_dic[i]
        for key in piece_data:
            if key not in batch_data:
                batch_data[key] = []
        for key in piece_data:
            batch_data[key].append(piece_data[key])


    batch_data['encoder_input_id'] = batch_convert_ids_to_tensors(
        batch_data['encoder_input_id'], 
        pad_id)#bart pad



    for i in batch_data:
        try:
            batch_data[i] =torch.tensor(batch_data[i])
        # do nothing
        except Exception:
            tmp = None

    return batch_data

class NormalConll03Dataset(data.Dataset):
    def __init__(self,file_path,tokenizer,args=None) -> None:
        super().__init__()
        self.path = file_path
        self.__load_data_from_file__(self.path,cache_path=self.path+'.cache')
        self.class2sampleid = {}
        self.tokenizer = tokenizer
        self.args=args
        self.use_bart_augment=False
        self.Ptuing = args.p_tuning
        self.N = 4 #Conll03 has four entity type
        self.label2id={'O':0,'MISC':1,'LOC':2,'PER':3,'ORG':4}
    
    
    def __load_data_from_file__(self, filepath,cache_path=None):
        if cache_path!=None and os.path.exists(cache_path):
            with open(cache_path,'rb') as f:
                self.samples = pickle.load(f)
        else :
            samples = []
            with open(filepath, 'r', encoding='utf-8')as f:
                lines = f.readlines()
            samplelines = []
            for line in lines:
                line = line.strip()
                if line:
                    samplelines.append(line)
                else:
                    sample = Sample(samplelines)
                    samples.append(sample)
            self.samples = samples

            if cache_path!=None:
                with open(cache_path ,'wb') as f:
                    pickle.dump(samples, f)


    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,index):
        sample = self.samples[index]
        tokens = sample.words
        tag = sample.tags
        return_dic = self.__contact_template_for_bart(tokens,tag)
        return return_dic

    def __contact_template_for_bart(self, raw_tokens, labels):
        raw_tokens, labels=expand_tag_words(self.tokenizer,raw_tokens, labels)
        if self.args!=None:
            raw_tokens = raw_tokens[:self.args.max_length-3]
            labels =labels [:self.args.max_length-3]
        assert len(raw_tokens) == len(labels)
        labels = [self.label2id[i] for i in labels]

        entity_list, entity_label_list = self._get_entity_list(
            raw_tokens, labels)
        return_dic = {}
        if self.Ptuing:
            prompt = self._get_encoder_ptuning_template(self.P_template_len)
  
        ori_input_token_id = self.tokenizer.convert_tokens_to_ids(raw_tokens)


        if self.use_bart_augment:
            pass

        if self.Ptuing:
            inserted_input_tokens = self.tokenizer.convert_tokens_to_ids(
                prompt + raw_tokens)
            return_encoder_sentence_id = [
                self.tokenizer.bos_token_id] + inserted_input_tokens + [self.tokenizer.eos_token_id]

        else:
            return_encoder_sentence_id = [
                self.tokenizer.bos_token_id] + ori_input_token_id + [self.tokenizer.eos_token_id]

        #decoder_input_sentence_id = self.__get_decoder_output_sentence_id(target_classes,entity_list,entity_label_list,raw_tokens)

        self.target_shift = self.N + 4  # +4 <eos> & <bos>& <pad> & O tag
        if self.Ptuing:
            self.target_shift += len(prompt)

        target_span = []
        decoder_input_sentence_id = [self.tokenizer.bos_token_id]

        for i in range(len(entity_list)):
            assert len(entity_list[i])==2

            entity_list[i][0]+=1 # bos 
            entity_list[i][1]+=1

            target_span .append(entity_list[i])
            target_span[i].append(entity_label_list[i])
            decoder_input_sentence_id.append(
                entity_list[i][0]+self.target_shift)  # start
            decoder_input_sentence_id.append(
                entity_list[i][1]+self.target_shift)  # end
            decoder_input_sentence_id.append(
                entity_label_list[i]+3)  # entity_type

        decoder_input_sentence_id += [self.tokenizer.eos_token_id]  # <eos>

        return_dic['span'] = target_span
        return_dic['origin_input_id'] = ori_input_token_id
        return_dic['encoder_input_id'] = return_encoder_sentence_id
        return_dic['encoder_input_length'] = len(return_encoder_sentence_id)
        return_dic['decoder_input_id'] = decoder_input_sentence_id
        return_dic['decoder_input_length'] = len(decoder_input_sentence_id)

        
        return_dic['raw_tokens']=raw_tokens
        return_dic['raw_labels']=labels
        return return_dic

    def _get_entity_list(self, tokens, labels):
        entity_list = []
        entity_label_list = []
        assert len(tokens) == len(labels)
        entity_st_ptr = -1
        entity_ed_ptr = -1
        entity_label = -1
        for i in range(len(tokens)):
            if(labels[i] != 0):
                if(entity_st_ptr == -1):
                    entity_st_ptr = i
                    entity_ed_ptr = i
                    entity_label = labels[i]
                else:
                    entity_ed_ptr += 1
            else:
                if(entity_st_ptr != -1):
                    entity_list.append([entity_st_ptr, entity_ed_ptr])
                    entity_label_list.append(entity_label)
                    entity_st_ptr = -1
                    entity_ed_ptr = -1
                    entity_label = -1
        if(entity_st_ptr != -1):
            entity_list.append([entity_st_ptr, entity_ed_ptr])
            entity_label_list.append(entity_label)
        del entity_st_ptr, entity_ed_ptr, entity_label, i

        return entity_list, entity_label_list

def get_args():
    import argparse
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
    parser.add_argument("--need_pretrian", default=False, type=bool, 
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
    parser.add_argument("--device", default='cuda:0', type=str, 
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


if __name__ == "__main__":
    # n_gram dataset
    N = 5
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    dataset = NormalConll03Dataset(
        './data/conll03/test.txt',tokenizer=tokenizer,args = get_args())
    loader = data.DataLoader(dataset,batch_size=2,collate_fn=bart_pretrain_data_collator)
    print(dataset[0])
    print(len(dataset))
    print(next(iter(loader)))
    # print(t['seg'].shape)
