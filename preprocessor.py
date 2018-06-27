import random
import os 

import pandas as pd 
import numpy as np
np.random.seed(896)

import torch
from torch.autograd import Variable
import torchtext
from torchtext.data import Field, ReversibleField, TabularDataset
from torchtext import vocab

import sklearn
from sklearn.feature_extraction.text import CountVectorizer

import ebm_nlp_demo as e

USE_CUDA = False 

def _to_torch_var(x):
    var_x = Variable(torch.LongTensor(x))
    if USE_CUDA:
        var_x = var_x.cuda()
    return var_x


def print_spans(tokens, y):
    inside = False
    cur_str = []
    for t_i, y_i in zip(tokens, y):
        if y_i == 1:
            inside = True 
            cur_str.append(t_i)
        else:
            if inside: 
                print (" ".join(cur_str))
                print ("\n\n")
            inside = False 



def load_test_data(anno_type='starting_spans', label_set='participants'):
    worker_map, doc_map = e.read_anns(anno_type, label_set, ann_type='aggregated', model_phase='test/gold')
    return maps_to_dicts(doc_map)

def maps_to_dicts(doc_map):
    docs, ys = {}, {}
    for pmid, doc in doc_map.items():
        ys[pmid] = list(doc.anns['AGGREGATED'])
        docs[pmid] = [t.lower() for t in doc.tokens]
    return docs, ys 

def dicts_to_X_y(docs, ys, v):
    X, y = [], []
    for id_ in docs:
        X.append(v.string_to_seq(docs[id_]))
        y.append(_to_torch_var(ys[id_]))
    return X, y

def get_vectorizer(anno_type='starting_spans', label_set='participants', return_data_too=False):
    
    worker_map, doc_map = e.read_anns(anno_type, label_set, \
                                        ann_type = 'aggregated', model_phase = 'train')

    docs, ys = maps_to_dicts(doc_map)

    # for purposes of vectorization, put all text 
    all_text = sum(docs.values(), [])
    

    vectorizer = CountVectorizer(max_features=20000)
    vectorizer.fit(all_text)
    tokenizer = vectorizer.build_tokenizer() 

    str_to_idx = vectorizer.vocabulary_
    str_to_idx["<pad>"] = max(vectorizer.vocabulary_.values())
    str_to_idx["<unk>"] = str_to_idx["<pad>"]+1
    
    if return_data_too:
        return SimpleVectorizer(str_to_idx, tokenizer), (docs, ys)

    return SimpleVectorizer(str_to_idx, tokenizer)



class SimpleVectorizer:

    def __init__(self, str_to_idx, tokenizer):
        self.str_to_idx = str_to_idx
        self.idx_to_str = [None]*(len(self.str_to_idx))
        
        for w, idx in self.str_to_idx.items():
            try:
                self.idx_to_str[idx] = w 
            except:
                import pdb; pdb.set_trace()

        self.tokenizer = tokenizer

    def string_to_seq(self, s, as_torch_var=True):
        if type(s) != type(list()):
            # if it's a string / not already tokenized, 
            # tokenize it.
            tokenized = self.tokenizer(s)
        else:
            tokenized = s 

        vectorized = []
        for token in tokenized:
            idx = self.str_to_idx["<unk>"]
            try:
                idx = self.str_to_idx[token]
            except:
                pass 
            vectorized.append(idx)

        if as_torch_var:
            return _to_torch_var(np.array(vectorized))

        return np.array(vectorized)


    def vectorize(self, tokens):
        tokens = [t.lower() for t in tokens]
        vectorized_article = self.string_to_seq(tokens)

        return vectorized_article

    def decode(self, v):
        return [self.idx_to_str[idx] for idx in v]
