from itertools import count
import random
import csv
import re
import os.path as osp
import torch
import numpy as np
from collections import Counter

def _infer_vector_dim(vector_file_path):
    base_path = osp.basename(vector_file_path)
    res = re.search(r'.([0-9]+)d.', base_path)
    if res is not None:
        embed_size = int(res.group(1))
    else:
        with open(vector_file_path, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
            for row in reader:
                embed_size = len(row[1:])
                break
    return embed_size



class TokenizerRandom():
    def __init__(self, embed_size, corpus=None, min_count=1, max_length=None, extra_token=None, return_tensors=None):
        super().__init__()
        self.ext_token = extra_token
        self.max_length = max_length
        self.embed_size = embed_size
        self.return_tensors = return_tensors
        self.min_count = min_count
        self.corpus = corpus #"list of str"
        self._init_special_token()
        self._init_vector()

    def _init_special_token(self):
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.sep_token = '[SEP]'
        self.pad_token_id = 2
        self.unk_token_id = 1
        self.sep_token_id = 0
        self.special_token = [self.pad_token, self.unk_token, self.sep_token]
        self.special_token_id = [0, 1, 2]

    def _init_vector(self):
        #通过corpus中的词汇随机生成embed向量
        assert hasattr(self.corpus, '__len__') and isinstance(self.corpus[0], str)
        token_freq = {}
        corpus_list = []
        for sentence in self.corpus:
            tokens = sentence.strip().split()
            corpus_list.extend(tokens)
        
        counter = Counter(corpus_list)
        filtered_counter = filter(lambda it:it[1] >= self.min_count, counter.items())
        sorted_counter = sorted(filtered_counter, reverse=True, key=lambda item:item[1])
        self.token_freq_dict = dict(sorted_counter)
        tokens = [tuple[0] for tuple in sorted_counter]
        if self.ext_token is not None:
            for ext in self.ext_token:
                assert ext not in tokens, print('extra token is already in pretrained tokens')
                if ext not in tokens:
                    tokens.insert(0, ext)

        for spe in self.special_token:
            assert spe not in tokens, print('special token is already in pretrained tokens')
            tokens.insert(0, spe)
        
        self.token_to_id, self.id_to_token = dict(), dict()
        for id, token in enumerate(tokens):
            self.token_to_id[token] = id
            self.id_to_token[id] = token
        
        vocab_size = len(tokens)
        vectors = np.random.uniform(-1, 1, [vocab_size, self.embed_size]).tolist()
        vectors[self.pad_token_id] = [0.0 for i in range(self.embed_size)]
        self.vectors = np.array(vectors, dtype=float)

    def convert_tokens_to_ids(self, tokens, max_length):
        #tokens: "list of token(str)"
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            tokens = tokens + [self.pad_token for i in range(max_length - len(tokens))]
        ids = [self.token_to_id.get(token, self.unk_token_id) for token in tokens]
        return ids
    
    


    def encode_plus(self, sentences):
        #sentences: "list of sentences(str)" or "sentences(str)"
        if not isinstance(sentences, (list, tuple)):
            sentences = [sentences]
        batch_size = len(sentences)
        sentences = [s.lower().strip().split() for s in sentences]
        batch_max_length = max(len(s) for s in sentences)
        if self.max_length is not None:
            if self.max_length < batch_max_length:
                batch_max_length = self.max_length
        
        mask_1d = [len(s) if len(s) <= batch_max_length else batch_max_length for s in sentences]
        mask_2d = []
        for mask in mask_1d:
            mask_2d.append([1 if i <= mask else 0 for i in range(1, batch_max_length+1)])
        
        token_ids = [
            self.convert_tokens_to_ids(s, batch_max_length) for s in sentences
        ]

        if self.return_tensors == 'pt':
            token_ids, mask_2d = torch.LongTensor(token_ids), torch.LongTensor(mask_2d)
        elif self.return_tensors == 'np':
            token_ids, mask_2d = np.array(token_ids, dtype=int), np.array(mask_2d, dtype=int)

        return {
            'input_ids' : token_ids, 
            'mask' : mask_2d
        }
    
    def __call__(self, sentences):
        return self.encode_plus(sentences)
    
    def decode(self, token_ids):
        #ids: "list of list of int" or "list of int"
        if not hasattr(token_ids[0], '__len__'):
            token_ids = [token_ids]
        sentences = []
        for ids in token_ids:
            sentences.append(
                self.convert_ids_to_tokens(ids)
            )
        return sentences

    def convert_ids_to_tokens(self, token_ids):
        #token_ids: "list of int"
        sent = ' '.join([self.id_to_token.get(id, '') for id in token_ids if id not in self.special_token_id])
        return sent
    


class TokenizerFromGlove():
    def __init__(self, vector_file_path, max_length=None, extra_token=None, return_tensors=None):
        super().__init__()
        self.ext_token = extra_token
        self.max_length = max_length
        self.embed_size = _infer_vector_dim(vector_file_path)
        self.return_tensors = return_tensors
        self._init_special_token()
        self._load_vector(vector_file_path)
        
    def _init_special_token(self):
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.sep_token = '[SEP]'
        self.pad_token_id = 2
        self.unk_token_id = 1
        self.sep_token_id = 0
        self.special_token = [self.pad_token, self.unk_token, self.sep_token]
        self.special_token_id = [0, 1, 2]

    def _load_vector(self, vec_file):
        with open(vec_file, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
            tokens, vectors = [], []
            for row in reader:
                tokens.append(row[0])
                vectors.append(row[1:])
        if self.ext_token is not None:
            for ext in self.ext_token:
                assert ext not in tokens, print('extra token is already in pretrained tokens')
                if ext not in tokens:
                    tokens.insert(0, ext)
                    vectors.insert(0,
                        [random.uniform(-1, 1) for i in range(self.embed_size)]
                    )

        for spe in self.special_token:
            assert spe not in tokens, print('special token is already in pretrained tokens')
            tokens.insert(0, spe)
            if spe == self.pad_token:
                vectors.insert(0,
                    [0.0 for i in range(self.embed_size)]
                )
            else:
                vectors.insert(0,
                    [random.uniform(-1, 1) for i in range(self.embed_size)]
                )
            
        self.id_to_token, self.token_to_id = {}, {}
        
        for id, token in enumerate(tokens):
            self.id_to_token[id] = token
            self.token_to_id[token] = id

        self.tokens = tokens
        self.vectors = np.array(vectors, dtype=float)
    
    def convert_tokens_to_ids(self, tokens, max_length):
        #tokens: list of token(str)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            tokens = tokens + [self.pad_token for i in range(max_length - len(tokens))]
        ids = [self.token_to_id.get(token, self.unk_token_id) for token in tokens]
        return ids
        
    def encode_plus(self, sentences):
        #sentences: "list of sentences(str)" or "sentence(str)"
        if not isinstance(sentences, (list, tuple)):
            sentences = [sentences]
        batch_size = len(sentences)
        
        sentences = [s.lower().strip().split() for s in sentences]
        batch_max_length = max(len(s) for s in sentences)
        if self.max_length is not None:
            if self.max_length < batch_max_length:
                batch_max_length = self.max_length
        
        mask_1d = [len(s) if len(s) <= batch_max_length else batch_max_length for s in sentences]
        mask_2d = []
        for mask in mask_1d:
            mask_2d.append([1 if i <= mask else 0 for i in range(1, batch_max_length+1)])
        
        token_ids = [
            self.convert_tokens_to_ids(s, batch_max_length) for s in sentences
        ]

        if self.return_tensors == 'pt':
            token_ids, mask_2d = torch.LongTensor(token_ids), torch.LongTensor(mask_2d)
        elif self.return_tensors == 'np':
            token_ids, mask_2d = np.array(token_ids, dtype=int), np.array(mask_2d, dtype=int)

        return {
            'input_ids' : token_ids, 
            'mask' : mask_2d
        }
    
    def __call__(self, sentences):
        return self.encode_plus(sentences)
    
    def decode(self, token_ids):
        #ids: "list of list of int" or "list of int"
        if not hasattr(token_ids[0], '__len__'):
            token_ids = [token_ids]
        sentences = []
        for ids in token_ids:
            sentences.append(
                self.convert_ids_to_tokens(ids)
            )
        return sentences

    def convert_ids_to_tokens(self, token_ids):
        #token_ids: "list of int"
        sent = ' '.join([self.id_to_token.get(id, '') for id in token_ids if id not in self.special_token_id])
        return sent
        


        
        



    
