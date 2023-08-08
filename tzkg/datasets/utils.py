import torch
from torch.utils.data import Dataset
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from utils import load_pkl


def _padding(x, max_len, pad=0):
    return x + [0] * (max_len - len(x))

def _masking(x, max_len, pad=[1, 0]):
     return [pad[0]] * len(x) + [pad[-1]] * (max_len - len(x))


def collate_fn(batch, word2id=None, label2id=None, model="ner"):
    """
    对当前batch进行padding处理, 然后区分x, y;
    Arg : 
        batch () : 数据集
    Returna : 
        x (dict) : key为词, value为长度
        y (List) : 关系对应值的集合
    """
    batch.sort(key=lambda data: data['seq_len'], reverse=True)
    max_len = 512

    if model == "re":
        x, y = dict(), []
        word, word_len = [], []
        for data in batch:
            word.append(_padding(data['token2idx'], max_len, 0))
            word_len.append(data['seq_len'])
            y.append(int(data['rel2idx']))

        x['word'] = torch.tensor(word)
        x['lens'] = torch.tensor(word_len)
        y = torch.tensor(y)
        
        return x, y
    
    elif model == "ner":
        assert word2id is not None and label2id is not None
        inputs = []
        targets = []
        masks = []

        UNK = word2id.get('<unk>')
        PAD = word2id.get('<pad>')
        for item in batch:
            input = item[0].split(' ')
            target = item[-1].copy()
            input = [word2id.get(w, UNK) for w in input]
            target = [label2id.get(l) for l in target]
            assert len(input) == len(target)
            inputs.append(_padding(input, max_len, pad=PAD))
            targets.append(_padding(target, max_len, 0))
            masks.append(_masking(input, max_len, pad=[1, 0]))

        return torch.tensor(inputs), torch.tensor(targets), torch.tensor(masks).bool()