import unittest
import os
import torch
import torch.nn as nn

from tzkg.reasoners.kge.kge import KGE
from tzkg.inference.callbacks import ModelCheckpoint


def read_triple(file_path, entity2id, relation2id):
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)


class TestPlogicNet(unittest.TestCase):
    def setUp(self, data_path, entity2id, relation2id):
        self.train_triples = read_triple(os.path.join(data_path, "train_kge.txt"), entity2id, relation2id)
        self.train_original_triples = read_triple(os.path.join(data_path, "train.txt"), entity2id, relation2id)
        self.valid_triples = read_triple(os.path.join(data_path, "valid.txt"), entity2id, relation2id)
        self.test_triples = read_triple(os.path.join(data_path, "test.txt"), entity2id, relation2id)
        self.hidden_triples = read_triple(os.path.join(data_path, "hidden.txt"), entity2id, relation2id)
        
    def test_fit(self):
        model = KGE()

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=0.0001
        )
        scheduler = None
        callback = ModelCheckpoint()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model.compile(optimizer=optimizer, callbacks=callback, schedulers=scheduler, device=device)
        model.fit()
        output = model.predict()
        print(output.shape)

if __name__ == "__main__":
    main_path = "/Users/hechengda/Documents/codes/TZ-KG/data/FB15k"
    train_path = "/Users/hechengda/Documents/codes/TZ-KG/data/FB15k/training"
    ensure_dir(train_path)

    mln_threshold_of_rule = 0.1
    mln_threads = 8

    os.system('cp {}/train.txt {}/train.txt'.format(main_path, train_path))
    os.system('cp {}/train.txt {}/train_augmented.txt'.format(main_path, train_path))
    cmd_mln = './mln/mln -observed {}/train.txt -out-hidden {}/hidden.txt -save {}/mln_saved.txt -thresh-rule {} -iterations 0 -threads {}'.format(main_path, main_path, main_path, mln_threshold_of_rule, mln_threads)
    
    model = TestPlogicNet()
    model.setUp()
    model.test_fit()