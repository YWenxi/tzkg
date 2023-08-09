import os
import json
import copy
from multiprocessing.sharedctypes import Value
import time
from datetime import timedelta
from pyld import jsonld

import numpy as np
import pandas as pd
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from sklearn.metrics import f1_score
from transformers import BertTokenizer

from tzkg.callbacks import CallbackList
from tzkg.datasets.reasoner_dataset import TrainDataset, TestDataset, BidirectionalOneShotIterator


class BaseModel(nn.Module):
    def __init__(self, name="base"):
        super(BaseModel, self).__init__()
        self.name = name

        # init for trainer
        self.stop_training = False
        self.history = None

    def forward(self, x):
        return x

    def fit(self,
            train_set,
            er_set,
            valid_set=None,
            negative_sample_size=128,
            batch_size=1024,
            valid_batch_size=4,
            adversarial_temperature=1.0,
            regularization=0.0,
            topk=100,
            uni_weight=False,
            negative_adversarial_sampling=False,
            steps=10000,
            save_checkpoint_steps=1000,
            verbose=0,
            num_workers=1,
            shuffle=True
            ):
        # self.train_set = train_set
        # self.entity2id, self.relation2id = er_set
        self.num_entity = len(er_set[0])
        self.num_relation = len(er_set[1])
        self.batch_size = batch_size
        self.negative_sample_size = negative_sample_size
        self.negative_adversarial_sampling = negative_adversarial_sampling
        self.adversarial_temperature = adversarial_temperature
        self.uni_weight = uni_weight
        self.regularization = regularization
        self.topk = topk
        ### ============== dataset process ============== ###

        ## init dataset
        train_dataset_head = TrainDataset(train_set, self.num_entity, self.num_relation, self.negative_sample_size, 'head-batch')
        train_dataset_tail = TrainDataset(train_set, self.num_entity, self.num_relation, self.negative_sample_size, 'tail-batch')
        train_dataloader_head = DataLoader(train_dataset_head, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, collate_fn=TrainDataset.collate_fn)
        train_dataloader_tail = DataLoader(train_dataset_tail, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, collate_fn=TrainDataset.collate_fn)
        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

        if valid_set:
            valid_triples = valid_set[0]
            all_triples = valid_set[1]
            valid_dataset_head = TestDataset(valid_triples, all_triples, self.num_entity, self.num_relation, self.negative_sample_size, 'head-batch')
            valid_dataset_tail = TestDataset(valid_triples, all_triples, self.num_entity, self.num_relation, self.negative_sample_size, 'tail-batch')
            valid_dataloader_head = DataLoader(valid_dataset_head, batch_size=valid_batch_size, num_workers=num_workers, shuffle=shuffle, collate_fn=TestDataset.collate_fn)
            valid_dataloader_tail = DataLoader(valid_dataset_tail, batch_size=valid_batch_size, num_workers=num_workers, shuffle=shuffle, collate_fn=TestDataset.collate_fn)
            # valid_iterator = BidirectionalOneShotIterator(valid_dataloader_head, valid_dataloader_tail)
            valid_iterator = [valid_dataloader_head, valid_dataloader_tail]

        ### ============== dataset process ============== ###

        ### init train state
        self.callbacks.on_train_begin()

        val_loss = 0
        best_f1 = 0
        start_time = time.time()
        for step in tqdm(range(steps)):
            logs = {}
            self.callbacks.on_epoch_begin(step)

            train_loss = self.train_epoch(train_iterator)
            logs["train_loss"] = train_loss

            if valid_set:
                metrics, predictions = self.val_epoch(valid_iterator)
                logs["metrics"] = metrics
                logs["predictions"] = predictions

            if step % save_checkpoint_steps:
                self.callbacks.on_epoch_end(step, logs)

            msg = ""
            if verbose:
                msg += f"step {step+1:<3} |  train_loss {train_loss:<.5f}"

                if valid_set:
                    msg += f"|  val_metrics {metrics:<.5f}"

                total_time = int(time.time() - start_time)
                msg += f"|  {str(timedelta(seconds=total_time)) + 's':<6}"
                print(msg)

            ### early stop
            if self.stop_training:
                print("early stop on step: {}".format(step + 1))
                break
        
    def train_epoch(self, dataloader, logs_out=False):
        model = self.train()
        self.optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(dataloader)

        positive_sample = positive_sample.to(self.device)
        negative_sample = negative_sample.to(self.device)
        subsampling_weight = subsampling_weight.to(self.device)

        negative_score = model((positive_sample, negative_sample), mode=mode)

        if self.negative_adversarial_sampling:
            negative_score = (F.softmax(negative_score * self.adversarial_temperature, dim = 1).detach() 
                              * F.logsigmoid(-negative_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        positive_score = model(positive_sample)
        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

        if self.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2
        
        if self.regularization != 0.0:
            #Use L3 regularization for ComplEx and DistMult
            regularization = self.regularization * (
                model.entity_embedding.norm(p = 3)**3 + 
                model.relation_embedding.norm(p = 3).norm(p = 3)**3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}
            
        loss.backward()

        self.optimizer.step()
        if self.scheduler:
           self.scheduler.step()

        if logs_out:
            log = {
                **regularization_log,
                'positive_sample_loss': positive_sample_loss.item(),
                'negative_sample_loss': negative_sample_loss.item(),
                'loss': loss.item()
            }

            return log
        else:
            return loss

    def val_epoch(self, dataloader):
        model = self.eval()
        logs = []
        
        predictions = []
        for ds in dataloader:
            for positive_sample, negative_sample, filter_bias, mode in ds:
                positive_sample = positive_sample.to(self.device)
                negative_sample = negative_sample.to(self.device)
                filter_bias = filter_bias.to(self.device)

                # Save prediction results
                prediction = positive_sample.data.cpu().numpy().tolist()
                batch_size = positive_sample.size(0)

                with torch.no_grad():
                    score = torch.sigmoid(model((positive_sample, negative_sample), mode))
                score += filter_bias

                #Explicitly sort all the entities to ensure that there is no test exposure bias
                valsort, argsort = torch.sort(score, dim = 1, descending=True)

                if mode == 'head-batch':
                    positive_arg = positive_sample[:, 0]
                elif mode == 'tail-batch':
                    positive_arg = positive_sample[:, 2]
                else:
                    raise ValueError('mode %s not supported' % mode)

                for i in range(batch_size):
                    #Notice that argsort is not ranking
                    ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                    assert ranking.size(0) == 1

                    # For each test triplet, save the ranked list (h, r, [ts]) and ([hs], r, t)
                    if mode == 'head-batch':
                        prediction[i].append('h')
                        prediction[i].append(ranking.item() + 1)
                        ls = zip(argsort[i, 0: self.topk].data.cpu().numpy().tolist(), valsort[i, 0: self.topk].data.cpu().numpy().tolist())
                        prediction[i].append(ls)
                    elif mode == 'tail-batch':
                        prediction[i].append('t')
                        prediction[i].append(ranking.item() + 1)
                        ls = zip(argsort[i, 0: self.topk].data.cpu().numpy().tolist(), valsort[i, 0: self.topk].data.cpu().numpy().tolist())
                        prediction[i].append(ls)

                    #ranking + 1 is the true ranking used in evaluation metrics
                    ranking = 1 + ranking.item()
                    logs.append({
                        'MR': float(ranking),
                        'MRR': 1.0/ranking,
                        'HITS@1': 1.0 if ranking <= 1 else 0.0,
                        'HITS@3': 1.0 if ranking <= 3 else 0.0,
                        'HITS@10': 1.0 if ranking <= 10 else 0.0,
                    })

                predictions += prediction

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics, predictions

    def compile(self, optimizer=None, loss=None, loss_weights=None, metrics=None, callbacks=None, schedulers=None, device="cpu"):
        self.device = device
        self.loss_weigts = loss_weights
        self.optimizer = optimizer
        
        self.callbacks = CallbackList(callbacks or [], self)
        self.schedulers = schedulers or []

        # init metics
        self.metrics = metrics or []

        # init device
        self.to(self.device)

        # init loss
        if loss:
            if isinstance(loss, list):
                if not self.loss_weights:
                    raise ImportError("loss_weights should not be None when using Multi-loss function")
                if not isinstance(self.loss_weights, list):
                    raise ValueError("loss_weights type expected as List")
                self.criterion = loss
                for l in self.criterion:
                    l.to(self.device)
            else:
                self.criterion = loss
                self.criterion.to(self.device)
        else:
            self.criterion = None

    def predict(self, triples, batch_size):
        scores = []
        model = self.eval()
        for k in range(0, len(triples), batch_size):
            bg = k
            ed = min(k + batch_size, len(triples))
            batch = triples[bg:ed]
            batch = torch.LongTensor(batch).to(self.device)
            score = torch.sigmoid(model(batch)).squeeze(1)
            scores += score.data.cpu().numpy().tolist()
        return scores