import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import logging
from tzkg.reasoners import KGE
from tzkg.datasets import TrainDataset, BidirectionalOneShotIterator
from tzkg.datasets.utils import read_triples, read_dict, ensure_dir
from .config import override_config

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    args.log_file = log_file

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))

def init_model(args):

    kge_model = KGE(
        model_name=args.model_name,
        nentity=args.nentity,
        nrelation=args.nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding
    )

    if args.cuda:
        kge_model = kge_model.cuda()
    
    return kge_model

def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    
    argparse_dict = dict(vars(args))
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )
    
    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'entity_embedding'), 
        entity_embedding
    )
    
    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding'), 
        relation_embedding
    )

def augment_triplet(pred_file:str, trip_file:str, out_file:str, threshold:float, save_new_triple_only=False):
    """Augment triplets set with new triplets whose prediction score > threshold

    Args:
        pred_file (str): prediction output file. Usually named as `{workspace_path}/pred_mln.txt`
        trip_file (str): original triplet file. Usually named as `{main_path}/train.txt`
        out_file (str): output triplet file. Usually named and saved as `{workspace_path}/train_augmented.txt`
        threshold (float): probability score threshold
        save_new_triple_only (bool, optional): If True, save new triplets only, otherwise first store old triplets and then the new ones. Defaults to False.
    """
    with open(pred_file, 'r') as fi:
        data = []
        for line in fi:
            l = line.strip().split()
            data += [(l[0], l[1], l[2], float(l[3]))]

    with open(trip_file, 'r') as fi:
        trip = set()
        if not save_new_triple_only:
            for line in fi:
                l = line.strip().split()
                trip.add((l[0], l[1], l[2]))

        for tp in data:
            if tp[3] < threshold:
                continue
            trip.add((tp[0], tp[1], tp[2]))

    with open(out_file, 'w') as fo:
        for h, r, t in trip:
            fo.write('{}\t{}\t{}\n'.format(h, r, t))


# This function computes the probability of a triplet being true based on the MLN outputs.
def mln_triplet_prob(h, r, t, hrt2p):
    # KGE algorithms tend to predict triplets like (e, r, e), which are less likely in practice.
    # Therefore, we give a penalty to such triplets, which yields some improvement.
    if h == t:
        if hrt2p.get((h, r, t), 0) < 0.5:
            return -100
        return hrt2p[(h, r, t)]
    else:
        if (h, r, t) in hrt2p:
            return hrt2p[(h, r, t)]
        return 0.5


# This function reads the outputs from MLN and KGE to do evaluation.
# Here, the parameter weight controls the relative weights of both models.
def evaluate(mln_pred_file:str, kge_pred_file:str, output_file:str, weight):
    hit1 = 0
    hit3 = 0
    hit10 = 0
    mr = 0
    mrr = 0
    cn = 0

    hrt2p = dict()
    with open(mln_pred_file, 'r') as fi:
        for line in fi:
            h, r, t, p = line.strip().split('\t')[0:4]
            hrt2p[(h, r, t)] = float(p)

    with open(kge_pred_file, 'r') as fi:
        while True:
            truth = fi.readline()
            preds = fi.readline()

            if (not truth) or (not preds):
                break

            truth = truth.strip().split()
            preds = preds.strip().split()

            h, r, t, mode, original_ranking = truth[0:5]
            original_ranking = int(original_ranking)

            if mode == 'h':
                preds = [[pred.split(':')[0], float(pred.split(':')[1])] for pred in preds]

                for k in range(len(preds)):
                    e = preds[k][0]
                    preds[k][1] += mln_triplet_prob(e, r, t, hrt2p) * weight

                preds = sorted(preds, key=lambda x:x[1], reverse=True)
                ranking = -1
                for k in range(len(preds)):
                    e = preds[k][0]
                    if e == h:
                        ranking = k + 1
                        break
                if ranking == -1:
                    ranking = original_ranking

            if mode == 't':
                preds = [[pred.split(':')[0], float(pred.split(':')[1])] for pred in preds]

                for k in range(len(preds)):
                    e = preds[k][0]
                    preds[k][1] += mln_triplet_prob(h, r, e, hrt2p) * weight

                preds = sorted(preds, key=lambda x:x[1], reverse=True)
                ranking = -1
                for k in range(len(preds)):
                    e = preds[k][0]
                    if e == t:
                        ranking = k + 1
                        break
                if ranking == -1:
                    ranking = original_ranking

            if ranking <= 1:
                hit1 += 1
            if ranking <=3:
                hit3 += 1
            if ranking <= 10:
                hit10 += 1
            mr += ranking
            mrr += 1.0 / ranking
            cn += 1

    mr /= cn
    mrr /= cn
    hit1 /= cn
    hit3 /= cn
    hit10 /= cn

    print('MR: ', mr)
    print('MRR: ', mrr)
    print('Hit@1: ', hit1)
    print('Hit@3: ', hit3)
    print('Hit@10: ', hit10)

    with open(output_file, 'w') as fo:
        fo.write('MR: {}\n'.format(mr))
        fo.write('MRR: {}\n'.format(mrr))
        fo.write('Hit@1: {}\n'.format(hit1))
        fo.write('Hit@3: {}\n'.format(hit3))
        fo.write('Hit@10: {}\n'.format(hit10))


################################################################################################

def train(args):

    # Write logs to checkpoint and console
    if args.save_path is None:
        args.save_path = os.path.join(args.workspace_path, args.model_name)
        ensure_dir(args.save_path)
    set_logger(args)

    # get triples
    train_triples = read_triples(os.path.join(args.workspace_path, "train_kge.txt"))
    
    valid_triples = read_triples(os.path.join(args.train_test_data_dir, 'valid.txt'))
    train_original_triples = read_triples(os.path.join(args.train_test_data_dir, 'train.txt'))
    test_triples = read_triples(os.path.join(args.train_test_data_dir, 'test.txt'))
    hidden_triples = read_triples(os.path.join(args.workspace_path, 'hidden.txt'))
    all_true_triples = train_original_triples + valid_triples + test_triples

    entity2id, id2entity = read_dict(os.path.join(args.train_test_data_dir, "entities.dict"))
    relation2id, id2relation = read_dict(os.path.join(args.train_test_data_dir, "relations.dict"))

    override_config(
        args,
        d = {"nentity": len(entity2id), "nrelation": len(relation2id)}
    )

    logging.info('Model: %s' % args.model_name)
    logging.info('Data Path: %s' % args.train_test_data_dir)
    logging.info('#entity: %d' % args.nentity)
    logging.info('#relation: %d' % args.nrelation)

    logging.info('#train: %d' % len(train_triples))
    logging.info('#train original: %d' % len(train_original_triples))
    logging.info('#valid: %d' % len(valid_triples))
    logging.info('#test: %d' % len(test_triples))
    logging.info('#hidden: %d' % len(hidden_triples))

    # init models
    kge_model = init_model(args)

    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    # Set training dataloader iterator
    train_dataloader_head = DataLoader(
        TrainDataset(train_triples, args.nentity, args.nrelation, args.negative_sample_size, 'head-batch'), 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=max(1, args.cpu_num//2),
        collate_fn=TrainDataset.collate_fn
    )
    
    train_dataloader_tail = DataLoader(
        TrainDataset(train_triples, args.nentity, args.nrelation, args.negative_sample_size, 'tail-batch'), 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=max(1, args.cpu_num//2),
        collate_fn=TrainDataset.collate_fn
    )
    
    train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
    
    # Set training configuration
    current_learning_rate = args.learning_rate
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, kge_model.parameters()), 
        lr=current_learning_rate
    )
    if args.warm_up_steps:
        warm_up_steps = args.warm_up_steps
    else:
        warm_up_steps = args.max_steps // 2

    # check if need checkpoint
    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
    
        current_learning_rate = checkpoint['current_learning_rate']
        warm_up_steps = checkpoint['warm_up_steps']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model_name)
        init_step = 0

    step = init_step
    
    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('learning_rate = %d' % current_learning_rate)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)

    if args.record:
        local_path = args.workspace_path
        ensure_dir(local_path)

        opt = vars(args)
        with open(local_path + '/opt.txt', 'w') as fo:
            for key, val in opt.items():
                fo.write('{} {}\n'.format(key, val))


    training_logs = []

    ###################################################################################    
    #Training Loop

    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('learning_rate = %d' % current_learning_rate)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)

    for step in tqdm(range(init_step, args.max_steps)):
        
        log = kge_model.train_step(kge_model, optimizer, train_iterator, args)
        
        training_logs.append(log)
        
        if step >= warm_up_steps:
            current_learning_rate = current_learning_rate / 10
            logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, kge_model.parameters()), 
                lr=current_learning_rate
            )
            warm_up_steps = warm_up_steps * 3
        
        if step % args.save_checkpoint_steps == 0:
            save_variable_list = {
                'step': step, 
                'current_learning_rate': current_learning_rate,
                'warm_up_steps': warm_up_steps
            }
            save_model(kge_model, optimizer, save_variable_list, args)
            
        if step % args.log_steps == 0:
            metrics = {}
            for metric in training_logs[0].keys():
                metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
            log_metrics('Training average', step, metrics)
            training_logs = []
            
        if args.do_valid and (step + 1) % args.valid_steps == 0:
            logging.info('Evaluating on Valid Dataset... [Under Development]')
            metrics, preds = kge_model.test_step(kge_model, valid_triples, all_true_triples, args)
            log_metrics('Valid', step, metrics)

            # --------------------------------------------------
            # Save the prediction results of KGE on validation set.
            # --------------------------------------------------

            if args.record:
                # Save the final results
                with open(local_path + '/result_kge_valid.txt', 'w') as fo:
                    for metric in metrics:
                        fo.write('{} : {}\n'.format(metric, metrics[metric]))

                # Save the predictions on test data
                with open(local_path + '/pred_kge_valid.txt', 'w') as fo:
                    for h, r, t, f, rk, l in preds:
                        # do we really need to convert to entity here?
                        # fo.write('{}\t{}\t{}\t{}\t{}\n'.format(id2entity[h], id2relation[r], id2entity[t], f, rk))
                        fo.write('{}\t{}\t{}\t{}\t{}\n'.format(h, r, t, f, rk))
                        for e, val in l:
                            fo.write('{}:{:.4f} '.format(e, val))
                        fo.write('\n')
    
    save_variable_list = {
        'step': step, 
        'current_learning_rate': current_learning_rate,
        'warm_up_steps': warm_up_steps
    }
    save_model(kge_model, optimizer, save_variable_list, args)
    
    # --------------------------------------------------
    # Save the annotations on hidden triplets.
    # --------------------------------------------------

    if args.record:
        # Annotate hidden triplets
        scores = kge_model.infer_step(kge_model, hidden_triples, args)
        with open(local_path + '/annotation.txt', 'w') as fo:
            for (h, r, t), s in zip(hidden_triples, scores):
                fo.write('{}\t{}\t{}\t{}\n'.format(id2entity[h], id2relation[r], id2entity[t], s))
    
    if args.evaluate_train:
        logging.info('Evaluating on Training Dataset...')
        metrics, preds = kge_model.test_step(kge_model, train_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)