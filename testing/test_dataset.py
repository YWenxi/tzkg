# Test dataset utils
from tzkg.datasets import TrainDataset, BidirectionalOneShotIterator
from tzkg.datasets.utils import setup_workspace, setup_in_one_step, ensure_dir, read_triples, read_dict
from tzkg.data_processing import Transfer
from tzkg.reasoners import mln_preprocessing
from torch.utils.data import DataLoader
import pytest
from .test_data_processing import datasets_dir, datasets_name, test_output_dir, name_space
import os


def test_setup_workspace():
    for dir in datasets_dir:
        for name in datasets_name:
            source_dir = os.path.join(dir, name)

            # stores train/valid/test dataset files
            train_dir = os.path.join(source_dir, test_output_dir)
            if os.path.exists(train_dir):
                os.system(f"rm -r {train_dir}")
            ensure_dir(train_dir)

            # search csv, owl, or rdf in source dir
            source_list = os.listdir(source_dir)
            namespace = ""
            for file in source_list:
                if file.split(".")[-1] in ["csv", "owl", "rdf", "txt"]:
                    source_file = os.path.join(source_dir, file)
                    trf = Transfer(source_file, name_space)
                    
            trf.save_to_trainable_sets(train_dir)

            # stores files that should be used for mln preprocessing
            main_path = os.path.join(train_dir, "record")
            mln_preprocessing(main_path, train_dir)
            

            # stores training set for kge
            workspace = setup_workspace(0, main_path)

def test_setup_in_one_step_and_pytorch_dataset(file_list=None):
    if file_list is None:
        file_list = [
            "./test-data/weapons/weapons.csv",
            "./test-data/cdmo/cdmo.owl",
        ]
    for file in file_list:
        data_dir = os.path.dirname(file)
        if os.path.exists(os.path.join(data_dir, "test_output")):
            os.system(f"rm -r {os.path.join(data_dir, 'test_output')}")
        metadata = setup_in_one_step(
            file,
            os.path.join(data_dir, "test_output"),
            os.path.join(data_dir, "test_output", "record"),
            0,
            "http://tzkg/#"
        )

        # get triples
        train_triples = read_triples(
            os.path.join(metadata["workspace_path"], "train_kge.txt")
        )
        entity2id, id2entity = read_dict(
            os.path.join(metadata["train_test_data_dir"], "entities.dict")
        )
        relation2id, id2relation = read_dict(
            os.path.join(metadata["train_test_data_dir"], "relations.dict")
        )
        nentity = len(entity2id)
        nrelation = len(relation2id)
        negative_sample_size = 256
        batch_size = 1024
        cpu_num = 8

        # Set training dataloader iterator
        train_dataloader_head = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, negative_sample_size, 'head-batch'), 
            batch_size=batch_size,
            shuffle=True, 
            num_workers=max(1, cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )
        
        train_dataloader_tail = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, negative_sample_size, 'tail-batch'), 
            batch_size=batch_size,
            shuffle=True, 
            num_workers=max(1, cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )
        
        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

        check_conditions = [
            ("tensor", [batch_size, 3]),
            ("tensor", [batch_size, negative_sample_size]),
            ("tensor", [batch_size]),
            ("str",)
        ]
        for _ in range(2):
            first_batch = next(train_iterator)
            # print("-"*14 + " Print Batch " + "-"*14)
            for check_condition, out in zip(check_conditions, first_batch):
                if check_condition[0] == "tensor":
                    assert list(out.shape) == check_condition[1]
                elif check_condition[0] == "str":
                    pass
        # print("-"*17 + " Done! " + "-"*17)
    
    return train_iterator, nentity, nrelation, negative_sample_size