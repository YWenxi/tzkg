import torch
import os
from tzkg.reasoners import mln
from torch.utils.data import Dataset
from typing import Union
from shutil import copy
from tzkg.data_processing import Transfer
import warnings
import pandas as pd

def ensure_dir(d: str):
    """Make the sure the input path is a directory

    Args:
        d (str): path to the directory. If not exists, make a new one.
    """
    if not os.path.exists(d):
        os.makedirs(d)    


def setup_mainspace(
    main_path: str,
    source_dir: str
):
    """Setup mainspace directory.

    Args:
        main_path (str): targeted path for mainspace.
            This directory would hold two files: `train.txt` and `train_augmented.txt`.
            These two files would used for mln training stage.
        source_dir (str): source directory which must at least hold `train.txt`

    Raises:
        FileNotFoundError: if `train.txt` is not in source_dir

    Returns:
        str: absolute path for main_path/mainspace
    """
    ensure_dir(main_path)

    if "train.txt" in os.listdir(main_path) and "train_augmented.txt" in os.listdir(main_path):
        warnings.warn(f"train.txt and train_augmented.txt is already in {main_path}")
        return os.path.abspath(main_path)

    #only need to copy one files: train.txt -> train.txt, train_augmented.txt
    file_to_copy = "train.txt"
    if file_to_copy in os.listdir(source_dir):
        copy(os.path.join(source_dir, file_to_copy), os.path.join(main_path, "train.txt"))
        copy(os.path.join(source_dir, file_to_copy), os.path.join(main_path, "train_augmented.txt"))
    else:
        raise FileNotFoundError(f"Could not find file {file_to_copy} under {os.path.abspath(main_path)}")

    return os.path.abspath(main_path)


def setup_workspace(
    iteration_id: Union[int, float, str],
    main_path: str
):
    """Setup Workspace Directory for training under main_path

    Args:
        iteration_id (Union[int, float, str]): this would be the name of the workspace directory under the `main_path`
        main_path (str): where we could found the mln preprocessed outputs.

    ---
    Returns:
        str: workspace directory, where the training source data is placed.
    """
    workspace_path = os.path.join(main_path, str(iteration_id))
    ensure_dir(workspace_path)

    preprocessed_file_list = os.listdir(main_path)
    files_to_copy = {
        "train_augmented.txt": "train_kge.txt",
        "hidden.txt": "hidden.txt"
    }   # how do we copy files
    for k, v in files_to_copy.items():
        if k in preprocessed_file_list:
            copy(os.path.join(main_path, k), os.path.join(workspace_path, v))
        else:
            raise FileNotFoundError(f"Could not find file {k} under {main_path}")

    return os.path.abspath(workspace_path)


def setup_in_one_step(source_file: str, train_test_data_dir: str, main_path: str,
                         iteration_id: Union[int, float, str], name_space: str,
                         random_state=42, **config) -> dict:
    """A one-step functionals to setup all working directories.

    Args:
        source_file (str): source file
        train_test_dir (str): where stores train, test, valid sets
        main_path (str): where stores training inputs for mln
        iteration_id (Union[int, float, str]): EM training iteration id
        name_space (str): move mln outputs to this directory and then ready for kge training
        random_state (int, optional): random state for dataset splitation. Defaults to 42.

    Raises:
        ValueError: If source file is not in desired format

    Returns:
        dict: metadata
    """
    # set up training files which could be used for kge training from .owl/.txt/.csv/rdf
    if source_file.split('.')[-1] not in ["csv", "txt", "owl"]:
        raise ValueError(f"Input source file {source_file} is not in required format, "
                         "should be in [csv, txt, owl].")
    trf = Transfer(source_file, name_space)

    ensure_dir(train_test_data_dir)
    trf.save_to_trainable_sets(train_test_data_dir, convert_entities=True, convert_relations=True, random_state=random_state)

    main_path = setup_mainspace(main_path, train_test_data_dir)
    mln(main_path)
    workspace_path = setup_workspace(iteration_id, main_path)

    metadata = {
            "source_file": os.path.abspath(source_file),
            "train_test_data_dir": os.path.abspath(train_test_data_dir),
            "main_path": main_path,
            "main_dir": main_path,
            "iteration_id": iteration_id,
            "workspace_path": workspace_path,
            "name_space": name_space
        }
    if isinstance(config, dict):
        config.update(metadata)
        return config

    return metadata

def read_triples(data_file):
    """Get triples data from file.

    Args:
        data_file (str): path of data file

    Returns:
        list: Nx3
    """
    df = pd.read_csv(data_file, header=None, sep="\t")
    # return df.to_numpy().tolist()

    # need to change the inside list to be tuples so that it is hashable
    temp = df.to_numpy().tolist()
    return [tuple(inner_list) for inner_list in temp]

def read_dict(data_file):
    df = pd.read_csv(data_file, sep="\t", header=None, names=["id", "ent_rel"])
    ent_rel_2_id = df.set_index("ent_rel").to_dict()["id"]
    id_2_ent_rel = df.set_index("id").to_dict()["ent_rel"]
    return ent_rel_2_id, id_2_ent_rel

#########################################


# def _padding(x, max_len, pad=0):
#     return x + [0] * (max_len - len(x))

# def _masking(x, max_len, pad=[1, 0]):
#      return [pad[0]] * len(x) + [pad[-1]] * (max_len - len(x))


# def collate_fn(batch, word2id=None, label2id=None, model="ner"):
#     """
#     对当前batch进行padding处理, 然后区分x, y;
#     Arg : 
#         batch () : 数据集
#     Returna : 
#         x (dict) : key为词, value为长度
#         y (List) : 关系对应值的集合
#     """
#     batch.sort(key=lambda data: data['seq_len'], reverse=True)
#     max_len = 512

#     if model == "re":
#         x, y = dict(), []
#         word, word_len = [], []
#         for data in batch:
#             word.append(_padding(data['token2idx'], max_len, 0))
#             word_len.append(data['seq_len'])
#             y.append(int(data['rel2idx']))

#         x['word'] = torch.tensor(word)
#         x['lens'] = torch.tensor(word_len)
#         y = torch.tensor(y)
        
#         return x, y
    
#     elif model == "ner":
#         assert word2id is not None and label2id is not None
#         inputs = []
#         targets = []
#         masks = []

#         UNK = word2id.get('<unk>')
#         PAD = word2id.get('<pad>')
#         for item in batch:
#             input = item[0].split(' ')
#             target = item[-1].copy()
#             input = [word2id.get(w, UNK) for w in input]
#             target = [label2id.get(l) for l in target]
#             assert len(input) == len(target)
#             inputs.append(_padding(input, max_len, pad=PAD))
#             targets.append(_padding(target, max_len, 0))
#             masks.append(_masking(input, max_len, pad=[1, 0]))

#         return torch.tensor(inputs), torch.tensor(targets), torch.tensor(masks).bool()