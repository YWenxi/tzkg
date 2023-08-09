import torch
import os
from torch.utils.data import Dataset
from typing import Union
from shutil import copy

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
    ensure_dir(main_path)

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
    """Setup Workspace Directory for training

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