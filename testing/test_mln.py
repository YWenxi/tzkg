import pytest
import os
from tzkg.reasoners import mln, mln_preprocessing

from tzkg.datasets.utils import ensure_dir
from .test_data_processing import datasets_dir, datasets_name, test_output_dir

mln_path = "tzkg/reasoners/mln/mln"

def test_compiled():
    print("--> Check if mln.cpp is compiled.")
    print(__name__)
    if not os.path.isfile("./tzkg/reasoners/mln/mln"):
        raise FileNotFoundError("Must compile mln.cpp first! Try: g++ -O3 mln.cpp -o mln -lpthread")
    os.system("tzkg/reasoners/mln/mln")
    
def _test_mln_on_some_data(data_dir: str):
    print("--> Check mln.py")

    main_path = os.path.join(data_dir, "record")
    workspace_path = os.path.join(main_path, "0")
    ensure_dir(main_path)
    ensure_dir(workspace_path)
    config = {
        "mln_path": mln_path,
        "main_path": main_path,
        "workspace_path": workspace_path,
        "mln_threshold_of_rule": 0.1,
        "mln_threshold_of_triplet": 0.5,
        "mln_iters": 1000,
        "mln_lr": 0.0001,
        "mln_threads": 8,
        "preprocess": True
    }

    os.system('cp {}/train.txt {}/train.txt'.format(data_dir, main_path))
    os.system('cp {}/train.txt {}/train_augmented.txt'.format(data_dir, main_path))
    mln(**config)

def _test_mln_fb15k():
    _test_mln_on_some_data("test-data/FB15k")

@pytest.mark.dependency(["test_dataset"])
def test_mln_custom():
    for dir in datasets_dir:
        for dataset_name in datasets_name:
            test_mln_dir = os.path.join(dir, dataset_name, test_output_dir)
            _test_mln_on_some_data(test_mln_dir)

@pytest.mark.dependency(["test_get_train_directory"])
def test_mln_preprocessing():
    for dir in datasets_dir:
        for dataset_name in datasets_name:
            test_mln_dir = os.path.join(dir, dataset_name, test_output_dir)
            mln_preprocessing(
                main_path=os.path.join(test_mln_dir, "record"),
                train_dir=test_mln_dir
            )