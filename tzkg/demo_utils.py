import os
from pandas import DataFrame
from pathlib import Path
from tzkg.data_processing import read_rules_to_df, read_triplets_prediction_to_df
from tzkg.inference import train
from tzkg.inference.config import override_config, base_configs
from tzkg.inference.train_utils import augment_triplet, evaluate
from tzkg.datasets.utils import setup_in_one_step, setup_mainspace, setup_workspace
from tzkg.reasoners import mln

from typing import Union


__all__ = ["knowlegde_graph_completion", "train_from_source_file"]


def train_from_source_file(source_file, main_path, train_test_data_dir: None,
                           configs: base_configs = base_configs,
                           iterations=2):

    if train_test_data_dir is None:
        train_test_data_dir = Path(main_path).absolute() / "train_test_data"

    for iteration_id in range(iterations):
        if iteration_id == 0:
            metadata = setup_in_one_step(
                source_file= source_file,
                train_test_data_dir = train_test_data_dir,
                main_path = main_path,
                iteration_id = 0,
                name_space = configs.name_space
            )
        else:
            metadata = {
                "iteration_id": iteration_id,
                "workspace_path": setup_workspace(iteration_id, configs.main_path)
            }

        
        override_config(configs, metadata)

        train(configs)
        mln(configs.main_path, configs.workspace_path, preprocess=False)
        augment_triplet(
            pred_file = Path(configs.workspace_path) / "pred_mln.txt",
            trip_file = Path(configs.train_test_data_dir) / "train.txt",
            out_file = Path(configs.workspace_path) / "train_augmented.txt",
            threshold = configs.mln_threshold_of_triplet
        )
        os.system(f'cp {configs.workspace_path}/train_augmented.txt {configs.train_test_data_dir}/train_augmented.txt')
        evaluate(
            mln_pred_file=Path(configs.workspace_path) / "pred_mln.txt",
            kge_pred_file=Path(configs.workspace_path) / "pred_kge.txt",
            output_file=Path(configs.workspace_path) / "result_kge_mln.txt",
            # need to determine
            weight=configs.weight
        )

    return configs

def train_from_split_dataset(
        train_test_data_dir: str,
        main_path: str,
        configs: base_configs = base_configs,
        iterations: int = 2
):
    
    # setup mainspace
    override_config(configs, {
        "train_test_data_dir": Path(train_test_data_dir).absolute(),
        "main_path": setup_mainspace(main_path, train_test_data_dir),
        "iterations": iterations
    })

    for iteration_id in range(iterations):

        # mln preprocessing
        if iteration_id == 0:
            mln(configs.main_path)

        # setup workspace
        override_config(configs, {
            "iteration_id": iteration_id,
            "workspace_path": setup_workspace(iteration_id, configs.main_path)
        })

        # training, augment, and evaluate
        train(configs)
        mln(configs.main_path, configs.workspace_path, preprocess=False)
        augment_triplet(
            pred_file = Path(configs.workspace_path) / "pred_mln.txt",
            trip_file = Path(configs.train_test_data_dir) / "train.txt",
            out_file = Path(configs.workspace_path) / "train_augmented.txt",
            threshold = configs.mln_threshold_of_triplet
        )
        os.system(f'cp {configs.workspace_path}/train_augmented.txt {configs.train_test_data_dir}/train_augmented.txt')
        evaluate(
            mln_pred_file=Path(configs.workspace_path) / "pred_mln.txt",
            kge_pred_file=Path(configs.workspace_path) / "pred_kge.txt",
            output_file=Path(configs.workspace_path) / "result_kge_mln.txt",
            # need to determine
            weight=configs.weight
        )

        return configs

def knowlegde_graph_completion(
        # train:bool = False,
        source_file:Union[str,None] = None,
        train_test_data_dir:Union[str,None] = None,
        main_path:Union[str,None] = None,
        threshold: float = 0.5,
        iterations=2,
        configs: base_configs = base_configs
    ) -> DataFrame:
    

    override_config(configs, {
        "iterations": iterations,
        "mln_threshold_of_triplet": threshold
    })

    if source_file:
        configs = train_from_source_file(
            source_file, 
            main_path=main_path, 
            train_test_data_dir=train_test_data_dir, 
            iterations=iterations,
            configs=configs
            )
    elif train_test_data_dir:
        configs = train_from_split_dataset(train_test_data_dir, main_path, configs, iterations)
    else:
        raise FileNotFoundError("No input training source, try to specify source_file or train_test_data_dir!")

    triplets_with_score = read_triplets_prediction_to_df(
        pred_output=Path(configs.workspace_path) / "annotation.txt",
        entity_dict_file=Path(configs.train_test_data_dir) / "entities.dict",
        relations_dict_file=Path(configs.train_test_data_dir) / "relations.dict"
    )

    return triplets_with_score[triplets_with_score["score"] >= threshold]