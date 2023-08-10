# call mln tools
import os

__all__ = ["mln", "mln_preprocessing"]

def mln(
    main_path: str,
    workspace_path: str = None,
    mln_path: str = None,
    mln_threshold_of_rule: float = 0.1,
    mln_threshold_of_triplet: float = 0.5,
    mln_iters: int = 1000,
    mln_lr: float = 0.0001,
    mln_threads: int = 8,
    preprocess: bool = True
):
    """MLN Toolkits

    Args:
        main_path (str): where the all mln output is stored, usually named as `./{source_dir}/record` 
        workspace_path (str, optional): where we store outputs in each iterations, usually named as `{main_pace}/{iteration_id}`. 
            Only used when `preproces=False`. Defaults to None.
        mln_path (str, optional): which compiled output file to use. Defaults to None, which leads to the built-in compiled `mln.o`.
        mln_threshold_of_rule (float, optional): Defaults to 0.1.
        mln_threshold_of_triplet (float, optional): Defaults to 0.5.
        mln_iters (int, optional): Defaults to 1000.
        mln_lr (float, optional): Defaults to 0.0001.
        mln_threads (int, optional): Defaults to 8.
        preprocess (bool, optional): Defaults to True.

        - When doing preprocessing, the main_path should 

    Returns:
        str: shell command
    """
    if mln_path is None:
        mln_path = os.path.abspath(__file__).split(".")[0]

    if preprocess:
        cmd = '{} -observed {}/train.txt -out-hidden {}/hidden.txt -save {}/mln_saved.txt -thresh-rule {} -iterations 0 -threads {}'.format(mln_path, main_path, main_path, main_path, mln_threshold_of_rule, mln_threads)
    else:
        cmd = '{} -load {}/mln_saved.txt -probability {}/annotation.txt -out-prediction {}/pred_mln.txt -out-rule {}/rule.txt -thresh-triplet 1 -iterations {} -lr {} -threads {}'.format(mln_path, main_path, workspace_path, workspace_path, workspace_path, mln_iters, mln_lr, mln_threads)
    
    os.system(cmd)
    return cmd

def mln_preprocessing(main_path: str, train_dir: str, **config):
    from tzkg.datasets.utils import setup_mainspace
    main_path = setup_mainspace(main_path, train_dir)
    mln(main_path, **config)