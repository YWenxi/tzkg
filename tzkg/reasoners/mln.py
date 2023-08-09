# call mln tools
import os

__all__ = ["mln"]

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
    """Markov Logic Network Toolkit
        mln_path
    """
    if mln_path is None:
        mln_path = os.path.abspath(__file__)
        mln_path = os.path.join(mln, "mln", "mln")

    if preprocess:
        cmd = '{} -observed {}/train.txt -out-hidden {}/hidden.txt -save {}/mln_saved.txt -thresh-rule {} -iterations 0 -threads {}'.format(mln_path, main_path, main_path, main_path, mln_threshold_of_rule, mln_threads)
    else:
        cmd = '{} -load {}/mln_saved.txt -probability {}/annotation.txt -out-prediction {}/pred_mln.txt -out-rule {}/rule.txt -thresh-triplet 1 -iterations {} -lr {} -threads {}'.format(mln_path, main_path, workspace_path, workspace_path, workspace_path, mln_iters, mln_lr, mln_threads)
    
    os.system(cmd)
    return cmd