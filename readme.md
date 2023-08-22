# Knowledge Reasoning Demo



## To-Do
- [x] `dataloader.py`
  - [x] testing
- [x] mln
  - [x] testing
- [x] kge modules
  - [x] testing
  - [x] solve the problems that the logger does output to console and save the workspace directory

## Installation 
1. Setup Python Enviroment.
    ```shell
    python -m venv {your-virtual-enviroment-name}
    source {your-virtual-enviroment-name}/bin/activate
    pip install -r requirements.txt
    ```
2. Compile `mln.cpp` file by the following scripts.
    ```shell
    cd tzkg/reasoners/mln
    g++ -O3 mln.cpp -o mln -lpthread
    ```

Or simply,
```shell
bash setup.sh
```

## Quick Start

### Train from a single source file

_Please refer to the `test_train.py`, which contains a whole training process._

In this case, we train from a single source file. This file may ended with `.csv, .txt, .owl, .rdf(Under Development!!!)`.

We first setup a few directories:
  - train_test_data_dir: stores training/valid/test sets and dictionary files to convert indices in the datasets to strings.
  - main_path: stores all training outputs, both from `mln` and `kge`
  - workingspace(s): under `main_path`, stores `kge` outputs in different iterations.
    - therefore they are usually named after the name of iterations, e.g. `workspace_path = {main_path}/{iteration}`
  
The following code snippet shows a typical training process with only one iteration.

```python
from tzkg.inference import train
from tzkg.inference.config import testing_configs, override_config
from tzkg.datasets.utils import setup_in_one_step
from tzkg.reasoners import mln

# set the above directories and do mln preprocessing
metadata = setup_in_one_step(
  source_file=testing_configs.source_file,
  train_test_dir=testing_configs.train_test_data_dir,
  main_path=testing_configs.main_path,
  iteration_id=0,
  name_space=testing_configs.name_space
)

# update the configuration
override_config(testing_configs, metadata)

# train with kge
train(testing_configs)

# feed back to mln
mln(testing_configs.main_path, testing_configs.workspace_path, preprocess=False)
```

**Notes:**
- The `setup_in_oen_step()` function reads the original inputs (`csv/txt/owl/rdf`) and then genetate a set of `{train/test/valid}.txt` files which should be used by mln and kge modules. It contains following steps:
  - data preprocessing: from original data format to a directory where we stores `{train/test/valid}.txt` files.
  - mln preprocessing: search for rules, find hidden triples
  - move output files to `workspace` directory
- The output `metadata` is a dict object storing information which would later be used to update configuration. It could be used to update `configs`
- The `train()` function runs all KGE training process, including validation and test. And it does an `infer_step` to output final scores to all hidden triples in `annotation.txt`.

### Train from train/test/valid sets
If you have already stored `train/test/valid` files in some directory (make sure you stores them using a csv format ending with `.txt` and using separation `\t`; also, make sure you stores the correspoding dictionary files), you only have to configure it using
```python
configs.train_test_data_dir = f"{your_train_test_dir}"
```
Also, set up where you want to stores training outputs and iteration name/id
```python
configs.main_path = f"{some-empty-directory}"
configs.iteration_id = # some id

setup_mainspace(configs.main_path, configs.train_test_data_dir)
configs.workspace_path = setup_workspace(configs.iteration_id, configs.main_path)
```
Now train as before
```python
train(configs)
```


## Configuration for Testing
- Some test settings could be found at 
  - `tzkg/inference/config.py`, or
  - saved `configs.json` on gpu server: `pLogicNet/record/{timestamp}/{iteration}/TransE`

## References
- `run.py` at `pLogicNet` and `run.py` ar `pLogicNet/kge`