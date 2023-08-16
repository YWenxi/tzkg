# Knowledge Reasoning Demo



## To-Do
- [ ] `dataloader.py`
  - [ ] testing
- [ ] mln
  - [ ] testing
- [ ] kge modules
  - [ ] testing

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

## How to run a training process

Please refer to the `test_train.py`, which contains a whole training process.

  ```python
  from tzkg.inference import train
  from tzkg.inference.config import testing_configs, override_config
  from tzkg.datasets.utils import setup_in_one_step
  from tzkg.reasoners import mln

  metadata = setup_in_one_step(
      source_file=testing_configs.source_file,
      train_test_dir=testing_configs.train_test_data_dir,
      main_path=testing_configs.main_path,
      iteration_id=0,
      name_space=testing_configs.name_space
  )

  override_config(testing_configs, metadata)

  train(testing_configs)

  mln(testing_configs.main_path, testing_configs.workspace_path, preprocess=False)
  ```

- The `setup_in_oen_step()` function reads the original inputs (`csv/txt/owl/rdf`) and then genetate a set of `{train/test/valid}.txt` files which should be used by mln and kge modules.
- The output `metadata` is a dict object storing all possible 