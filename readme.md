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