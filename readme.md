# Knowledge Reasoning Demo



## To-Do
- [ ] `dataloader.py`
  - [ ] [to-do items]

## Installation 
1. Setup Python Enviroment.
    ```shell
    python -m venv knowledge-reasoning
    source knowledge-reasoning/bin/activate
    pip install -r requirements.txt
    ```
2. Compile `mln.cpp` file by the following scripts.
    ```shell
    cd algorithm/mln
    g++ -O3 mln.cpp -o mln -lpthread
    cd ..
    cd ..
    ```