# Knowledge Reasoning Demo



## To-Do
[] `dataloader.py`

## Installation 
- Setup Python Enviroment.
```shell
python -m venv knowledge-reasoning
source knowledge-reasoning/bin/activate
pip install -r requirements.txt
```
- Compile `mln.cpp` files by the following scripts
```shell
cd algorithm/mln
g++ -O3 mln.cpp -o mln -lpthread
cd ..
cd ..
```