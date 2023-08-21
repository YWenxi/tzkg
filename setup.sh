# setup virtual environment
python3 -m venv kg-demo-venv
source kg-demo-venv/bin/activate # Windows: myenv\Scripts\activate
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# compile mln.cpp
cd tzkg/reasoners/mln
g++ -O3 mln.cpp -o mln -lpthread
cd ../../..

