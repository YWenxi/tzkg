# setup virtual environment
python -m venv {your-virtual-enviroment-name}
source {your-virtual-enviroment-name}/bin/activate # Windows: myenv\Scripts\activate
pip install -r requirements.txt

# compile mln.cpp
cd tzkg/reasoners/mln
g++ -O3 mln.cpp -o mln -lpthread
cd ../../..