conda create -n molsetrep python=3.10

pip install torch torchvision torchaudio
pip install torch_geometric

python -c 'import torch; print(torch.__version__)'

pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.1+cu117.html

pip install -e .

pip install wandb


# For benchmarks that need pymatgen
conda install --channel conda-forge pymatgen