conda create -n csa python=3.9 -y
conda activate csa

conda install pytorch=1.10 torchvision torchaudio -c pytorch -c nvidia -y
conda install pyg=2.0.4 -c pyg -c conda-forge -y
conda install -c conda-forge graph-tool==2.45 -y

# RDKit is required for OGB-LSC PCQM4Mv2 and datasets derived from it.  
conda install openbabel fsspec rdkit -c conda-forge -y

pip install torchmetrics
pip install performer-pytorch
pip install ogb
pip install tensorboardX
pip install wandb

conda clean --all -y