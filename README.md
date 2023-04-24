# CSA: Self-Attention in Colors

Official implementation of the graph transformer presented in:
[![arXiv](https://img.shields.io/badge/arXiv-2304.10933-b31b1b.svg)](https://arxiv.org/abs/2304.10933)

This repository was initially forked off the [GraphGPS](https://github.com/rampasek/GraphGPS) repository

![CSA-viz](CSA_network.png)

### Python environment setup with Conda

Launch `bash -i install.sh` to build conda environment, followed by `conda activate csa` to run the code on GPUs.


### Running CSA
```bash
conda activate csa

# Running CSA with RWSE and tuned hyperparameters for ZINC.
python main.py --cfg configs/CSA/zinc-CSA+RWSE.yaml  wandb.use False

# Running a debug/dev config for ZINC.
python main.py --cfg tests/configs/graph/zinc.yaml  wandb.use False
```


### Benchmarking GPS on 4 datasets
See `run/run_experiments.sh` script to run multiple random seeds per each of the 11 datasets. We rely on Slurm job scheduling system.

Alternatively, you can run them in terminal following the example below. Configs for all 11 datasets are in `configs/GPS/`.
```bash
conda activate csa
# Run 10 repeats with 10 different random seeds (0..9):
python main.py --cfg configs/CSA/zinc-CSA+RWSE.yaml  --repeat 10  wandb.use False
# Run a particular random seed:
python main.py --cfg configs/GPS/zinc-CSA+RWSE.yaml  --repeat 1  seed 42  wandb.use False
```


### W&B logging
To use W&B logging, set `wandb.use True` and have a `gtransformers` entity set-up in your W&B account (or change it to whatever else you like by setting `wandb.entity`).



### Unit tests

To run all unit tests, execute from the project root directory:

```bash
python -m unittest -v
```

Or specify a particular test module, e.g.:

```bash
python -m unittest -v unittests.test_eigvecs
```


## Citation

If you find this work useful, please cite our paper:
```
@article{menegaux2023CSA,
  title={Self-Attention in Colors: Another Take on Encoding Graph Structure in Transformers}, 
  author={Romain Menegaux and Emmanuel Jehanno and Margot Selosse and Julien Mairal},
  year={2023},
  eprint = {2304.10933},
  archivePrefix = {arXiv},
  primaryClass={cs.LG},
}
```
