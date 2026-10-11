# BPDC FL Framework

A PyTorch federated learning framework with FedAvg, FairRFL, FairMedian, FedCDA and FedMut, plus MNIST, CIFAR-10 and Tiny ImageNet loaders.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/s4meer-git/BPDC_FL_Framework/blob/main/JPNotebooks/Colab.ipynb)

## Run on Google Colab

Open the notebook above, select **Runtime → Change runtime type → GPU**, and run the cells in order. The launcher clones this repository, connects Drive, lets you choose one method and dataset, and saves method CSVs and analysis PDFs.

Start with a short run to measure the assigned GPU's speed. The launcher's experiment preset uses 31 clients, 40 rounds and 5 local epochs. Each method runs separately, and a rerun replaces that method's CSV in the chosen comparison folder.

See [the Colab guide](JPNotebooks/COLAB.md) for the complete workflow, saved-file locations and GitHub synchronization.

## Run locally

Use an environment with PyTorch, torchvision and NumPy installed. Install the notebook dependencies with `python -m pip install -r JPNotebooks/requirements.txt`.

Choose your settings in `configuration.py`, then run `python main.py` from the project root. Device selection uses CUDA when available, otherwise MPS. MNIST and CIFAR-10 download automatically; Tiny ImageNet should be extracted into `data/tiny-imagenet-200`.

The analysis notebook is `JPNotebooks/Analysis.ipynb`. Choose the Tiny ImageNet, MNIST or CIFAR-10 section, select its method CSV folder in the loading cell, and run that section's three plotting cells. It checks run completeness and exports accuracy and loss PDFs.

## Methods and experiment limits

See [method references](methods/REFERENCES.md) for original authors, papers and the framework's adaptations. FairMedian uses one minibatch per client per round and needs an odd client count. FedCDA starts history search after its configured warmup. These settings matter when comparing methods.

This repository does not include dataset downloads, virtual environments or saved experiments. It does not implement checkpoint/resume: saved CSVs contain measurements, not restorable training state.

The project license is [MIT](LICENSE). The adapted FedMut code retains its original [MIT notice](LICENSE.FedMut).
