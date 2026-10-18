# BPDC FL Framework

A PyTorch federated learning framework with FedAvg, FairRFL, FedCDA, FedMut and FCFL, using MNIST, CIFAR-10 or Tiny ImageNet.

[![Open setup notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/s4meer-git/BPDC_FL_Framework/blob/main/JPNotebooks/setup_remote.ipynb)

Open **[JPNotebooks/setup_remote.ipynb](JPNotebooks/setup_remote.ipynb)** to configure and run the framework locally or remotely. It creates the configuration, prepares datasets, runs the selected methods, and provides optional diagnostic and final plotting cells. `main.py` is the training engine called by the notebook.

For local use, install PyTorch and torchvision for your machine, then install the notebook dependencies with `python -m pip install -r JPNotebooks/requirements.txt`. Open the notebook using your Python environment in Jupyter or VS Code. For Colab, select a GPU runtime; setup keeps the existing PyTorch installation and mounts Drive for result backups.

The notebook starts with a five-round, one-seed smoke test. Set `SMOKE_TEST = False` for the experiment setup: 30 clients, 40 rounds, five local epochs and three seeds, summarised as mean ± standard deviation. Choose datasets and methods in lists, and use `config.selfishness = True` or `False` to compare attacked and honest runs. Completed matching runs are skipped by default. Incomplete runs restart from round 1; CSV files are not model checkpoints.

Results are saved as `Results/<dataset>/RUN1, RUN2, RUN3/<method>.csv` (one RUN per seed), for example `Results/tinyimagenet/RUN1/fedavg.csv`; selfish runs use `<dataset>_selfish` and smoke tests `smoke_<dataset>`. Final plotting cells save PDFs beside these CSVs. On Colab, CSVs are backed up to Drive after every completed round and at the end; final PDFs are copied when you run their cells. Diagnostic plots and local testing files stay under ignored `temp/`.

See [the setup guide](JPNotebooks/SETUP.md) for settings, folders and synchronization. The separate [analysis notebook](JPNotebooks/Analysis.ipynb) remains available for older results, average accuracy/loss plots and final client testing accuracy.

See [method references](methods/REFERENCES.md) for the authors, equations and adaptations. Introducing selfish clients does not guarantee a particular ranking. Compare mean accuracy, worst-client accuracy and client standard deviation together, including the honest-client group.

Datasets, virtual environments and saved experiments are not part of the source distribution. The project license is [MIT](LICENSE); the adapted FedMut code retains its original [MIT notice](LICENSE.FedMut).
