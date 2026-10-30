# Running the framework

Use **setup_remote.ipynb** for both local and remote runs. It replaces `Colab.ipynb`. `main.py` remains the engine imported by the notebook; running it directly prints the notebook path.

## Setup and settings

Open the notebook from the project root or `JPNotebooks`. Locally, use an environment with PyTorch, torchvision, NumPy and the packages in `JPNotebooks/requirements.txt`. Device selection uses CUDA, then MPS, then CPU. Remote machines with a normal Jupyter environment follow this same local setup; Google Colab additionally clones or updates the GitHub checkout and connects Drive.

Run the setup, storage and settings cells in order. The settings cell constructs `configuration()` and overrides its defaults, so the values you see in that cell are the shared experiment values. The CSV records the actual configuration used. Method-specific defaults still come from `configuration.py`.

- `DATASETS`: choose any combination of `mnist`, `cifar10`, `tinyimagenet`.
- `METHODS`: default comparison is `fedavg`, `fairrfl`, `fedcda`, `fedmut`, `fcfl`.
- `SEEDS`: `[42, 43, 44]` for the setup's three independent runs. Each seed changes the partition and the initial model, and every method uses the same ones for a given seed. The smoke test uses `SMOKE_SEEDS = [42]`.
- To rerun a dataset with different settings, first move or rename its existing folder; the runner refuses to mix different settings in one folder.
- `SMOKE_TEST = True`: five rounds, saved under `smoke_<dataset>`. This still uses the full dataset and chosen client count. It checks that training and attack histories work; it is not a final comparison.
- `SMOKE_TEST = False`: the setup's 30 clients, 40 rounds, five local epochs, learning rate 0.01, momentum 0.5 for MNIST and 0.9 otherwise, and batch size 64.
- `RERUN_COMPLETED = False`: skip matching completed method CSVs. Set it to `True` to replace them intentionally.

Tiny ImageNet decodes a JPEG for every training image, which held the earlier T4 runs at about 4–6 hours per method regardless of GPU. On CUDA, `config.loader_workers = None` therefore decodes Tiny ImageNet with up to four background processes; results are identical to single-process loading for the same seed apart from the random flip augmentation, which stays the same for every method in a comparison. Set it to `0` to disable, or to a number to choose the process count.

On CUDA, training and evaluation forward passes run in mixed precision (`config.mixed_precision = None`, fp16 with PyTorch's gradient scaler), about twice as fast on T4, L4 and A100 tensor cores. Aggregation, fairness weighting and recovery stay in fp32. Results are statistically equivalent but not bit-identical to fp32 (a few tenths of a percentage point in checks), every method in a run uses the same precision, and the CSV records it; the runner refuses to mix fp16 and fp32 results in one folder. Set it to `False` for full fp32. CPU and MPS runs are unchanged.

MNIST and CIFAR-10 download through torchvision. The runner downloads Tiny ImageNet when its training directory is absent. Data and active CSVs stay on the machine running Python to avoid reading every batch through Drive.

FedCDA warms up for **10 rounds** in a full run (one in the smoke test), the same quarter of training as the authors' 50 of 200, and then searches its history. The old 50-round default never reached search in a 40-round run, so older FedCDA results describe a different setting and should not be mixed into a new comparison.

Each client is now tested on test images with its own training class mix (`config.test_partition = "matched"`), so client accuracy reflects how well the global model serves that client's data. Earlier runs used an independent Dirichlet test split and are recorded as `"dirichlet"`.

The CIFAR-10 CNN now has batch normalization in each convolution block, as the setup specifies. Earlier CIFAR-10 results used the model without it and are not comparable with new runs.

## Selfish-client comparison

`config.selfishness = False` runs the honest baseline. `True` applies the documented known-client-count attack to outgoing parameter updates. The settings cell exposes:

- `selfish_fraction`: `0.05` gives one selfish client (the paper's main scenario), `0.3` gives nine of 30 clients, using the integer part of the fraction times the client count, with at least one attacker when enabled.
- `selfish_strength = 0.7`: the attack equation's mixing coefficient.
- `selfish_start_round = 3`: rounds 1 and 2 train honestly to establish history.

The seed fixes the client IDs without consuming the training random generator. The IDs are identical across methods with the same settings. CSV IDs start at zero; plot labels start at one. Every enabled round records both the selected IDs and those actually applying the attack. Changing the switch also changes the output folder, preventing accidental baseline/attack mixing.

All five methods use the same attack rule and attacker selection. FairRFL applies it after fairness weighting and before recovery; the other methods apply it to trained parameters before aggregation. Buffers are not attacked. Full participation is required for the current attack-history implementation.

FairRFL keeps the paper's dynamic q and recovery, and weights clients by the received global model's loss as in its Algorithm 3. `fairrfl_dynamic_q = False` holds q constant, and `fairrfl_lipschitz` fixes the q-FFL constant instead of estimating it from the local SGD steps. Move or rename the existing dataset folder before rerunning with either change. The optional diagnostic cell shows the global update size alongside the median local update size. A very small global step can reveal stalled aggregation even when local training loss falls. It is a diagnostic, not a convergence test.

An attack does not guarantee FairRFL will be best or that any method will remain numerically stable. The framework stops on invalid values rather than generating ordinary-looking scores. The final plots display measured accuracy without adjusting or sorting points to favor a method. Read mean accuracy, worst-client accuracy and standard deviation together; a low spread alone can mean uniformly poor accuracy.

## Running and saving

The training cell runs each selected method for each dataset. It checks all planned configurations and existing folders before training. Matching completed results are reused; incomplete runs restart at round 1. There is no model/optimizer checkpoint resume.

Results are saved by dataset, then one RUN folder per seed (RUN1 is the first seed in `SEEDS`), then the scenario:

```text
Results/
├── mnist/
│   ├── RUN1/                         seed 42
│   │   ├── non_selfish/              config.selfishness = False
│   │   │   ├── fedavg.csv  fairrfl.csv  fcfl.csv  fedcda.csv  fedmut.csv
│   │   │   └── final_test_accuracy_all_methods.pdf
│   │   └── selfish/                  config.selfishness = True
│   │       └── (same files)
│   ├── RUN2/                         seed 43
│   ├── RUN3/                         seed 44
│   ├── summary_non_selfish.csv       Table II: mean ± sample std over the RUNs
│   └── summary_selfish.csv
└── smoke_mnist/                      same layout, SMOKE_TEST = True
```

- The settings cell prints every CSV path a run will write before training starts.
- On Colab the identical tree is kept on Drive under `MyDrive/FLFrameWork/Results/`, updated after every round and when each plot or summary is saved.
- The plot and summary cells read `<method>.csv` for the methods in `METHODS` from each seed folder and print every file they load. If the Colab disk was reset, they use the Drive copy and say so.
- Diagnostic PDFs go to `temp/diagnostics/` with the same sub-folders, on the runtime's local disk.

Repeating a plotting cell replaces that plot's PDF.

On Colab, the current CSV is copied after each fully evaluated round and when training exits normally or with a Python exception. A hard runtime termination can lose work since the latest successful copy. A copied partial CSV is useful for diagnostics but is rejected by final analysis. Backup failures print the local path so you can retrieve it. Local use does not mount Drive.

After reconnecting, the runner restores missing method CSVs from the matching Drive folder. To generate final figures without training again, run the setup, storage and settings cells, then the plotting cell directly. Use the same settings that produced the CSVs. All selected methods must exist and have complete results. Diagnostic plots can inspect incomplete runs.

For older experiments with folders such as `Results_GColab/experiment_1_cifar10`, use the final two cells in `Analysis.ipynb` and set the folder and filenames there. Its earlier sections retain the average accuracy and loss plots. Analysis reads the settings in the CSVs, not today's configuration, and rejects mismatched scenarios or missing clients/rounds.

## GitHub synchronization

The setup notebook uses `https://github.com/s4meer-git/BPDC_FL_Framework` on its checked-out branch. Local changes reach remote runtimes only after you commit and push them. Stop training, restart the remote Python session, and rerun setup to fetch new code. Pulling files does not replace already imported functions or change a running model. Reopen the notebook from GitHub to get changes to notebook cells themselves.

The setup cell will not pull code into a session that already imported the training configuration. It asks you to restart to avoid mixing old and new functions. A conflicting local Git edit causes the pull to stop rather than discard that edit.

## Existing method adaptations

All methods use the shared local optimizer. FCFL measures its unfairness gaps on each client's training data, so client test data are only used for evaluation. FedMut on CIFAR-10 retains gradient clipping at norm 10; other methods and other datasets do not use that safeguard. LoGoFair is not included because it needs a binary task with a sensitive attribute. See [method references](../methods/REFERENCES.md) for details and limitations.
