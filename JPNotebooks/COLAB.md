# Running this project on Google Colab

Open `JPNotebooks/Colab.ipynb` at https://colab.research.google.com using **File → Upload notebook**. Select **Runtime → Change runtime type → GPU** before running the cells in order.

## Upload the prepared ZIP

Leave `REPOSITORY_URL = ""` in the source cell. When its upload picker opens, select `FLFrameWork-Colab.zip` from your Desktop. This archive contains the Python source and notebooks; it excludes your virtual environment, datasets and saved experiments.

The setup cell installs the notebook dependencies and keeps Colab's existing PyTorch/torchvision installation. Approve the Drive connection when Colab asks. The launcher creates `MyDrive/FLFrameWork` for saved results.

In the settings cell, choose `mnist`, `cifar10` or `tinyimagenet`, and one of the five methods. Start with 3 rounds to check runtime, then set the intended experiment values. Use a new comparison name when changing settings. The launcher starts with 31 clients, 40 rounds and 5 local epochs; your normal Mac configuration is separate.

MNIST and CIFAR-10 download automatically during the run. The dataset cell downloads Tiny ImageNet from Stanford when needed. Images and the active CSV stay on Colab's local disk to avoid slow per-batch Drive access.

Run the training cell for one method. After it finishes, change the method in the settings cell and rerun from that cell onward. Keep the same dataset, experiment settings and comparison name to collect methods together.

The analysis cell selects the three plotting cells for the current dataset from `Analysis.ipynb`. It reads all method CSVs in the selected comparison and saves those figures without running the other datasets' sections.

For local analysis, `Analysis.ipynb` has separate Tiny ImageNet, MNIST and CIFAR-10 sections. Each has an editable loading cell followed by accuracy, loss-per-round and client-loss comparison cells. Run the section whose CSVs are available; an unrun dataset's section reports that its folder has no CSVs. Each section checks the recorded dataset so different datasets cannot be mixed accidentally.

## Which configuration is used

The launcher creates `config = configuration()`, then overrides experiment settings in its settings cell. The class still supplies device selection, method dispatch and other defaults. Values explicitly set in the notebook take priority: if `configuration.py` says 3 rounds but the launcher assigns 40, Colab runs 40. The recorded CSV settings describe the actual run. The analysis uses those recorded settings rather than today's Python configuration.

## Where results are saved

For `COMPARISON_NAME = "experiment_1"` and MNIST:

- CSVs: `MyDrive/FLFrameWork/Results/experiment_1_mnist/fedavg.csv`, and similarly for other methods.
- PDFs: `MyDrive/FLFrameWork/CellResults/experiment_1_mnist/result_1/`, then `result_2/`, and so on.

A method rerun replaces its CSV, following the repository's existing behavior. Earlier methods and numbered figure folders are restored from Drive when you reconnect.

The training cell copies the CSV when it returns or Python reports an error/interruption. A forcibly terminated or disconnected runtime may not execute that copy. There is no checkpoint/resume support: a CSV records results, not the model, optimizer or method state. An incomplete CSV is rejected by the analysis.

If a Drive copy fails, the local CSV path is printed so it can be downloaded through Colab's Files panel before disconnecting.

## Keeping local code and Colab in sync through GitHub

The local project is connected to `https://github.com/s4meer-git/BPDC_FL_Framework` on branch `main`. The Colab source cell already contains this URL. It clones the repository on first use and runs `git pull --ff-only` on later setup runs.

After local fixes, commit and push the changed files from your Mac, for example:

```sh
git add configuration.py
git commit -m "Adjust experiment settings"
git push origin main
```

Stage the files you actually changed; Git does not upload edits until you commit and push them. Stop training, restart the Colab Python session and rerun setup to load the new code. Git updates files; it does not replace functions already loaded into Python or update a model already training. Restarting discards that training session's in-memory state.

Keep Colab source files unchanged when using this workflow. A pull stops if local edits conflict or the branches diverge. If switching from an uploaded ZIP to GitHub, start a fresh runtime so the clone destination is empty. You can also open the notebook itself from GitHub through Colab's **File → Open notebook → GitHub** tab.

## Method settings to keep explicit

- FairMedian needs an odd client count and uses one minibatch per client; `local_epochs` does not apply to it.
- FedCDA's default 50 warmup rounds mean a 40-round run never searches histories. Set warmup below the run length explicitly if you want to test the corrected search.
- FairRFL retains the authors' hardcoded local momentum of 0.9, even when the general MNIST setting is 0.5.
- Configuration selects CUDA when available, otherwise MPS. The MPS availability guard remains commented out as requested. The Colab launcher's GPU check prevents starting there without CUDA.

Colab's free GPU availability and session length vary, and free sessions can last at most 12 hours. See the [official Colab FAQ](https://research.google.com/colaboratory/faq.html) for limits, Drive access and importing notebooks.
