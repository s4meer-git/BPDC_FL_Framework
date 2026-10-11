# Method references

Set `self.method` in `configuration.py` to `"fedcda"` or `"fedmut"`, then run
`.venv/bin/python main.py`. The current defaults select FedAvg with
3 clients and 3 rounds. FairMedian requires an odd client count; the other methods do not.

## FairRFL

Source: the [official ndslab-group/FairRFL repository](https://github.com/ndslab-group/FairRFL),
commit `806331c970d90034f950653eef3e4c1f89576ab1`.
Unmodified `utils.py`, `selfishness.py`, and `experiments.py` are saved in
`.reference/fairrfl` for the existing author-comparison tests.

The adapter retains the authors' dynamic q, summed minibatch losses, per-batch
SGD optimizer construction, fixed momentum/weight decay, parameter-update
formulas, and ten-step recovery search. Zero median absolute deviation now
uses an explicit comparison: only norms above the median are flagged, matching
the original division's decisions without divide-by-zero warnings. A zero
previous median loss falls back to the median of positive previous losses (or
`1.0` when all are zero) to keep dynamic q defined without extreme exponents.
This is a numerical edge-case correction; nonzero-loss formulas are unchanged.

## FairMedian

Source: Chen et al., [Distributed Training with Heterogeneous Data: Bridging
Median- and Mean-Based Algorithms, NeurIPS 2020](https://papers.neurips.cc/paper_files/paper/2020/hash/f629ed9325990b10543ab5946c1362fb-Abstract.html),
Algorithm 4 (noisy MEDIANSGD). `fairmedian` is this framework's name for the
adaptation. Each client contributes one minibatch gradient; independent
Gaussian noise is added before taking the coordinate-wise median, followed by
one server step. Local epochs and sample weighting do not apply.

Both FairRFL and FairMedian now carry learned BatchNorm buffers back to the
server. Running means and variances are averaged using client sample counts;
`num_batches_tracked` uses the largest local counter. This is an explicit ResNet
adaptation, not part of either paper's parameter-update formula. Other buffers
keep their existing global values. The resulting statistics are an average of
local running estimates, not an exact pooled-data recalibration.

## FedCDA

Source: the authors' `FedCDA_code/FedAvgNonIIDSplitAvg_vanilla` directory in the
[ICLR 2024 supplement](https://proceedings.iclr.cc/paper_files/paper/2024/file/a559a5a8aa5ae6682ced009ad97cdb16-Supplementary-Conference.zip).
[Paper page](https://proceedings.iclr.cc/paper_files/paper/2024/hash/a559a5a8aa5ae6682ced009ad97cdb16-Abstract-Conference.html).
Unmodified reference files are saved in `.reference/fedcda` for tests.
The downloaded archive SHA-256 is
`909c05f8709e837a2ff5659dfb9c0af2396bf771ce7d831f90621c3fad05b810`.

The adapter keeps FIFO model histories, warmup and equal client aggregation.
The recursive search now always retains its best score. The old
`fedcda_fix_search` switch has been removed so normal runs cannot select the
known broken search.

The score now implements Equation (5): mean cached local loss plus
`fedcda_smoothness / 2` times mean squared distance from the selected models'
parameter average. Distances include model parameters only, excluding running
statistics and counters. Each local model's loss is measured after training in
an additional forward-only pass over that client's training loader, with the
model in evaluation mode. Data augmentation still follows the training loader.
This is an empirical estimate of local risk for the fixed trained model, not
an average of losses measured while its parameters were changing.

`fedcda_smoothness = 1.0` is an explicit tunable coefficient for L; it is not an
estimated or guaranteed smoothness bound for this network, nor a value supplied
by the authors' released code. Losses and models are evicted together. Group
search follows Equation (8): inactive clients and earlier groups are included,
while later groups enter when their turn arrives. The final model averages the
chosen states of all clients seen so far.

Author defaults retained: `fedcda_memory_k = 3`, `fedcda_warmup = 50`, and
`fedcda_batch_client_num = 5`. `fedcda_clients_per_round = None` uses all clients.
With the current three rounds, only warmup runs. Lower warmup explicitly for a
short search experiment or run more than 50 rounds.

These corrections deliberately differ from the released vanilla code's
unsquared per-state norms and recursive search bug. They do not imply a
reproduction of published accuracy. Cached histories still live on CPU, and
search remains exponential in the number of clients in a search group.

## FedMut

Source: [HMHelloWorld/FedMut](https://github.com/HMHelloWorld/FedMut), commit
`7716db06d2d72e56ab678818f9df13ae8949ae6f`.
Unmodified `Algorithm/Training_FedMut.py`, `models/Fed.py`, `utils/options.py`, and
the MIT license are saved locally in `.reference/fedmut`.
The published project also retains the license in `LICENSE.FedMut`.

The adapter follows the SGD branch, equal-weight `Aggregation(w_locals, None)`,
global-model differences, and `mutation_spread`. Each state entry receives its
own shuffled pairs of positive and negative mutation controls. Acceleration
decays with the zero-based round index. The last population slot is unmutated
for odd client counts. Slots are assigned to newly sampled clients each round;
they are not persistent personalized models for fixed client IDs.

Author mutation defaults retained: `fedmut_radius = 4.0`, `fedmut_acc_rate = 0.3`,
and `fedmut_bound = 50`. `fedmut_fraction = 1.0` retains full participation;
set it to `0.1` for the authors' participation default. The published unused
`min_radius` option and diagnostic update norm are omitted because neither
changes the released training algorithm.

All state entries, including BatchNorm buffers, follow the author arithmetic.
Integer buffers may become floating point in the population and are cast by
`load_state_dict` when loaded, just as in the original implementation.
Mutation can extrapolate running statistics; the adapter does not clamp them.
The model interface uses this framework's tensor logits instead of the authors'
`{"output": logits}` dictionary.

## Experiment settings and verification

Both methods retain the framework's dataset, architecture, learning rate,
momentum, batch size, local epochs, seed, and MPS requirement. These settings
differ from the authors' complete experiments, so this integration does not
claim to reproduce their reported accuracy. Round 1 resets each method's state.

The original test folder was moved to `/Users/sameer/Desktop/FLFrame9oct/tests`.
Archived tests that assert frozen BatchNorm buffers or the original broken
FedCDA search describe old behavior and must not be used as the expected result
for these corrections. Focused numerical checks cover the revised objective,
search, buffer aggregation, invalid predictions and notebook completeness.
