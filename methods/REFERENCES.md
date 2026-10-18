# Method references

Use `JPNotebooks/setup_remote.ipynb` to select methods and datasets. The notebook
creates the configuration and calls the training engine. The framework
compares FedAvg, FairRFL, FedCDA, FedMut and FCFL under the common setup in
`Fairness_FL_Exp_setUp.pdf`: 30 clients, 40 rounds, five local epochs, SGD with
learning rate 0.01, momentum 0.5 (MNIST) or 0.9 (CIFAR-10, Tiny ImageNet),
Dirichlet alpha 0.5 and three seeds. Every method uses the shared local trainer
(`participants/train.py`), the same partitions and the same initial model for a
seed; only the aggregation, client-selection or fairness mechanism differs.

Client accuracy (Eq. 1 of the setup) uses "the test set associated with client
i". The setup fixes the Dirichlet split for training data but not how the test
set is divided, so each client's test set follows its own training class mix:
for every class, a client receives the same share of that class's test images
as it holds of its training images (`config.test_partition = "matched"`). The
earlier code drew a second, independent Dirichlet split for testing; on MNIST
its class mix differed from each client's training mix by a total variation of
about 0.55, against about 0.01 now. Every test image still belongs to exactly
one client, global accuracy still uses the full test set, and
`test_partition = "dirichlet"` restores the earlier split.

## FairRFL

Sources: Augello et al., "FairRFL: Fair and Robust Federated Learning in the
Presence of Selfish Clients" ([accepted PDF](https://sites.unipa.it/networks/ndslab/pdf/0274.pdf)),
Algorithm 3 (Dq-FFL) and the recovery rule; and the
[official ndslab-group/FairRFL repository](https://github.com/ndslab-group/FairRFL),
commit `806331c970d90034f950653eef3e4c1f89576ab1`.
Unmodified `utils.py`, `selfishness.py`, and `experiments.py` are saved in
`.reference/fairrfl`.

Retained from the authors: q-FFL update and normalizer formulas, dynamic q
(`q_i = q * l_i / l_med` from the previous round), `tau = 2.5`, the median/MAD
detection and the ten-step recovery search applied to the weighted updates.
Zero median absolute deviation uses an explicit comparison: only norms above
the median are flagged, matching the original division's decisions without
divide-by-zero warnings.

Three corrections were needed to run it under the common setup:

1. **Client loss.** Algorithm 3 weights client `i` by `F_i(w[k])`, the loss of
   the received global model. The released code instead sums every minibatch
   loss during local training. In the authors' experiments every client held
   200 samples (one batch per epoch), so the sum was only a constant multiple.
   Under the Dirichlet split, the sum grows with the client's batch count: the
   previous CIFAR-10 run had summed losses from 85 to 408, raised to dynamic
   exponents from 0.45 to 2.16, so weights spanned about five orders of
   magnitude. Recovery then clipped the heavily weighted clients while their
   normalizers kept the denominator large; by round 40 the recovered updates
   kept about 0.4% of their magnitude and accuracy stayed near 20%. The adapter now measures the mean
   cross-entropy of the received global model on the client's training data,
   in evaluation mode, before local training, as in the paper.
2. **L.** q-FFL treats `L (w - w_local)` as one gradient and uses `L = 1/lr`.
   That holds for the authors' single-batch epochs with an optimizer reset every
   batch, but not for five epochs of momentum SGD: the `||L Δ||^2` term would
   dominate and reduce every step to 2–6% of a FedAvg step. With `S` local steps
   at learning rate `η` and momentum `β`, `Δ ≈ η S / (1 − β) · ∇F`, so the
   adapter uses `L = (1 − β) / (η · mean S)`. This reduces to the authors'
   `1/lr` when `S = 1` and `β = 0`. Set `fairrfl_lipschitz` to a number to fix L;
   `1 / lr` reproduces the released formula exactly.
3. **Optimizer.** The released code recreates SGD every minibatch with fixed
   momentum 0.9 and weight decay 5e-4. That is a training condition rather than
   the fairness mechanism, so FairRFL now uses the shared optimizer like every
   other method.

With `L = 1/lr` and the same losses, the aggregation equals the previous
implementation to float32 rounding; with `q = 0` it reduces to the plain mean.
A three-round MNIST check with 30 clients reached 43.0, 71.5 and 78.7% (FedAvg:
58.1, 75.0 and 81.7%), with no honest client flagged. The global step was about
60% of the median local step, instead of being suppressed.

FairRFL carries learned BatchNorm buffers back to the server. Running means and
variances are averaged using client sample counts; `num_batches_tracked` uses
the largest local counter. This is an explicit adaptation, not part of the
paper's parameter-update formula.

## FedCDA

Source: the authors' `FedCDA_code/FedAvgNonIIDSplitAvg_vanilla` directory in the
[ICLR 2024 supplement](https://proceedings.iclr.cc/paper_files/paper/2024/file/a559a5a8aa5ae6682ced009ad97cdb16-Supplementary-Conference.zip).
[Paper page](https://proceedings.iclr.cc/paper_files/paper/2024/hash/a559a5a8aa5ae6682ced009ad97cdb16-Abstract-Conference.html).
Unmodified reference files are saved in `.reference/fedcda`.
The downloaded archive SHA-256 is
`909c05f8709e837a2ff5659dfb9c0af2396bf771ce7d831f90621c3fad05b810`.

The adapter keeps FIFO model histories, warmup and equal client aggregation.
The score implements Equation (5): mean cached local loss plus
`fedcda_smoothness / 2` times mean squared distance from the selected models'
parameter average. Distances include model parameters only. Each local model's
loss is measured after training in a forward-only pass over its training data.
The search retains its best score (the released recursive search could keep the
last candidate instead). Group search follows Equation (8): inactive clients and
earlier groups are included, while later groups enter when their turn arrives.

The objective is evaluated from dot products: `sum ||w_i − c||^2 = sum ||w_i||^2 −
||sum w_i||^2 / n`, so one pass over the parameters per group gives every
candidate's score. The previous version rebuilt all cached models in float64
for each of the 3^5 candidates per group, about 10^11 operations per CIFAR-10
round and eight times more for ResNet-18. Choices and scores match the previous
search in randomized checks.

The author memory length of 3 and group size of 5 are retained, with all clients
participating. The authors warm up for 50 of 200 rounds; `fedcda_warmup = 10`
keeps that quarter of a 40-round run before history search begins (the smoke
test uses 1). `fedcda_smoothness = 1.0` is a tunable coefficient for L, not a
value supplied by the released code.

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
for odd client counts. Author defaults retained: `fedmut_radius = 4.0`,
`fedmut_acc_rate = 0.3` and `fedmut_bound = 50`, with full participation.

All state entries, including BatchNorm buffers, follow the author arithmetic.
Mutated running statistics only seed local training, where BatchNorm uses batch
statistics; evaluation always uses the averaged model.

For CIFAR-10 only, local FedMut training clips the combined gradient L2 norm to
`config.fedmut_cifar_grad_norm = 10.0` before SGD. This stability safeguard was
added after float32 overflow on the earlier CIFAR-10 CNN without BatchNorm; it
clipped 0.36% of batches in the completed run. Set it to `None` for the original
unclipped trainer. Results using it should disclose the safeguard.

## FCFL

Source: Wang et al., "FCFL: A Fairness Compensation-Based Federated Learning
Scheme with Accumulated Queues", ECML-PKDD 2024, and the authors'
[wlffffff/FCFL repository](https://github.com/wlffffff/FCFL), commit
`70fe7f613213ec09f93096d850b28341fbd0baed` (`main_fcfl.py`,
`utils/FCFL_options.py`, `models/qFed.py`).

Each round the server measures the global model's accuracy on every client and
computes the unfairness `uf_i = max(Acc_global − acc_i, 0)`. The accumulated
queue is `Q_i(t) = max(Q_i(t−1) + alpha · uf_i − p_i(t−1), 0)`, where `p_i` is the
client's previous aggregation weight (0 when not selected). The clients with
the largest queues are selected (`sample_by_Q_top`) and aggregated with weights
`Q_i / sum Q`. The first round, or all-zero queues, use data-volume weights.
`Acc_global` for the next round is the aggregation-weighted accuracy of the
trained local models on their own training data (`acc_global_estimate`).
Selection, queue arithmetic and `weight_agg` match the authors' functions in
randomized checks, including the order of their random tie-break draw.

Adaptations: the released code measures `acc_i` on a held-out 10% of each
client's own partition and trains on 80%. Here each client's test partition is
reserved for evaluation, so `acc_i` is measured on the client's training data;
this keeps the common partition and avoids using test data for training
decisions. The released first-round weights are indexed by `range(m)` instead
of the selected client IDs; the adapter uses the selected clients' own sizes.
`fcfl_alpha = 0.5` is the authors' default. Their participation default is 10%
of clients; `fcfl_fraction = 1.0` keeps every client training every round, as
the common setup requires. Set it to `0.1` for the authors' participation.

## LoGoFair

Source: Zhang et al., AAAI 2025, [liizhang/LoGofair](https://github.com/liizhang/LoGofair),
commit `7044815cf813cdad53fca7bab426c4d2194ab505`.

Not integrated. LoGoFair post-processes the scores of a binary classifier to
satisfy demographic parity or equalized odds between two groups of a sensitive
attribute (for example, sex in Adult). Its client and server code require a
binary label `Y`, a binary sensitive attribute `A` and per-group statistics.
MNIST, CIFAR-10 and Tiny ImageNet are multi-class and have no sensitive
attribute, and the setup measures client-level accuracy fairness rather than
group fairness. Running it here would require inventing an attribute and
changing the task, so it would no longer be the authors' method.


## Experiment settings and verification

The settings differ from each author's complete experiments, so this
integration does not claim to reproduce their reported accuracy. Round 1 resets
each method's state. Focused checks in `temp/` cover the author comparisons
above, identical initial weights across methods, the seed summary, and complete
logging for every method with and without selfish clients.


## Selfish-client simulation

The switch implements Eq. (5) of the paper with the attacker's share
`s = w_s / Gamma` of the aggregate. For current legitimate outgoing update `u`,
observed global change `d`, previous transmitted update `p`, previous share
`s_prev` and strength `phi`:

```text
others = d - s_prev * p
sent = (phi * u + (1 - phi) * others / (1 - s) - others) / s
```

The aggregate then moves a fraction `phi` toward the attacker's own update, as
in the paper. With equal shares `s = 1/n` this is exactly the known-client-count
branch of `selfish_training` in the
[authors' selfishness.py](https://github.com/ndslab-group/FairRFL/blob/806331c970d90034f950653eef3e4c1f89576ab1/selfishness.py)
(checked numerically). FedCDA and FedMut average equally and use `1/n`; FedAvg
uses each client's sample share and FCFL its queue weight. The earlier version
used `1/n` everywhere, which mis-scaled attackers under sample-weighted FedAvg.
FairRFL keeps the authors' own experiment: the known-count attack applied to
its fairness-weighted vectors. The attacker estimates other clients'
contribution from its own previous transmission and the two latest global
models; it does not receive their individual updates.

The framework explicitly stores each selfish client's actual prior outgoing
update before server recovery. FairRFL's outgoing vector already includes
fairness weighting, as in the authors' experiments. FedMut's local model starts
from its assigned mutation slot, while the attack estimates its outgoing
displacement relative to the common global model. For FedCDA, the local loss
remains the honestly measured loss; loss-report manipulation is not included.
Only parameters are modified; BatchNorm buffers are left intact.

By default the first two rounds are honest and strength is 0.7. The paper
studies one selfish client and 15 of 50; `selfish_fraction = 0.05` gives one
selfish client among 30, and `0.3` gives nine. As the paper notes, several
selfish clients compete and cancel each other; with nine, FedAvg and the other
unprotected methods can diverge. Full participation is
required. Selfishness disabled is a no-op and does not consume extra training
randomness. `fairrfl_dynamic_q = False` holds q constant; results from changed
method settings must use a different comparison name. No setting guarantees a
particular accuracy or fairness ranking.
