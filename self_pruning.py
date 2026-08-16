"""
Self-Pruning Neural Network
============================

A feed-forward image classifier for CIFAR-10 built from a custom
`PrunableLinear` layer. Each layer learns a set of `gate_scores` (one per
weight) alongside the usual weight/bias parameters. During the forward
pass the weights are multiplied by `sigmoid(gate_scores)` before being
used, so the network can learn to suppress individual connections by
driving their gate toward zero, instead of relying on a fixed pruning
schedule applied after training.

IMPORTANT WORDING NOTE (read before editing or presenting this code):
`sigmoid(x)` is strictly between 0 and 1 for every finite `x` -- it never
outputs an exact 0. So the L1 sparsity penalty encourages gate values
*toward* zero, not *to* zero. Everywhere in this file, in the README, and
in report.md, "pruned" means "gate value below `sparsity_threshold`
(default 1e-2)", not "gate value is mathematically exactly zero." Do not
describe gates as becoming exactly zero.

Run this file directly to train the model at several sparsity-pressure
(lambda) values, record test accuracy / sparsity for each, save a
results table, and plot the final gate-value distribution of the
selected best model.

    python self_pruning.py

See README.md for setup instructions and report.md for a discussion of
the results.
"""

import os
import csv
import random
import argparse
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------
# All the hyperparameters that matter for the experiment live here, in one
# place, instead of being scattered through the script.

@dataclass
class Config:
    # Data / training
    data_dir: str = "./data"
    results_dir: str = "./results"
    batch_size: int = 128
    epochs: int = 10
    learning_rate: float = 1e-3

    # Architecture (flattened 3x32x32 CIFAR-10 image -> hidden -> hidden -> 10)
    input_dim: int = 3 * 32 * 32
    hidden_dim_1: int = 512
    hidden_dim_2: int = 256
    num_classes: int = 10

    # Pruning
    # NOTE on scale: sparsity_loss() sums gate values across ALL weights in
    # the network (roughly 1.7M for this architecture: 3072*512 + 512*256 +
    # 256*10 gates), so it starts out on the order of 1e6, since gates
    # initialize near sigmoid(0.5) ~= 0.62. Classification loss
    # (cross-entropy) is O(1-2). For lambda to act as a meaningful, gradually
    # increasing pressure rather than instantly swamping the classification
    # signal, it needs to be small relative to 1 / (number of gate
    # parameters). The values below were chosen with that scale in mind: low
    # (barely visible pressure), medium, and high (stronger pruning
    # pressure). These are starting points, not tuned/verified results --
    # if a real run shows they don't produce a meaningful accuracy/sparsity
    # trade-off, widen the range (e.g. add 1e-3 or 1e-7) and re-run. The
    # list is fully configurable, either by editing this default or via
    # `python self_pruning.py --lambdas 1e-6 1e-5 1e-4 1e-3`.
    lambda_values: list = field(default_factory=lambda: [1e-6, 1e-5, 1e-4])

    # A gate value below this is treated as "pruned." This is a practical
    # cutoff, not a claim that sigmoid produced an exact zero -- see the
    # module docstring above.
    sparsity_threshold: float = 1e-2

    # Used only by select_best_model(): how close (in absolute test-accuracy
    # units, e.g. 0.02 = 2 percentage points) a candidate model's accuracy
    # must be to the single best accuracy observed across all lambda runs
    # to be considered "comparable" for the purposes of preferring sparsity.
    # See select_best_model() docstring for the full rule.
    accuracy_tolerance: float = 0.02

    # Misc
    seed: int = 42
    num_workers: int = 2


# ---------------------------------------------------------------------------
# 2. REPRODUCIBILITY
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed every RNG we touch so results are as reproducible as possible.

    Note: even with all these seeds set, exact bit-for-bit reproducibility on
    GPU is not guaranteed, because some CUDA/cuDNN kernels use
    non-deterministic algorithms for performance reasons unless you also set
    torch.use_deterministic_algorithms(True) and disable cuDNN autotuning
    (both of which can slow training down noticeably). We do not force full
    determinism here for that reason, but the seeding below removes the vast
    majority of run-to-run variance. This applies on any machine/environment
    the script is run on -- it is not specific to any particular GPU.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# 3. CUSTOM LAYER: PrunableLinear
# ---------------------------------------------------------------------------

class PrunableLinear(nn.Module):
    """A drop-in replacement for nn.Linear that can learn to prune itself.

    Alongside the usual `weight` (out_features x in_features) and `bias`
    (out_features,) parameters, this layer owns a `gate_scores` parameter
    with EXACTLY the same shape as `weight`. On every forward pass we turn
    the raw gate_scores into gates in the open interval (0, 1) with a
    sigmoid, and multiply them element-wise into the weight matrix before
    doing the linear transform:

        gates           = sigmoid(gate_scores)      # shape == weight.shape
        pruned_weight   = weight * gates             # element-wise
        output          = pruned_weight @ input + bias

    Because this whole computation is built out of standard differentiable
    ops (sigmoid, multiply, matmul/addmm via F.linear), autograd builds a
    graph that back-propagates gradients into `weight`, `gate_scores`, and
    `bias` simultaneously. No `.detach()` is used anywhere, so gate_scores
    always stays attached to the graph and receives real gradients.

    We do NOT subclass or wrap nn.Linear -- weight, bias, and gate_scores
    are all raw nn.Parameters that we manage ourselves, as required by the
    assignment.

    A note on "pruning": a gate near 0 makes a connection's contribution to
    the output near-zero, which is functionally equivalent to removing it,
    but sigmoid(x) is strictly greater than 0 for every finite x. The gate
    never becomes exactly 0 during ordinary training. "Pruned" in this
    codebase always means "gate value below `sparsity_threshold`," a
    practical, thresholded definition -- not a mathematical exact zero.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # --- weight & bias --------------------------------------------------
        # We use Kaiming (He) uniform initialization for the weight, which is
        # the same scheme nn.Linear uses internally. It is a good default for
        # layers that will be followed by a ReLU activation, because it keeps
        # the variance of activations roughly stable across layers.
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        fan_in = in_features
        bound = 1 / (fan_in ** 0.5)
        nn.init.uniform_(self.bias, -bound, bound)

        # --- gate_scores -----------------------------------------------------
        # gate_scores is the raw, unconstrained parameter the optimizer
        # actually updates; sigmoid(gate_scores) is what we use as the gate.
        # We initialize gate_scores to a small POSITIVE constant (not 0) so
        # that sigmoid(gate_scores) starts close to, but above, 0.5. This
        # means every connection starts out "mostly open" (gate ~= 0.62),
        # so the network is not crippled at the very start of training, but
        # gates are still free to move down toward 0 (pruned, per the
        # threshold) or up toward 1 (kept) as training progresses and the L1
        # sparsity penalty pushes on them. Starting at exactly 0 would also
        # work (gate = 0.5), but a slightly positive value gives the
        # classification loss a small head start before pruning pressure
        # dominates.
        self.gate_scores = nn.Parameter(torch.full((out_features, in_features), 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_scores)   # (out, in), each entry strictly in (0, 1)
        pruned_weight = self.weight * gates         # element-wise gating
        return F.linear(x, pruned_weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


# ---------------------------------------------------------------------------
# 4. MODEL
# ---------------------------------------------------------------------------

class SelfPruningNet(nn.Module):
    """A 3-layer feed-forward classifier built entirely from PrunableLinear.

    CIFAR-10 images are 3x32x32 = 3072 values. We flatten each image into a
    single vector and pass it through two hidden PrunableLinear + ReLU
    blocks before a final PrunableLinear projects down to 10 class logits.

    A plain MLP (no convolutions) is a deliberate, assignment-appropriate
    choice here: the case study asks for a "feed-forward neural network",
    and the point of the exercise is to demonstrate gated/learnable
    pruning, not to chase state-of-the-art CIFAR-10 accuracy. Two hidden
    layers of size 512 and 256 give the network enough capacity to learn a
    non-trivial mapping (and therefore something meaningful for the gates
    to prune) while remaining fast to train multiple times (once per
    lambda) on a CPU.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.fc1 = PrunableLinear(cfg.input_dim, cfg.hidden_dim_1)
        self.fc2 = PrunableLinear(cfg.hidden_dim_1, cfg.hidden_dim_2)
        self.fc3 = PrunableLinear(cfg.hidden_dim_2, cfg.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # flatten (B, 3, 32, 32) -> (B, 3072)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # raw logits; CrossEntropyLoss applies softmax internally

    def prunable_layers(self):
        """Convenience accessor used by the sparsity loss / sparsity metric.
        gate_scores on fc1/fc2/fc3 are ordinary nn.Parameters owned by
        submodules, so they are automatically included in
        model.parameters() / model.named_parameters() -- nothing extra is
        needed to make the optimizer see them.
        """
        return [self.fc1, self.fc2, self.fc3]


# ---------------------------------------------------------------------------
# 5. LOSSES
# ---------------------------------------------------------------------------

def sparsity_loss(model: SelfPruningNet) -> torch.Tensor:
    """L1 penalty on the gate values of every PrunableLinear layer.

    Per the assignment spec, this is the SUM (not the mean) of gate values
    across all PrunableLinear layers in the model:

        SparsityLoss = sum(|sigmoid(gate_scores)|)   over every layer

    gates = sigmoid(gate_scores) is always in the open interval (0, 1),
    i.e. always non-negative, so sum(abs(gates)) == sum(gates). We still use
    .abs() to keep the code explicit about "this is an L1 penalty", but note
    that the abs() is a no-op here given the sigmoid range.

    Minimizing this term pushes gate values toward (not to) zero -- open
    connections cost loss, closed ones cost less -- which is what
    encourages sparsity. Plain CrossEntropyLoss has no mechanism that
    rewards small gate values, so without this term gates would simply
    drift to whatever value best fits the training data (typically staying
    open, i.e. near 1).

    We intentionally do NOT normalize/average this sum by the number of
    gates. The assignment specifies a raw sum, and normalizing it would
    change what a given lambda means; instead, lambda itself is chosen at a
    small enough magnitude to compensate for the large gate count (see the
    comment on Config.lambda_values).
    """
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    for layer in model.prunable_layers():
        gates = torch.sigmoid(layer.gate_scores)
        total = total + gates.abs().sum()
    return total


def total_loss_fn(logits, targets, model, lambda_value):
    """Total Loss = Classification Loss + lambda * Sparsity Loss."""
    cls_loss = F.cross_entropy(logits, targets)
    sp_loss = sparsity_loss(model)
    return cls_loss + lambda_value * sp_loss, cls_loss.item(), sp_loss.item()


# ---------------------------------------------------------------------------
# 6. DATA
# ---------------------------------------------------------------------------

def get_dataloaders(cfg: Config):
    """Build CIFAR-10 train/test DataLoaders. Downloads the dataset into
    cfg.data_dir automatically on first run (requires internet access to
    the CIFAR-10 host used by torchvision)."""
    # Standard CIFAR-10 per-channel mean/std used throughout the literature.
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=cfg.data_dir, train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=cfg.data_dir, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# 7. TRAIN / EVAL LOOPS
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, lambda_value):
    """One full pass over `loader`: for every batch, load it, move it to
    `device`, zero grads, forward pass, compute classification + sparsity +
    total loss, backpropagate, and step the optimizer (which updates
    weight, bias, AND gate_scores, since all three are ordinary parameters
    of the model)."""
    model.train()
    running_total, running_cls, running_sp = 0.0, 0.0, 0.0
    correct, seen = 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss, cls_val, sp_val = total_loss_fn(logits, labels, model, lambda_value)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_total += loss.item() * batch_size
        running_cls += cls_val * batch_size
        running_sp += sp_val * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += batch_size

    return {
        "loss": running_total / seen,
        "cls_loss": running_cls / seen,
        "sparsity_loss": running_sp / seen,
        "accuracy": correct / seen,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, seen = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += labels.size(0)
    return correct / seen


@torch.no_grad()
def compute_sparsity(model, threshold: float = 1e-2):
    """Fraction of gate values below `threshold`, across every
    PrunableLinear layer in the model, expressed as a percentage.

        gates   = sigmoid(gate_scores)      # for each PrunableLinear layer
        pruned  = gates < threshold
        sparsity_pct = 100 * (# pruned gates) / (# total gates)

    We deliberately look at gate values, NOT raw weight values -- a weight
    can be numerically large while its gate is near zero, and it is the
    gate, not the weight, that decides whether the connection is
    effectively "on" in this architecture. This function walks every layer
    returned by model.prunable_layers(), so it always covers fc1, fc2, and
    fc3.
    """
    below, total = 0, 0
    for layer in model.prunable_layers():
        gates = torch.sigmoid(layer.gate_scores)
        below += (gates < threshold).sum().item()
        total += gates.numel()
    return 100.0 * below / total


@torch.no_grad()
def collect_gate_values(model):
    """Flatten every gate value in the model into a single 1D numpy array,
    for histogram plotting. Always computed from the ACTUAL trained model
    passed in -- never synthesized."""
    values = []
    for layer in model.prunable_layers():
        gates = torch.sigmoid(layer.gate_scores)
        values.append(gates.flatten().cpu().numpy())
    return np.concatenate(values)


@torch.no_grad()
def compute_gate_diagnostics(model, threshold: float = 1e-2) -> dict:
    """DIAGNOSTIC ONLY -- does not define, replace, or influence the official
    sparsity metric. The official metric stays exactly compute_sparsity():
    percentage of gates < 1e-2. This function exists only to show *why*
    that percentage can be 0% even while sparsity_loss (the raw sum of
    gates) is falling: is the whole gate distribution drifting down but
    still sitting above 1e-2, or is something else happening?

    Computed across every PrunableLinear layer (fc1, fc2, fc3) of the
    ACTUAL model passed in -- nothing here is synthesized or estimated.
    """
    gate_values, gate_scores = [], []
    for layer in model.prunable_layers():
        gate_values.append(torch.sigmoid(layer.gate_scores).flatten().cpu().numpy())
        gate_scores.append(layer.gate_scores.detach().flatten().cpu().numpy())
    gate_values = np.concatenate(gate_values)
    gate_scores = np.concatenate(gate_scores)

    return {
        "total_gates": int(gate_values.size),
        "sum_gate_values": float(gate_values.sum()),
        "mean_gate_value": float(gate_values.mean()),
        "min_gate": float(gate_values.min()),
        "max_gate": float(gate_values.max()),
        "median_gate": float(np.median(gate_values)),
        "p1_gate": float(np.percentile(gate_values, 1)),
        "p5_gate": float(np.percentile(gate_values, 5)),
        "p10_gate": float(np.percentile(gate_values, 10)),
        "pct_below_0.1": float(100.0 * np.mean(gate_values < 0.1)),
        "pct_below_0.05": float(100.0 * np.mean(gate_values < 0.05)),
        "pct_below_0.01": float(100.0 * np.mean(gate_values < threshold)),  # == compute_sparsity()
        "pct_below_0.005": float(100.0 * np.mean(gate_values < 0.005)),
        "pct_below_0.001": float(100.0 * np.mean(gate_values < 0.001)),
        "min_gate_score": float(gate_scores.min()),
        "max_gate_score": float(gate_scores.max()),
        "mean_gate_score": float(gate_scores.mean()),
    }


def print_gate_diagnostics(lambda_value, d: dict) -> None:
    """Pretty-print the dict returned by compute_gate_diagnostics(). Purely
    a reporting helper -- computes nothing itself."""
    print(f"\n  --- Gate distribution diagnostics (lambda={lambda_value}) [DIAGNOSTIC ONLY] ---")
    print(f"  total gates                 : {d['total_gates']}")
    print(f"  sum of gate values          : {d['sum_gate_values']:.2f}")
    print(f"  mean gate value             : {d['mean_gate_value']:.6f}")
    print(f"  min / max gate              : {d['min_gate']:.6f} / {d['max_gate']:.6f}")
    print(f"  median gate                 : {d['median_gate']:.6f}")
    print(f"  1st / 5th / 10th pctile     : {d['p1_gate']:.6f} / {d['p5_gate']:.6f} / {d['p10_gate']:.6f}")
    print(f"  % gates < 0.1               : {d['pct_below_0.1']:.4f}%")
    print(f"  % gates < 0.05              : {d['pct_below_0.05']:.4f}%")
    print(f"  % gates < 0.01  (OFFICIAL)  : {d['pct_below_0.01']:.4f}%   <- identical to compute_sparsity()")
    print(f"  % gates < 0.005             : {d['pct_below_0.005']:.4f}%")
    print(f"  % gates < 0.001             : {d['pct_below_0.001']:.4f}%")
    print(f"  min / max / mean gate_score : {d['min_gate_score']:.4f} / {d['max_gate_score']:.4f} / {d['mean_gate_score']:.4f}")


def inspect_gate_gradients(model: SelfPruningNet, images: torch.Tensor, labels: torch.Tensor,
                            lambda_value: float) -> dict:
    """DIAGNOSTIC ONLY -- decomposes the gradient that reaches gate_scores on
    ONE real batch into its two additive components:

        d(total_loss)/d(gate_scores) = d(cls_loss)/d(gate_scores)
                                      + lambda * d(sparsity_loss)/d(gate_scores)

    This is exact, not an approximation: total_loss is a linear combination
    of cls_loss and lambda*sparsity_loss, so autograd's gradients add
    linearly. We get each piece by backpropagating each loss term
    separately into gate_scores (zeroing grad between calls), then confirm
    they sum to the same gradient a single combined backward() would give.

    Call this with a REAL batch from a REAL (trained or mid-training) model
    to get real numbers. It does not modify weight/bias/gate_scores and
    does not step any optimizer -- it only reads .grad after backward().
    """
    model.zero_grad(set_to_none=True)
    logits = model(images)
    cls_loss = F.cross_entropy(logits, labels)
    sp_loss = sparsity_loss(model)

    # 1) classification-only contribution
    cls_loss.backward(retain_graph=True)
    cls_grad = {n: p.grad.detach().clone() for n, p in model.named_parameters() if "gate_scores" in n}
    model.zero_grad(set_to_none=True)

    # 2) sparsity-only contribution (already scaled by lambda, matching total_loss_fn)
    (lambda_value * sp_loss).backward(retain_graph=True)
    sp_grad = {n: p.grad.detach().clone() for n, p in model.named_parameters() if "gate_scores" in n}
    model.zero_grad(set_to_none=True)

    # 3) combined, exactly as train_one_epoch computes it -- for a sanity check
    total = cls_loss + lambda_value * sp_loss
    total.backward()
    total_grad = {n: p.grad.detach().clone() for n, p in model.named_parameters() if "gate_scores" in n}

    cls_abs = torch.cat([g.flatten() for g in cls_grad.values()]).abs()
    sp_abs = torch.cat([g.flatten() for g in sp_grad.values()]).abs()
    total_abs = torch.cat([g.flatten() for g in total_grad.values()]).abs()

    # confirm additivity: cls_grad + sp_grad should equal total_grad, per-element
    max_additivity_error = max(
        (cls_grad[n] + sp_grad[n] - total_grad[n]).abs().max().item() for n in total_grad
    )

    gate_scores_all = torch.cat([layer.gate_scores.detach().flatten() for layer in model.prunable_layers()])
    return {
        "mean_gate_score": float(gate_scores_all.mean()),
        "min_gate_score": float(gate_scores_all.min()),
        "max_gate_score": float(gate_scores_all.max()),
        "mean_gate": float(torch.sigmoid(gate_scores_all).mean()),
        "mean_abs_total_grad": float(total_abs.mean()),
        "mean_abs_cls_grad": float(cls_abs.mean()),
        "mean_abs_sparsity_grad": float(sp_abs.mean()),
        "max_additivity_error": float(max_additivity_error),
    }


def print_gate_gradient_inspection(lambda_value, d: dict) -> None:
    print(f"\n  --- Gate gradient decomposition (lambda={lambda_value}) [DIAGNOSTIC ONLY, ONE BATCH] ---")
    print(f"  mean gate_score              : {d['mean_gate_score']:.6f}")
    print(f"  min / max gate_score         : {d['min_gate_score']:.6f} / {d['max_gate_score']:.6f}")
    print(f"  mean gate                    : {d['mean_gate']:.6f}")
    print(f"  mean |total grad on gate_score|      : {d['mean_abs_total_grad']:.6e}")
    print(f"  mean |cls-loss grad on gate_score|   : {d['mean_abs_cls_grad']:.6e}")
    print(f"  mean |sparsity grad on gate_score|   : {d['mean_abs_sparsity_grad']:.6e}  (lambda-scaled, matches total_loss_fn)")
    print(f"  cls/sparsity grad ratio              : {d['mean_abs_cls_grad'] / d['mean_abs_sparsity_grad']:.4f}")
    print(f"  max |cls_grad+sparsity_grad-total_grad| (should be ~0, confirms additivity): {d['max_additivity_error']:.2e}")


# ---------------------------------------------------------------------------
# 8. SANITY + INTEGRATION CHECKS
# ---------------------------------------------------------------------------
# These checks do NOT require CIFAR-10 or internet access -- they use small
# random/synthetic tensors purely to validate that the code is wired
# correctly (shapes, gradient flow, optimizer updates, loop mechanics).
# They are not a substitute for the real experiment on real CIFAR-10 data.

def run_sanity_checks(cfg: Config, device: str) -> None:
    """Fast, isolated checks on the PrunableLinear layer and the full model.
    Covers assignment checklist items: parameter registration, gate_scores
    shape/range, forward pass, backward pass, gradient flow to gate_scores,
    and optimizer updates to gate_scores."""
    print("Running sanity checks on PrunableLinear / SelfPruningNet...")

    layer = PrunableLinear(16, 8).to(device)

    # 1. Parameters registered, and gate_scores has the same shape as weight.
    param_names = {name for name, _ in layer.named_parameters()}
    assert {"weight", "bias", "gate_scores"} <= param_names, "Missing expected parameters"
    assert layer.gate_scores.shape == layer.weight.shape, "gate_scores shape must match weight shape"

    # 2. Gates lie strictly in (0, 1) -- never exactly 0 or 1 for finite gate_scores.
    gates = torch.sigmoid(layer.gate_scores)
    assert torch.all(gates > 0) and torch.all(gates < 1), "Gates must be strictly between 0 and 1"

    # 3. Forward pass produces the expected output shape.
    dummy_input = torch.randn(4, 16, device=device)
    output = layer(dummy_input)
    assert output.shape == (4, 8), f"Unexpected output shape: {output.shape}"

    # 4. Backward pass reaches weight, bias, and gate_scores.
    output.sum().backward()
    assert layer.weight.grad is not None, "weight received no gradient"
    assert layer.bias.grad is not None, "bias received no gradient"
    assert layer.gate_scores.grad is not None, "gate_scores received no gradient"
    assert torch.any(layer.gate_scores.grad != 0), "gate_scores gradient is all zeros"

    # 5. The optimizer actually changes gate_scores after a step.
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.1)
    before = layer.gate_scores.detach().clone()
    optimizer.step()
    after = layer.gate_scores.detach()
    assert not torch.allclose(before, after), "optimizer did not update gate_scores"

    # 6. Full model forward pass on a CIFAR-10-shaped batch.
    model = SelfPruningNet(cfg).to(device)
    fake_images = torch.randn(2, 3, 32, 32, device=device)
    logits = model(fake_images)
    assert logits.shape == (2, cfg.num_classes), f"Unexpected model output shape: {logits.shape}"

    # 7. Sparsity loss is positive when gates are not all exactly 0 (they never are).
    sp = sparsity_loss(model)
    assert sp.item() > 0, "sparsity loss should be positive"

    # 8. Total loss = classification + lambda * sparsity, and is strictly
    #    greater than the classification loss alone when lambda > 0.
    fake_labels = torch.randint(0, cfg.num_classes, (2,), device=device)
    total, cls_val, sp_val = total_loss_fn(logits, fake_labels, model, lambda_value=1e-5)
    assert total.item() > cls_val, "total loss should exceed classification loss when lambda*sparsity > 0"

    # 9. Sparsity metric is a valid percentage.
    sparsity_pct = compute_sparsity(model, cfg.sparsity_threshold)
    assert 0.0 <= sparsity_pct <= 100.0

    print("Sanity checks passed (parameters, shapes, gate range, forward/backward, optimizer, losses).\n")


def run_integration_smoke_test(cfg: Config, device: str) -> None:
    """A lightweight end-to-end run of train_one_epoch / evaluate /
    compute_sparsity / the per-lambda experiment structure, using tiny
    SYNTHETIC (random) data shaped like CIFAR-10 batches.

    This is NOT a substitute for training on real CIFAR-10 -- its only
    purpose is to catch wiring bugs (e.g. a loss that doesn't decrease at
    all, gate_scores that don't move, a sparsity computation that errors
    out) before spending time on the real, much longer experiment. No
    number produced here is reported as a result anywhere in this project.
    """
    print("Running integration smoke test with synthetic data (NOT real CIFAR-10)...")

    set_seed(cfg.seed)
    X_train = torch.randn(128, 3, 32, 32)
    y_train = torch.randint(0, cfg.num_classes, (128,))
    X_test = torch.randn(64, 3, 32, 32)
    y_test = torch.randint(0, cfg.num_classes, (64,))
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

    # Run the per-lambda experiment structure with two small lambdas, one epoch
    # each, to confirm each run gets a genuinely fresh model + optimizer and
    # that different lambdas actually lead to different learned gates.
    smoke_lambdas = [1e-6, 1e-4]
    final_gate_means = []
    for lam in smoke_lambdas:
        set_seed(cfg.seed)  # same init for every lambda, as in the real experiment
        model = SelfPruningNet(cfg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

        gate_before = model.fc1.gate_scores.detach().clone()
        stats = train_one_epoch(model, train_loader, optimizer, device, lambda_value=lam)
        gate_after = model.fc1.gate_scores.detach().clone()

        assert not torch.allclose(gate_before, gate_after), \
            f"gate_scores did not change during training (lambda={lam})"

        acc = evaluate(model, test_loader, device)
        assert 0.0 <= acc <= 1.0
        sparsity_pct = compute_sparsity(model, cfg.sparsity_threshold)
        assert 0.0 <= sparsity_pct <= 100.0

        final_gate_means.append(collect_gate_values(model).mean())
        print(f"  [smoke] lambda={lam}: synthetic train_loss={stats['loss']:.4f}, "
              f"synthetic test_acc={acc:.4f}, synthetic sparsity={sparsity_pct:.2f}% "
              f"(these numbers are meaningless -- random synthetic labels)")

    # A larger lambda should generally push mean gate values down at least
    # somewhat relative to a much smaller lambda, even after a single epoch
    # on random data. We only warn (not assert) here, since one epoch on
    # noise is a weak signal and this is not the real experiment.
    if not (final_gate_means[1] <= final_gate_means[0]):
        print("  [smoke] NOTE: higher lambda did not reduce mean gate value on this "
              "single-epoch synthetic run -- expected on noisy, 1-epoch synthetic "
              "data and not a sign of a bug; the real multi-epoch CIFAR-10 run is "
              "what actually matters.")

    print("Integration smoke test passed.\n")


# ---------------------------------------------------------------------------
# 9. EXPERIMENT DRIVER
# ---------------------------------------------------------------------------

def run_experiment_for_lambda(cfg: Config, lambda_value: float, train_loader, test_loader, device):
    """Train one model end-to-end for a single lambda value and return its
    results plus the trained model.

    A fresh model and a fresh optimizer are created inside this function on
    every call -- nothing is reused across lambda values. The random seed is
    reset before model creation so every lambda run starts from the same
    weight initialization; the only thing that differs between calls is
    `lambda_value`, which isolates lambda's effect from initialization
    noise.
    """
    set_seed(cfg.seed)  # fresh, identical initialization for every lambda run
    model = SelfPruningNet(cfg).to(device)             # fresh model
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)  # fresh optimizer

    print(f"\n=== Training with lambda = {lambda_value} ===")
    for epoch in range(1, cfg.epochs + 1):
        stats = train_one_epoch(model, train_loader, optimizer, device, lambda_value)
        print(
            f"  epoch {epoch:02d}/{cfg.epochs} | "
            f"loss={stats['loss']:.4f} cls={stats['cls_loss']:.4f} "
            f"sparsity_term={stats['sparsity_loss']:.4f} train_acc={stats['accuracy']:.4f}"
        )

    test_accuracy = evaluate(model, test_loader, device)
    sparsity_pct = compute_sparsity(model, cfg.sparsity_threshold)  # OFFICIAL metric, unchanged
    print(f"  -> lambda={lambda_value} | test_accuracy={test_accuracy:.4f} | sparsity={sparsity_pct:.2f}%")

    # Diagnostic-only breakdown of the gate distribution -- does not affect
    # test_accuracy, sparsity_pct, or anything returned/persisted below.
    diagnostics = compute_gate_diagnostics(model, cfg.sparsity_threshold)
    print_gate_diagnostics(lambda_value, diagnostics)
    assert abs(diagnostics["pct_below_0.01"] - sparsity_pct) < 1e-6, \
        "diagnostic pct_below_0.01 must match the official compute_sparsity() value"

    # Gradient decomposition on ONE real training batch, using the actual
    # trained model -- shows how much of the gradient reaching gate_scores
    # right now comes from cls_loss vs. from lambda*sparsity_loss.
    one_batch_images, one_batch_labels = next(iter(train_loader))
    one_batch_images, one_batch_labels = one_batch_images.to(device), one_batch_labels.to(device)
    grad_diag = inspect_gate_gradients(model, one_batch_images, one_batch_labels, lambda_value)
    print_gate_gradient_inspection(lambda_value, grad_diag)
    model.zero_grad(set_to_none=True)  # leave the model in a clean state

    return {"lambda": lambda_value, "test_accuracy": test_accuracy, "sparsity_pct": sparsity_pct}, model


def select_best_model(results, accuracy_tolerance: float = 0.02):
    """Select the best lambda run using a practical, plain-language rule --
    not an invented weighted formula. The rule:

      1. Find the highest test accuracy achieved across all lambda runs
         (`best_accuracy`).
      2. A run is considered to have "comparable" (i.e. not meaningfully
         worse) accuracy if its test accuracy is within `accuracy_tolerance`
         (an absolute amount, default 0.02 = 2 percentage points) of
         `best_accuracy`. This is the "maintaining strong classification
         accuracy" condition.
      3. Among the runs with comparable accuracy, pick the one with the
         highest sparsity. This is the "prefer meaningful sparsity" /
         "if two models have similar accuracy, prefer the more sparse
         model" condition.
      4. If only the single most-accurate run qualifies (i.e. every other
         run's accuracy dropped by more than the tolerance), that run is
         the best model by default -- we explicitly avoid choosing a
         high-sparsity run whose accuracy collapsed just because it is
         sparse.

    `accuracy_tolerance` is a configurable knob (Config.accuracy_tolerance),
    not a hidden constant, precisely so this rule can be tuned without
    touching the selection logic itself.
    """
    if not results:
        raise ValueError("results is empty -- no lambda experiments were run")

    best_accuracy = max(r["test_accuracy"] for r in results)
    comparable = [r for r in results if (best_accuracy - r["test_accuracy"]) <= accuracy_tolerance]

    # Among the accuracy-comparable candidates, prefer the most sparse one.
    best = max(comparable, key=lambda r: r["sparsity_pct"])
    return best


def save_results_csv(results, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lambda", "test_accuracy", "sparsity_pct"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def plot_gate_distribution(gate_values: np.ndarray, threshold: float, save_path: str):
    """Histogram of the ACTUAL gate values collected from the selected
    trained model via collect_gate_values(). Nothing here is synthesized or
    adjusted to force a particular shape -- whatever distribution the model
    actually learned is what gets plotted."""
    plt.figure(figsize=(7, 4.5))
    plt.hist(gate_values, bins=50, color="#4C72B0", edgecolor="white")
    plt.axvline(threshold, color="red", linestyle="--", linewidth=1.5,
                label=f"pruning threshold = {threshold}")
    plt.xlabel("Gate value (sigmoid(gate_scores))")
    plt.ylabel("Number of weights")
    plt.title("Distribution of learned gate values (selected best model)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# 10. MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Self-Pruning Neural Network on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from Config")
    parser.add_argument("--lambdas", type=float, nargs="+", default=None,
                         help="Override lambda_values from Config, e.g. --lambdas 1e-6 1e-5 1e-4 1e-3")
    parser.add_argument("--skip-sanity-checks", action="store_true",
                         help="Skip both the unit sanity checks and the synthetic-data integration smoke test")
    args = parser.parse_args()

    cfg = Config()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.lambdas is not None:
        cfg.lambda_values = args.lambdas

    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(cfg.data_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    set_seed(cfg.seed)

    if not args.skip_sanity_checks:
        run_sanity_checks(cfg, device)
        run_integration_smoke_test(cfg, device)

    train_loader, test_loader = get_dataloaders(cfg)

    results = []
    trained_models = {}
    for lambda_value in cfg.lambda_values:
        result, model = run_experiment_for_lambda(cfg, lambda_value, train_loader, test_loader, device)
        results.append(result)
        trained_models[lambda_value] = model

    # Persist the results table.
    csv_path = os.path.join(cfg.results_dir, "results.csv")
    save_results_csv(results, csv_path)
    print(f"\nSaved results table to {csv_path}")

    print("\n| Lambda | Test Accuracy | Sparsity Level (%) |")
    print("|--------|----------------|----------------------|")
    for r in results:
        print(f"| {r['lambda']} | {r['test_accuracy']:.4f} | {r['sparsity_pct']:.2f} |")

    # Select the best model using the practical rule described in
    # select_best_model()'s docstring, and plot its gate distribution.
    best = select_best_model(results, cfg.accuracy_tolerance)
    best_model = trained_models[best["lambda"]]
    print(f"\nBest model selected: lambda={best['lambda']} "
          f"(test_accuracy={best['test_accuracy']:.4f}, sparsity={best['sparsity_pct']:.2f}%)")

    gate_values = collect_gate_values(best_model)
    plot_path = os.path.join(cfg.results_dir, "gate_distribution.png")
    plot_gate_distribution(gate_values, cfg.sparsity_threshold, plot_path)
    print(f"Saved gate distribution plot to {plot_path}")


if __name__ == "__main__":
    main()