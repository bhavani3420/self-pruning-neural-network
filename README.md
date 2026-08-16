Absolutely. Based on your **actual `self_pruning.py` implementation and the final results you obtained**, use the following as your `README.md`.

It avoids claiming exact-zero pruning, avoids overstating accuracy, and clearly explains that the reported sparsity is **threshold-based** (`gate < 0.01`).

````markdown
# Self-Pruning Neural Network

A feed-forward neural network for CIFAR-10 classification with **learnable,
weight-level pruning gates**.

Instead of applying a fixed pruning schedule after training, the model learns
a gate for every individual weight during training. Each gate is obtained by
applying a sigmoid function to a learnable `gate_score`, and the effective
weight is computed as:

W_effective = W * sigmoid(gate_score)

A sparsity penalty encourages the learned gates toward zero. After training,
connections whose gate value falls below a predefined threshold are counted
as pruned.

> **Important:** sigmoid outputs values strictly between 0 and 1 for finite
> inputs. Therefore, this implementation does not claim that gates become
> mathematically exactly zero. A connection is considered **pruned** when its
> learned gate value is below the configured pruning threshold of `0.01`.

---

## Project Overview

The goal of this project is to demonstrate a neural network that can
**learn which individual connections are unnecessary** while simultaneously
performing image classification.

The model optimizes two objectives:

1. **Classification accuracy** on CIFAR-10.
2. **Sparsity**, by encouraging gate values to become small.

The total training objective is:

    Total Loss = Classification Loss + λ × Sparsity Loss

where:

- `Classification Loss` = Cross-Entropy Loss
- `Sparsity Loss` = Sum of all learned gate values
- `λ` = sparsity-pressure coefficient

A larger `λ` applies stronger pressure toward smaller gate values.

---

## Architecture

The network is a fully connected feed-forward neural network.

```text
CIFAR-10 Image
    │
    │ 3 × 32 × 32
    ▼
Flatten
    │
    │ 3072 features
    ▼
PrunableLinear
    │
    │ 512 neurons
    ▼
ReLU
    │
    ▼
PrunableLinear
    │
    │ 256 neurons
    ▼
ReLU
    │
    ▼
PrunableLinear
    │
    │ 10 classes
    ▼
Class Logits
````

### Layer dimensions

| Layer | Input | Output |
| ----- | ----: | -----: |
| `fc1` |  3072 |    512 |
| `fc2` |   512 |    256 |
| `fc3` |   256 |     10 |

Every weight in these layers has a corresponding learnable `gate_score`.

The total number of gates is:

```text
3072 × 512 + 512 × 256 + 256 × 10
= 1,706,496 gates
```

---

## Learnable Pruning Mechanism

Each `PrunableLinear` layer contains:

```text
weight
bias
gate_scores
```

The raw gate score is converted into a gate using sigmoid:

```python
gate = sigmoid(gate_score)
```

The effective weight is then:

```python
effective_weight = weight * gate
```

Therefore, the forward computation becomes:

```python
gates = torch.sigmoid(self.gate_scores)
pruned_weight = self.weight * gates
output = F.linear(x, pruned_weight, self.bias)
```

### Why this works

During backpropagation, gradients flow through:

```text
gate_score
    ↓
sigmoid
    ↓
gate
    ↓
weight × gate
    ↓
network output
    ↓
loss
```

Therefore, the optimizer can directly update the gate scores.

A small gate suppresses the corresponding connection, while a larger gate
allows the connection to contribute more strongly.

---

## Sparsity Objective

The sparsity loss is calculated over all gates:

```python
def sparsity_loss(model):
    total = 0
    for layer in model.prunable_layers():
        gates = torch.sigmoid(layer.gate_scores)
        total += gates.abs().sum()
    return total
```

Since sigmoid produces positive values:

```text
abs(gate) = gate
```

so this is effectively an L1 penalty on the gate values.

The complete objective is:

```text
L = L_classification + λ L_sparsity
```

Increasing `λ` increases the pressure to reduce gate values.

---

## Pruning Definition

The implementation uses a threshold-based definition of sparsity.

```text
pruning threshold = 0.01
```

A gate is counted as pruned when:

```python
gate < 0.01
```

The reported sparsity is therefore:

```text
Sparsity (%) =
(number of gates < 0.01 / total number of gates) × 100
```

This is a **functional/thresholded definition of pruning**.

The model does not force the sigmoid output to become exactly zero.

---

## Dataset

The model is trained and evaluated on **CIFAR-10**.

CIFAR-10 contains:

* 50,000 training images
* 10,000 test images
* 10 classes
* RGB images of size 32 × 32

The training pipeline uses:

* Random Crop
* Random Horizontal Flip
* Tensor conversion
* CIFAR-10 normalization

The test pipeline uses:

* Tensor conversion
* CIFAR-10 normalization

---

## Requirements

Python 3.10+ is recommended.

Install the required packages:

```bash
pip install torch torchvision numpy matplotlib
```

The project also includes:

```text
requirements.txt
```

which can be installed using:

```bash
pip install -r requirements.txt
```

---

## Project Structure

```text
.
├── self_pruning.py
├── README.md
├── report.md
├── requirements.txt
├── .gitignore
├── data/
└── results/
    ├── results.csv
    └── gate_distribution.png
```

The `data/` and `results/` directories are generated/used during execution.

---

## Running the Project

Create and activate a virtual environment:

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the default experiment:

```powershell
python self_pruning.py
```

The script will:

1. Select CPU or CUDA automatically.
2. Run model sanity checks.
3. Run a synthetic integration smoke test.
4. Download/load CIFAR-10.
5. Train a fresh model for each lambda.
6. Evaluate test accuracy.
7. Calculate threshold-based sparsity.
8. Print gate-distribution diagnostics.
9. Inspect gate gradients on one real training batch.
10. Save the results to `results/results.csv`.
11. Select the best model according to the configured selection rule.
12. Save the gate distribution plot.

---

## Command-Line Options

### Change the number of epochs

```powershell
python self_pruning.py --epochs 25
```

### Run specific lambda values

```powershell
python self_pruning.py --lambdas 1e-5 1e-4 1e-3
```

### Example final experiment

```powershell
python self_pruning.py --lambdas 1e-5 1e-4 1e-3 --epochs 25
```

### Skip sanity and synthetic smoke tests

```powershell
python self_pruning.py --skip-sanity-checks
```

---

## Experimental Results

The final experiment was run for **25 epochs** using the following lambda
values:

```text
1e-5
1e-4
1e-3
```

The resulting test accuracy and threshold-based sparsity were:

| Lambda | Test Accuracy |   Sparsity |
| -----: | ------------: | ---------: |
| `1e-5` |        56.84% |     16.56% |
| `1e-4` |    **57.40%** | **72.53%** |
| `1e-3` |        51.92% |     98.73% |

The results are also saved automatically to:

```text
results/results.csv
```

---

## Results Interpretation

The experiments demonstrate the expected accuracy-sparsity trade-off.

### λ = 1e-5

```text
Accuracy  = 56.84%
Sparsity  = 16.56%
```

The sparsity pressure is relatively weak, so only a portion of the
connections fall below the pruning threshold.

### λ = 1e-4

```text
Accuracy  = 57.40%
Sparsity  = 72.53%
```

This experiment produced the best observed balance between classification
accuracy and sparsity.

The model removed the majority of connections according to the thresholded
gate definition while achieving the highest test accuracy among the tested
settings.

### λ = 1e-3

```text
Accuracy  = 51.92%
Sparsity  = 98.73%
```

The much stronger sparsity pressure causes almost all gates to fall below
the pruning threshold. However, excessive suppression of connections reduces
classification performance.

This demonstrates that increasing sparsity pressure indefinitely is not
beneficial.

---

## Best Model

The project selects the best model using the following rule:

1. Find the highest test accuracy.
2. Consider models whose accuracy is within 2 percentage points of the best
   accuracy.
3. Among those models, select the one with the highest sparsity.

For the final experiment:

```text
Best lambda = 1e-4

Test accuracy = 57.40%
Sparsity       = 72.53%
```

The `λ = 1e-3` model was not selected because its accuracy dropped more than
2 percentage points compared with the best observed accuracy.

---

## Gate Distribution

The project generates:

```text
results/gate_distribution.png
```

The plot shows the distribution of the learned gate values of the selected
best model.

A vertical line marks:

```text
pruning threshold = 0.01
```

Gates to the left of this threshold are counted as pruned.

The histogram is generated directly from the trained model's actual gate
values; no values are manually modified or synthesized for visualization.

---

## Gate Diagnostics

The training script reports additional diagnostics for each lambda, including:

* Total number of gates
* Sum of gate values
* Mean gate value
* Minimum and maximum gate
* Median gate
* 1st, 5th and 10th percentiles
* Percentage of gates below `0.1`
* Percentage below `0.05`
* Percentage below the official `0.01` threshold
* Percentage below `0.005`
* Percentage below `0.001`
* Minimum, maximum and mean raw `gate_score`

These diagnostics help distinguish between:

```text
gates moving toward zero
```

and:

```text
gates actually crossing the pruning threshold
```

This distinction is important because a sigmoid gate does not become exactly
zero during ordinary finite-valued training.

---

## Gradient Verification

The project also performs a diagnostic decomposition of the gradients
reaching the `gate_scores`.

For a total loss:

```text
L = L_classification + λ L_sparsity
```

the gradient satisfies:

```text
dL/dgate_score =
dL_classification/dgate_score
+
λ dL_sparsity/dgate_score
```

The implementation verifies this numerically using one real training batch
after training.

The diagnostic reports:

* Mean absolute total gradient
* Mean absolute classification gradient
* Mean absolute sparsity gradient
* Classification/sparsity gradient ratio
* Gradient additivity error

The additivity error should be approximately zero.

This provides an additional check that the sparsity objective is actually
connected to and influencing the learnable gate parameters.

---

## Sanity and Integration Tests

Before training on CIFAR-10, the script runs several checks.

### Parameter checks

It verifies that every `PrunableLinear` layer contains:

```text
weight
bias
gate_scores
```

and that:

```text
gate_scores.shape == weight.shape
```

### Gate range check

Because sigmoid is used:

```text
0 < gate < 1
```

for finite gate scores.

### Forward-pass check

The layer and full model are tested using CIFAR-10-shaped tensors.

### Backward-pass check

The tests verify that gradients reach:

```text
weight
bias
gate_scores
```

### Optimizer check

The tests verify that an optimizer step actually changes the gate scores.

### Synthetic integration test

A small synthetic dataset is used to verify:

* Training loop execution
* Evaluation
* Sparsity calculation
* Gate updates
* Lambda-specific model creation

The synthetic results are **not treated as meaningful model performance**,
because the synthetic labels are random.

---

## Reproducibility

The project sets random seeds for:

```text
Python
NumPy
PyTorch
CUDA
```

Each lambda experiment creates:

* A fresh model
* A fresh optimizer
* The same configured initialization seed

This allows the effect of changing `λ` to be studied while keeping model
initialization controlled.

Exact bit-for-bit reproducibility is not guaranteed on every GPU because
some CUDA/cuDNN operations can be nondeterministic.

---

## Why a Fully Connected Network?

This project intentionally uses a feed-forward MLP rather than a CNN.

The purpose of the case study is to demonstrate:

* Learnable gates
* Differentiable pruning
* Sparsity regularization
* Gate optimization
* Accuracy-sparsity trade-offs

The model is therefore designed as a relatively simple feed-forward
architecture rather than a state-of-the-art CIFAR-10 classifier.

Consequently, the reported accuracy should be interpreted in the context of
this architecture and the primary objective of demonstrating
self-pruning.

---

## Key Takeaways

The experiment demonstrates that:

1. Every network weight can have an independently learned pruning gate.
2. The gates are differentiable and receive gradients during training.
3. L1 regularization on gate values encourages them toward zero.
4. Increasing `λ` increases the amount of threshold-based sparsity.
5. Excessive sparsity can reduce classification accuracy.
6. A moderate sparsity pressure can provide a useful balance between
   accuracy and connection reduction.
7. In the final experiment, `λ = 1e-4` produced the best observed balance:
   **57.40% test accuracy with 72.53% threshold-based sparsity**.

---

## Limitations

This implementation has several limitations:

* The model is a fully connected MLP rather than a convolutional network.
* CIFAR-10 accuracy is therefore limited compared with typical CNN-based
  approaches.
* The pruning metric is threshold-based rather than physically deleting
  parameters from the model.
* A gate below `0.01` is considered pruned, but the underlying parameter
  still exists.
* The experiments evaluate a limited set of lambda values.
* The selected model is based on test accuracy and thresholded sparsity,
  rather than a separate validation set.

---

## Future Improvements

Possible extensions include:

* Physically removing pruned connections after training.
* Converting the dense model into a genuinely sparse representation.
* Measuring parameter count and inference-time speedup after pruning.
* Evaluating memory reduction.
* Using a CNN architecture for CIFAR-10.
* Performing a wider hyperparameter search for `λ`.
* Introducing a validation set for model selection.
* Comparing the method against conventional post-training pruning methods.
* Evaluating the accuracy/sparsity trade-off over more training epochs.

---

## Files

### `self_pruning.py`

Main implementation containing:

* `PrunableLinear`
* `SelfPruningNet`
* Sparsity loss
* Training and evaluation loops
* Sparsity calculation
* Gate diagnostics
* Gradient diagnostics
* Experiment driver
* Model selection
* Gate distribution visualization

### `results/results.csv`

Contains the final accuracy and sparsity measurements for each tested
lambda.

### `results/gate_distribution.png`

Histogram showing the learned gate distribution for the selected best model.

### `report.md`

Detailed discussion of the implementation, methodology, experiments, and
results.

---

## Summary

This project implements a differentiable self-pruning neural network in
which every weight is associated with a learnable sigmoid gate.

The network jointly learns:

```text
classification
      +
sparsity
```

through the objective:

```text
Total Loss = Classification Loss + λ × Sparsity Loss
```

The final experiments demonstrate a clear relationship between sparsity
pressure and model performance. The best observed configuration was:

```text
λ = 1e-4
Test Accuracy = 57.40%
Threshold-based Sparsity = 72.53%
```

This demonstrates that learnable gates can substantially suppress network
connections while retaining useful classification performance.

````

### One important thing before you paste it

Your current `self_pruning.py` has the default:

```python
epochs = 10
lambda_values = [1e-6, 1e-5, 1e-4]
````

but your **final reported experiment** was:

```bash
python self_pruning.py --lambdas 1e-5 1e-4 1e-3 --epochs 25
```

That's completely fine. The README above explicitly identifies the **25-epoch command as the final experiment**, so there is no contradiction.

For GitHub, I would push the README together with:

```text
self_pruning.py
requirements.txt
report.md
results/results.csv
results/gate_distribution.png
.gitignore
```

and **not push the `data/` folder or `venv/` folder**.
