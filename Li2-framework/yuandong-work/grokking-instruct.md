# Instructions on how to investigate grokking

- Install dependencies `requirements.txt`
- Navigate to cogo folder to run `modular_addition_simple2.py` script
```
cd ssl/real-dataset/cogo
```

## Usage

The main training script is `modular_addition_simple2.py`. You can run it directly or use Hydra to override configurations.

### Single Run

To perform a single training run with specific parameters:

```bash
python modular_addition_simple2.py M=113 hidden_size=512 weight_decay=1e-4 num_epochs=10000
```

### Hyperparameter Sweep

You can perform hyperparameter sweeps using Hydra's multirun capability (`-m`). The following example demonstrates a sweep over weight decay, hidden size, modulus $M$, and random seeds, as found in the project's logs:

```bash
python modular_addition_simple2.py -m \
  activation=sqr \
  loss_func=mse \
  num_epochs=10000 \
  weight_decay=1e-5,5e-5,1e-4,2e-4,5e-4 \
  hidden_size=256,512,1024,2048 \
  learning_rate=0.01 \
  M=23,71,127 \
  test_size=0.1 \
  save_interval=100 \
  seed=1,2,3,4,5
```

This command will launch multiple jobs covering the Cartesian product of the specified parameter lists.

## Key Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `M` | Modulus for the addition/multiplication task (e.g., `113`). Can be a product string like `"2x3x4"` for product of Abelian groups. | `"127"` |
| `hidden_size` | Size of the hidden layer. | `512` |
| `weight_decay` | Weight decay coefficient. | `1e-4` |
| `learning_rate` | Learning rate for the optimizer. | `0.001` |
| `activation` | Activation function. Options: `sqr`, `relu`, `silu`, `relusqr`. | `sqr` |
| `optim` | Optimizer to use. Options: `adam`, `sgd`, `adamw`, `muon`. | `adam` |
| `num_epochs` | Total number of training epochs. | `3000` |
| `test_size` | Fraction of the dataset to use for testing (0.0 to 1.0). | `0.7` |
| `use_critical_ratio` | Whether to use critical ratio computed from proposed scaling laws $n \sim M \log M$. | `false` |
| `critical_ratio_delta` | Move `test_size` up or down around the critical ratio. | `0.0` |
| `seed` | Random seed for reproducibility. | `0` |
| `loss_func` | Loss function. Options: `nll` (Negative Log Likelihood), `mse` (Mean Squared Error). | `"nll"` |
| `save_interval` | Interval (in epochs) to save model checkpoints. | `1000` |

There are a lot of other switches. Please check the source codes for more details. 

## Other arguments 

| Argument | Description | Default |
| :--- | :--- | :--- |
| `group_type` | Type of group/expression to use. Options: `modular_addition`, `sym`, `expression`, `collection`. | `"modular_addition"` |
| `expression` | Expression to learn if group_type is `expression`. | `null` |
| `num_of_ops` | Number of operands for group arthemetic tasks. If num_of_ops = 3, then we are computing $a + b + c \pmod M$. | `2` |
| `set_weight_reg` | If set, then we solve the top-most layer $V$ as a ridge regression task, rather than updating $V$ with gradient descent. | `null` |
| `embed_trainable` | Whether to train the embedding. | `false` |
| `other_layers` | Number of other layers than the lowest and topmost layer. `0` means 2-layer, `1` means 3-layer, etc. | `0` |
| `group_collection_max_dk` | Maximum size of irreducible representation to use for group collection. | `0` |
| `use_complex_weights` | Whether to use complex weights. | `false` |

## Directory Structure

- `config/`: Contains Hydra configuration files (e.g., `dyn_madd.yaml`).
- `modular_addition_simple2.py`: Main training script.

## Requirements

- Python 3.x
- PyTorch
- Hydra 1.2
- Scikit-learn
