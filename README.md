<div align="center">
  <img src="spheroids/misc/Logos/Spheroids1.png" alt="Spheroids Logo" width="200"/>

  # Spheroids

  [![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
  [![License: GPL-3.0](https://img.shields.io/badge/License-GPL%203.0-blue.svg)](https://opensource.org/licenses/GPL-3.0)
  [![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)
  [![GitHub Issues](https://img.shields.io/github/issues/lsablica/spheroids.svg)](https://github.com/lsablica/spheroids2/issues)
  [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

  *High-performance spherical clustering with PyTorch and C++*

  [Key Features](#key-features) •
  [Installation](#installation) •
  [Quick Start](#quick-start) •
  [Documentation](#documentation) •
  [Contributing](#contributing)

</div>

---

## Key Features  

🚀 **High Performance**
- Core computations implemented in C++ with Eigen
- GPU acceleration via PyTorch
- Efficient batch processing

🎯 **Multiple Distributions**
- PKBD 
- Spherical Cauchy distribution
- Extensible architecture for new distributions

📊 **Clustering Capabilities**
- Automatic cluster number selection
- Robust parameter estimation
- Support for high-dimensional data

## Installation

### Prerequisites

Before installing, ensure you have:

- Python ≥3.8
- C++ compiler with C++14 support
- Armadillo library

```bash
# Ubuntu/Debian
sudo apt-get install libeigen3-dev

# macOS
brew install eigen

# Windows (with vcpkg)
vcpkg install eigen3
```

### Installing from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/spheroids.git
cd spheroids

# Install dependencies and package
pip install -e .
```

## Quick Start

```python
import torch
from spheroids import SphericalClustering

# Prepare your data (normalize to unit sphere)
X = torch.randn(1000, 3)
X = X / torch.norm(X, dim=1, keepdim=True)
Y = torch.randn(1000, 2)
Y = Y / torch.norm(Y, dim=1, keepdim=True)

# Create and fit model
model = SphericalClustering(
    num_covariates=3,
    response_dim=2,
    num_clusters=3,
    distribution="pkbd"
)

# Fit model
model.fit(X, Y, num_epochs=100)
```

## Using C++ Implementations

Access optimized C++ implementations directly:

```python
from spheroids.distributions import PKBD

# Generate random samples
samples = PKBD.random_sample(
    n=100,
    rho=0.5,
    mu=np.array([1.0, 0.0])
)

# Calculate log-likelihood
loglik = PKBD.log_likelihood(data, mu, rho)
```

## API Reference

### SphericalClustering

```python
SphericalClustering(
    num_covariates: int,     # Number of input features
    response_dim: int,       # Dimension of response variables
    num_clusters: int,       # Initial number of clusters
    distribution: str,       # "pkbd" or "spcauchy"
    min_weight: float = 0.05 # Minimum cluster weight
)
```

### Key Methods

```python
# Fit the model
model.fit(
    X: torch.Tensor,        # Input features (N x num_covariates)
    Y: torch.Tensor,        # Response variables (N x response_dim)
    num_epochs: int = 100,  # Number of training epochs
    lr: float = 1e-3       # Learning rate
)

# Get cluster assignments
assignments = model.predict(X)
```

## Examples

<details>
<summary>Basic Clustering Example</summary>

```python
import torch
from spheroids import SphericalClustering

# Create model
model = SphericalClustering(
    num_covariates=3,
    response_dim=2,
    num_clusters=3
)

# Fit and predict
model.fit(X, Y)
clusters = model.predict(X)
```
</details>

<details>
<summary>Advanced Usage with C++</summary>

```python
from spheroids.distributions import PKBD, SphericalCauchy

# PKBD distribution
pkbd_samples = PKBD.random_sample(1000, 0.5, mu)
pkbd_loglik = PKBD.log_likelihood(data, mu, rho)

# Spherical Cauchy distribution
scauchy_samples = SphericalCauchy.random_sample(1000, 0.5, mu)
scauchy_loglik = SphericalCauchy.log_likelihood(data, mu, rho)
```
</details>

## Contributing

We welcome contributions! Here's how you can help:

1. 🐛 [Report bugs](https://github.com/lsablica/spheroids/issues)
2. 💡 [Suggest features](https://github.com/lsablica/spheroids/issues)


## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use Spheroids in your research, please cite:

```bibtex
@software{spheroids,
  title = {Spheroids: A Python Package for Spherical Clustering Models},
  author = {Lukas Sablica},
  year = {2024},
  url = {https://github.com/lsablica/spheroids}
}
```

---
