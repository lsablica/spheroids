Here is the updated `README.md` with the necessary changes. I've incorporated updates to reflect the use of Armadillo, precompiled wheels via GitHub Actions, and adjusted installation instructions for modern setups. Below is the updated version, with changes explained afterward.

---

# Spheroids

<div align="center">
  <img src="spheroids/misc/Logos/Spheroids1.png" alt="Spheroids Logo" width="200"/>

  # Spheroids

  [![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
  [![License: GPL-3.0](https://img.shields.io/badge/License-GPL%203.0-blue.svg)](https://opensource.org/licenses/GPL-3.0)
  [![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)
  [![GitHub Issues](https://img.shields.io/github/issues/lsablica/spheroids.svg)](https://github.com/lsablica/spheroids/issues)
  [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

  *High-performance spherical clustering with PyTorch, C++, and Armadillo*

  [Key Features](#key-features) •
  [Installation](#installation) •
  [Quick Start](#quick-start) •
  [Documentation](#documentation) •
  [Contributing](#contributing)

</div>

---

## Key Features  

🚀 **High Performance**
- Core computations implemented in C++ with Armadillo
- Precompiled wheels for Linux, macOS, and Windows via GitHub Actions
- GPU acceleration via PyTorch

🎯 **Multiple Distributions**
- Poisson Kernel-Based Distribution (PKBD) 
- Spherical Cauchy distribution
- Extensible architecture for new distributions

📊 **Clustering Capabilities**
- Automatic cluster number selection
- Robust parameter estimation
- Support for high-dimensional data

---

## Installation

### Quick Install (Recommended)

You can install Spheroids directly from PyPI with precompiled wheels:

```bash
pip install spheroids
```

### Advanced Installation (Local Compilation)

For users who want to build the package locally (e.g., to modify the codebase), follow these steps:

#### Prerequisites
- Python ≥3.8
- C++ compiler with C++17 support
- [Armadillo](http://arma.sourceforge.net/) installed

#### Steps

##### On Linux

```bash
# Install required libraries
sudo apt-get update
sudo apt-get install -y libarmadillo-dev libomp-dev

# Clone the repository
git clone https://github.com/lsablica/spheroids.git
cd spheroids

# Build and install
pip install -e .
```

##### On macOS

```bash
# Install required libraries
brew update
brew install armadillo libomp

# Configure compiler paths (if necessary)
export CXXFLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -I/opt/homebrew/opt/armadillo/include"
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib -lomp -L/opt/homebrew/opt/armadillo/lib"

# Clone the repository
git clone https://github.com/lsablica/spheroids.git
cd spheroids

# Build and install
pip install -e .
```

##### On Windows

```bash
# Clone vcpkg for managing C++ libraries
git clone https://github.com/microsoft/vcpkg.git C:\vcpkg
cd C:\vcpkg
.\bootstrap-vcpkg.bat -disableMetrics
.\vcpkg.exe install armadillo

# Clone the repository
git clone https://github.com/lsablica/spheroids.git
cd spheroids

# Build and install
pip install -e .
```

---

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

---

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

---

## Contributing

We welcome contributions! Here's how you can help:

1. 🐛 [Report bugs](https://github.com/lsablica/spheroids/issues)
2. 💡 [Suggest features](https://github.com/lsablica/spheroids/issues)
3. 🔧 Submit a pull request with your improvements.

---

## Changes Made

1. **Updated Build System**: Mentioned the use of Armadillo instead of Eigen for linear algebra.
2. **Precompiled Wheels**: Highlighted that wheels are precompiled for Linux, macOS, and Windows via GitHub Actions.
3. **Removed Eigen**: Replaced installation instructions for Eigen with Armadillo (`apt-get`, `brew`, `vcpkg`).
4. **Advanced Installation**: Added local compilation instructions for all platforms to reflect Armadillo use.
5. **Quick Install**: Simplified the main installation instructions to focus on `pip`.
6. **Miscellaneous**: Adjusted references to modern practices (e.g., C++17, precompiled wheels).

Let me know if you’d like further adjustments!