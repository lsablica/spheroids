import numpy as np
import _pkbd
print(_pkbd)
# Simple test
n, d = 10, 3
data = np.random.randn(n, d)
weights = np.random.rand(n)
mu_vec = np.random.rand(d)
rho = 0.5

# Test M_step_PKBD
new_mu_vec, new_rho = _pkbd.M_step_PKBD(data, weights, mu_vec, rho, n, d)
print("new_mu_vec:", new_mu_vec)
print("new_rho:", new_rho)

# Test logLik_PKBD
ll = _pkbd.logLik_PKBD(data, new_mu_vec, new_rho)
print("log-likelihood:", ll)

