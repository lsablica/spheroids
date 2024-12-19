import numpy as np
import _pkbd
import _rpkbd
import _scauchy

print("Successfully imported functions!")

# Create some test data
n, d = 10, 3
np.random.seed(42)
X = np.random.randn(n, d)
# Normalize to unit vectors
X = X / np.linalg.norm(X, axis=1)[:, np.newaxis]
#print(X)
# Test PKBD
weights = np.ones(n) / n
mu = np.array([1.0, 0.0, 0.0])
rho = 0.5
print(X)
print(mu)

print("\nTesting PKBD functions:")
try:
    new_mu, new_rho = _pkbd.M_step_PKBD(X, weights, mu, rho, n, d)
    print("M_step_PKBD works!")
    print("new_mu:", new_mu)
    print("new_rho:", new_rho)
except Exception as e:
    print("Error in M_step_PKBD:", e)

try:
    ll = _pkbd.logLik_PKBD(X, mu, rho)
    print("logLik_PKBD works!")
    print("Log-likelihoods:", ll[:3])  # First three values
except Exception as e:
    print("Error in logLik_PKBD:", e)

print("\nTesting Spherical Cauchy functions:")
try:
    new_mu, new_rho = _scauchy.M_step_sCauchy(X, weights, n, d)
    print("M_step_sCauchy works!")
    print("new_mu:", new_mu)
    print("new_rho:", new_rho)
except Exception as e:
    print("Error in M_step_sCauchy:", e)
    
try:
    ll = _scauchy.logLik_sCauchy(X, mu, rho)
    print("logLik_sCauchy works!")
    print("Log-likelihoods:", ll[:3])  # First three values
except Exception as e:
    print("Error in logLik_PKBD:", e)    

print("\nTesting random sampling:")
try:
    samples_pkbd = _rpkbd.rPKBD_ACG(5, rho, mu)
    print("rPKBD_ACG works!")
    print("First sample:", samples_pkbd[0])
except Exception as e:
    print("Error in rPKBD_ACG:", e)

try:
    samples_cauchy = _scauchy.rspcauchy(5, rho, mu)
    print("rspcauchy works!")
    print("First sample:", samples_cauchy[0])
except Exception as e:
    print("Error in rspcauchy:", e)


