import numpy as np
import _pkbd
import _rpkbd
import _scauchy

# Test _pkbd
data = np.random.rand(100, 5)
weights = np.random.rand(100)
mu_vec = np.random.rand(5)
rho = 0.5
n, d = data.shape

# M_step_PKBD test
new_mu_vec, new_rho = _pkbd.M_step_PKBD(data, weights, mu_vec, rho, n, d)
print("M_step_PKBD:", new_mu_vec, new_rho)

ll = _pkbd.logLik_PKBD(data, new_mu_vec, new_rho)
print("logLik_PKBD shape:", ll.shape)

# Test _rpkbd
samples = _rpkbd.rPKBD_ACG(10, 0.5, np.random.rand(5))
print("rPKBD_ACG samples shape:", samples.shape)

# Test _scauchy
data_sc = np.random.randn(100, 5)
weights_sc = np.random.rand(100)
mu_sc = np.random.rand(5)
rho_sc = 0.5

# rspcauchy test
sc_samples = _scauchy.rspcauchy(10, rho_sc, mu_sc)
print("rspcauchy samples shape:", sc_samples.shape)

# M_step_sCauchy test
mu_new_sc, rho_new_sc = _scauchy.M_step_sCauchy(data_sc, weights_sc, 100, 5)
print("M_step_sCauchy:", mu_new_sc, rho_new_sc)

ll_sc = _scauchy.logLik_sCauchy(data_sc, mu_new_sc, rho_new_sc)
print("logLik_sCauchy shape:", ll_sc.shape)

