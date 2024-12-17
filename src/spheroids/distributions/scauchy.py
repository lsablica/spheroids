import numpy as np
from ..cpp import _scauchy

class SphericalCauchy:
    """Wrapper for Spherical Cauchy C++ functions"""
    
    @staticmethod
    def log_likelihood(data, mu_vec, rho):
        """Wrapper for logLik_sCauchy C++ function"""
        return _scauchy.logLik_sCauchy(data, mu_vec, rho)
    
    @staticmethod
    def m_step(data, weights, n, d, tol=1e-6, maxiter=100):
        """Wrapper for M_step_sCauchy C++ function"""
        return _scauchy.M_step_sCauchy(data, weights, n, d, tol, maxiter)
    
    @staticmethod
    def random_sample(n, rho, mu):
        """Wrapper for rspcauchy C++ function"""
        return _scauchy.rspcauchy(n, rho, mu)
