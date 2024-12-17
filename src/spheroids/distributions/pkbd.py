import numpy as np
from ..cpp import _pkbd, _rpkbd

class PKBD:
    """Wrapper for PKBD C++ functions"""
    
    @staticmethod
    def log_likelihood(data, mu_vec, rho):
        """Wrapper for logLik_PKBD C++ function"""
        return _pkbd.logLik_PKBD(data, mu_vec, rho)
    
    @staticmethod
    def m_step(data, weights, mu_vec, rho, n, d, tol=1e-6, maxiter=100):
        """Wrapper for M_step_PKBD C++ function"""
        return _pkbd.M_step_PKBD(data, weights, mu_vec, rho, n, d, tol, maxiter)
    
    @staticmethod
    def random_sample(n, rho, mu):
        """Wrapper for rPKBD_ACG C++ function"""
        return _rpkbd.rPKBD_ACG(n, rho, mu)
