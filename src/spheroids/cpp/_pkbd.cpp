// _pkbd.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <cmath>

namespace py = pybind11;
using namespace Eigen;

// Internal helper function
double hybridnewton(double c1, double c2, double c3, double tol = 1e-6, int maxiter = 100) {
  double x = 0.5;
  double a = 0.0;
  double b = 1.0;
  double oldx = 1.0;
  int niter = 0;
  double g, dg;
  
  while(std::fabs(x-oldx) && niter < maxiter) {
    oldx = x;
    g = x*c1 + x*c2/(1-x*x) - c3;
    dg = c1 + c2*(1+x*x)/std::pow(1-x*x,2);
    x -= g/dg;
    
    if(x < a || x > b) {
      if(g > 0) {
        b = oldx;
      } else {
        a = oldx;
      }
      x = (b+a)/2;
    }
    niter += 1;
  }
  return x;
}

// Log-likelihood calculation
VectorXd logLik_PKBD(const Ref<const MatrixXd>& data, 
                     const Ref<const VectorXd>& mu_vec, 
                     double rho) {
  int d = data.cols();
  VectorXd cross_prod = data * mu_vec;
  double rho_sq = rho * rho;
  
  // Create a vector of the constant term
  VectorXd result = VectorXd::Constant(data.rows(), std::log(1 - rho_sq));
  
  // Use array operations for element-wise operations
  VectorXd temp = VectorXd::Constant(data.rows(), 1 + rho_sq);
  temp.array() -= (2 * rho * cross_prod.array());
  result.array() -= d * temp.array().log() / 2;
  
  return result;
}

// Main M-step function
py::tuple M_step_PKBD(const Ref<const MatrixXd>& data, 
                      const Ref<const VectorXd>& weights,
                      const Ref<const VectorXd>& mu_vec,
                      double rho,
                      int n,
                      int d,
                      double tol = 1e-6,
                      int maxiter = 100) {
  
  // Calculate alpha (mean of weights)
  double alpha = weights.mean();
  
  // Calculate cross products
  VectorXd crossmat = data * mu_vec;
  
  // Use array operations for element-wise operations
  VectorXd wscale = VectorXd::Ones(n);
  double rho_sq = rho * rho;
  wscale.array() += rho_sq;
  wscale.array() -= 2 * rho * crossmat.array();
  
  // Calculate weighted statistics
  VectorXd scaled_weight = weights.array() / wscale.array();
  VectorXd mu = data.transpose() * scaled_weight;
  
  // Normalize mu
  double mu_norm = mu.norm();
  VectorXd mu_vec_new = mu / mu_norm;
  
  // Calculate sum of scaled weights
  double sums_scaled_weight = scaled_weight.sum();
  // Calculate new rho using hybrid Newton method
  double rho_new = hybridnewton(d*sums_scaled_weight, 2*n*alpha, d*mu_norm, tol, maxiter);
  
  // Return tuple of results
  return py::make_tuple(mu_vec_new, rho_new);
}

PYBIND11_MODULE(_pkbd, m) {
  m.doc() = "PKBD clustering algorithms implemented in C++";
  
  m.def("M_step_PKBD", &M_step_PKBD, 
        py::arg("data"),
        py::arg("weights"),
        py::arg("mu_vec"),
        py::arg("rho"),
        py::arg("n"),
        py::arg("d"),
        py::arg("tol") = 1e-6,
        py::arg("maxiter") = 100,
        R"pbdoc(
            Perform M-step for PKBD clustering.
            
            Args:
              data: Matrix of data points
            weights: Vector of weights
            mu_vec: Current mean direction vector
            rho: Current concentration parameter
            n: Number of data points
            d: Dimension of data
            tol: Tolerance for convergence
            maxiter: Maximum number of iterations
            
            Returns:
              tuple: (new_mu_vec, new_rho)
        )pbdoc"
  );
  
  m.def("logLik_PKBD", &logLik_PKBD,
        py::arg("data"),
        py::arg("mu_vec"),
        py::arg("rho"),
        R"pbdoc(
            Calculate log-likelihood for PKBD distribution.
            
            Args:
              data: Matrix of data points
            mu_vec: Mean direction vector
            rho: Concentration parameter
            
            Returns:
              Vector of log-likelihood values for each data point
        )pbdoc"
  );
}