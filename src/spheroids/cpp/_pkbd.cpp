#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <armadillo>
#include <cmath>
#include <cstring> 
#include "converter.hpp"

namespace py = pybind11;

double hybridnewton(double c1, double c2, double c3, double tol, int maxiter) {
  double x,a,b;
  a = 0;
  b = 1;
  x = 0.5;
  
  double g, dg, oldx = 1;
  int niter = 0;
  
  while(tol < std::abs(x-oldx) && niter < maxiter){
    oldx = x; 
    g = x*c1+x*c2/(1-x*x)-c3;
    dg = c1+c2*(1+x*x)/pow(1-x*x,2);
    x -= g/dg;
    if(x < a || x > b){
      if(g>0){
        b = oldx;
      }
      else{
        a = oldx;
      }
      x= (b+a)/2;
    }
    niter +=1;
  }
  return x;
}

py::tuple M_step_PKBD(const py::array_t<double> &data_arr, const py::array_t<double> &weights_arr, const py::array_t<double> &mu_vec_arr, double rho,
                 int n, int d, double tol = 1e-6, int maxiter = 100){ 
                  
  arma::mat data = pyarray_to_arma_mat(data_arr);
  arma::vec weights = pyarray_to_arma_vec(weights_arr);
  arma::vec mu_vec = pyarray_to_arma_vec(mu_vec_arr);
  double alpha = arma::mean(weights);
  
  arma::vec crossmat = data*mu_vec;
  arma::vec wscale =  1 + std::pow(rho, 2) - 2*rho*crossmat;
  arma::vec scaled_weight = weights/wscale;
  arma::vec mu = data.t() * scaled_weight; 
  double mu_norm = arma::vecnorm(mu);
  arma::vec mu_vec2 = mu/mu_norm;
  double sums_scaled_weight = sum(scaled_weight);
  rho = hybridnewton(d*sums_scaled_weight, 2*n*alpha, d*mu_norm, tol, maxiter); 
  return py::make_tuple(arma_vec_to_pyarray(mu_vec2), rho);
} 


py::array_t<double> logLik_PKBD(const py::array_t<double> &data_arr, const py::array_t<double> &mu_vec_arr, double rho){ 
  
  arma::mat data = pyarray_to_arma_mat(data_arr);
  arma::vec mu_vec = pyarray_to_arma_vec(mu_vec_arr);
  std::cout << "data: " << data << std::endl;
  std::cout << "mu_vec: " << mu_vec << std::endl;
  double d = data.n_cols;
  arma::vec result = log(1-rho*rho) - d*arma::log(1 + rho*rho -2*rho*data*mu_vec)/2;
  return arma_vec_to_pyarray(result); 
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