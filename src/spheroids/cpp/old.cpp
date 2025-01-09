#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <armadillo>
#include <cmath>
#include <cstring>
#include "converter.hpp"

namespace py = pybind11;

// [[Rcpp::export]]
Rcpp::List E_step_test(const arma::mat &data, arma::mat &beta_matrix, arma::vec &rho_vec, arma::mat &mu_matrix,
                       arma::rowvec &pi_vector,
                       int &K, double minalpha, int n, double p, double &lik, double reltol){ 
  void (*E_method)(arma::mat&, int, int);
  arma::mat (*Loglik)(const arma::mat&, const arma::mat&, const arma::vec&);
  
  E_method = hard;
  Loglik = logLik_PKBD_intern;
  
  bool conv =  E_step(/*c-r*/data,/*r*/beta_matrix,/*c-r*/rho_vec,/*c-r*/mu_matrix,/*c-r*/ pi_vector, 
                      E_method, Loglik, /*r*/K, minalpha, n, p, lik, reltol);
  
  return Rcpp::List::create(
    Rcpp::Named("converged")  = conv,
    Rcpp::Named("beta_matrix") = beta_matrix,
    Rcpp::Named("rho_vec")     = rho_vec,
    Rcpp::Named("mu_matrix")   = mu_matrix,
    Rcpp::Named("pi_vector")   = pi_vector,
    Rcpp::Named("lik")         = lik
  );
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


py::tuple M_step_sCauchy(const py::array_t<double> &data_arr, 
                        const py::array_t<double> &weights_arr,
                        int n, int d, double tol = 1e-6, int maxiter = 100){ 

  arma::mat data = pyarray_to_arma_mat(data_arr);
  arma::vec weights = pyarray_to_arma_vec(weights_arr);

  d = d - 1;
  weights =  arma::normalise(weights, 1);
  arma::vec weighted_means = data.t() * weights;
  int niter = 0;
  double norm, rho0, results_rho;
  arma::vec mu0, psi, psiold;
  arma::mat weighted_trans_data(n, d);
  psiold = 2*weighted_means;
  norm = arma::norm(weighted_means);
  
  mu0 = weighted_means/norm;
  rho0 = hybridnewton2(d, norm, tol, maxiter);
  psi = rho0*mu0;
  while(arma::norm(psi-psiold, 2) > tol && niter < maxiter){
    psiold = psi;
    weighted_trans_data = Moebius_S(data, - mu0, rho0).t() * weights;
    psi = psiold + ((d+1)*(1-rho0*rho0)/(2*d))*weighted_trans_data; 
    rho0 = arma::norm(psi, 2);
    mu0 = psi/rho0;
    niter += 1;
  } 
  
  return py::make_tuple(arma_vec_to_pyarray(mu0), rho0);
}  


PYBIND11_MODULE(_scauchy, m) {
  m.doc() = "Spherical Cauchy distribution implementations";
  
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
  
  m.def("M_step_sCauchy", &M_step_sCauchy,
        py::arg("data"),
        py::arg("weights"),
        py::arg("n"),
        py::arg("d"),
        py::arg("tol") = 1e-6,
        py::arg("maxiter") = 100,
        R"pbdoc(
            Perform M-step for spherical Cauchy clustering.
            
            Args:
              data: Matrix of data points
            weights: Vector of weights
            n: Number of data points
            d: Dimension of data
            tol: Tolerance for convergence
            maxiter: Maximum number of iterations
            
            Returns:
              tuple: (mu, rho) - Updated parameters
        )pbdoc"
  );
  
  
}



