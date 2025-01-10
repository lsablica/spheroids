#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <armadillo>
#include <cmath>
#include <cstring>
#include "converter.hpp"

namespace py = pybind11;

py::array_t<double> rpkbd(int n, double rho, const py::array_t<double> &mu_arr){
  arma::vec mu = pyarray_to_arma_vec(mu_arr);

  double lambda = 2*rho/(1+rho*rho);
  double norm = as_scalar(sum(arma::pow(mu,2)));
  int p = mu.n_elem;
  arma::mat A(n, p);
  if(lambda == 0 || norm == 0){/*uniform*/
    A.randn();
    A = arma::normalise(A, 2, 1);
    return arma_mat_to_pyarray(A);
  }
  mu = mu/std::sqrt(norm);
  int count = 0;
  int Nt = 0;
  double unif, mACG, PKBD, mutz, ratio, mutx;
  arma::vec candidate;
  
  double pp = (double)p;
  arma::vec coe = { -4*(pp-1) , 4*pp-lambda*lambda*(pp-2)*(pp-2), 2*pp*(pp-2)*lambda*lambda, -pp*pp*lambda*lambda};
  arma::vec RO = arma::sort(arma::real(arma::roots(coe)));
  double b = RO(1);
  
  double minuslogM = std::log((1+sqrt(1-lambda*lambda/b))/2);
  double b2 = -1 + std::sqrt(1/(1-b));
  double b1 = b/(1-b);  
  
  while(count<n){
    candidate = arma::randn<arma::vec>(p);
    mutz = arma::dot(mu, candidate) ;
    norm = sqrt(arma::dot(candidate,candidate) + b1*mutz*mutz);
    mutx = mutz*(1+b2)/norm ;  
    PKBD = -std::log(1-lambda*mutx);
    mACG =  std::log(1-b*mutx*mutx);
    unif = arma::randu<double>();
    ratio = 0.5*p*(PKBD + mACG + minuslogM);
    if(log(unif)<ratio){
      candidate = (candidate + b2*mutz*mu)/norm;
      A.row(count) = arma::trans(candidate);
      count += 1;
    }
    Nt += 1;
  }
  return arma_mat_to_pyarray(A);
}


py::array_t<double> rspcauchy(int n, double rho, const py::array_t<double> &mu_arr){
  arma::vec mu = pyarray_to_arma_vec(mu_arr);

  double norm = arma::as_scalar(arma::sum(arma::pow(mu,2)));
  int p = mu.n_elem;
  arma::mat A(n, p);
  A = normalise(A.randn(),2,1);
  if(rho == 0 || norm == 0){/*uniform*/
    return arma_mat_to_pyarray(A);
  }
  A = Moebius_S(A, mu, rho);
  return arma_mat_to_pyarray(A);
} 



py::array_t<double> loglik_spcauchy(const py::array_t<double> &data_arr,
                                   const py::array_t<double> &mu_vec_arr,
                                   double rho){ 

  arma::mat data = pyarray_to_arma_mat(data_arr);
  arma::vec mu_vec = pyarray_to_arma_vec(mu_vec_arr);
  double d = data.n_cols;
  arma::vec val = (d-1)*std::log(1-rho*rho) - (d-1)*arma::log(1 + rho*rho -2*rho*data*mu_vec); 
  return arma_vec_to_pyarray(val);
} 


py::array_t<double> loglik_pkbd(const py::array_t<double> &data_arr, const py::array_t<double> &mu_vec_arr, double rho){ 
  
  arma::mat data = pyarray_to_arma_mat(data_arr);
  arma::vec mu_vec = pyarray_to_arma_vec(mu_vec_arr);
  double d = data.n_cols;
  arma::vec result = log(1-rho*rho) - d*arma::log(1 + rho*rho -2*rho*data*mu_vec)/2;
  return arma_vec_to_pyarray(result); 
} 


PYBIND11_MODULE(_utils, m) {
  m.doc() = "Utilities for PKBD and spherical Cauchy distributions";
  
  m.def("rpkbd", &rpkbd,
        py::arg("n"),
        py::arg("rho"),
        py::arg("mu"),
        R"pbdoc(
            Generate random samples from PKBD distribution using ACG envelopes.
          
          Args:
          n: Number of samples to generate
          rho: Concentration parameter
          mu: Mean direction vector
          
          Returns:
          Matrix of generated samples (n x p)
        )pbdoc"
  );
  m.def("rspcauchy", &rspcauchy,
        py::arg("n"),
        py::arg("rho"),
        py::arg("mu"),
        R"pbdoc(
            Generate random samples from spherical Cauchy distribution.
          
          Args:
          n: Number of samples to generate
          rho: Concentration parameter
          mu: Mean direction vector
          
          Returns:
          Matrix of generated samples (n x p)
        )pbdoc"
  );

  m.def("loglik_spcauchy", &loglik_spcauchy,
        py::arg("data"),
        py::arg("mu_vec"),
        py::arg("rho"),
        R"pbdoc(
            Calculate log-likelihood for spherical Cauchy distribution.
            
            Args:
              data: Matrix of data points
            mu_vec: Mean direction vector
            rho: Concentration parameter
            
            Returns:
              Vector of log-likelihood values for each data point
        )pbdoc"
  );

    m.def("loglik_pkbd", &loglik_pkbd,
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
