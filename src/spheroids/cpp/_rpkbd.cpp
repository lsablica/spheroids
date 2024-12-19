#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <armadillo>
#include <cmath>
#include <cstring>
#include "converter.hpp"

namespace py = pybind11;

py::array_t<double> rPKBD_ACG(int n, double rho, const py::array_t<double> &mu_arr){
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

PYBIND11_MODULE(_rpkbd, m) {
  m.doc() = "Random sampling for PKBD distributions using ACG envelopes";
  
  m.def("rPKBD_ACG", &rPKBD_ACG,
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
}
