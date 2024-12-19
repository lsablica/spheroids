#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <armadillo>
#include <cmath>
#include <cstring>
#include "converter.hpp"

namespace py = pybind11;


double hyper2F1(const double b, const double c, const double x){
  double sum = 1.0;
  double element = 1.0;
  double i = 0.0;
  
  while(fabs(element/sum) > 1e-7){
    
    element *= (b+i) * x / (c+i);  
    sum +=  element;
    i += 1.0;
  }
  
  return sum;
} 


double n1d(const double d, const double x){
  double z = 4*x/((1+x)*(1+x));
  return 1 + 2*(1-z)*(1-hyper2F1(d/2, d, z))/z;
} 


double n1d_deriv(const double d, const double x){
  double z = 4*x/((1+x)*(1+x));
  double F = hyper2F1(d/2, d, z);
  return (1/2-1/(2*x*x))*(1-F) - (1-z)*((d/2+1)*hyper2F1(d/2+2, d+1, z) + d*(1-F)/z )/z;
} 

double hybridnewton(double d, double target, double tol = 1e-6, int maxiter = 100) {
  double x,a,b;
  a = 0;
  b = 1;
  x = 0.5;
  
  double g, dg, oldx = 1;
  int niter = 0;
  
  while(tol < std::abs(x-oldx) && niter < maxiter){
    oldx = x; 
    g = n1d(d,x) - target;
    dg = n1d_deriv(d,x);
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


arma::mat Moebius_S(arma::mat &X, arma::vec mu, double rho){
  
  arma::mat Y = (1-rho*rho)*(X.each_row() + rho*mu.t());
  Y = Y.each_col()/(1+2*rho*X*mu+rho*rho);
  Y = Y.each_row() + rho*mu.t();
  
  return Y;
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
  rho0 = hybridnewton(d, norm, tol, maxiter);
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


py::array_t<double> logLik_sCauchy(const py::array_t<double> &data_arr,
                                   const py::array_t<double> &mu_vec_arr,
                                   double rho){ 

  arma::mat data = pyarray_to_arma_mat(data_arr);
  arma::vec mu_vec = pyarray_to_arma_vec(mu_vec_arr);
  double d = data.n_cols;
  arma::vec val = (d-1)*std::log(1-rho*rho) - (d-1)*arma::log(1 + rho*rho -2*rho*data*mu_vec); 
  return arma_vec_to_pyarray(val);
} 


PYBIND11_MODULE(_scauchy, m) {
  m.doc() = "Spherical Cauchy distribution implementations";
  
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
  
  m.def("logLik_sCauchy", &logLik_sCauchy,
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
}



