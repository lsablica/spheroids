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
  double d = data.n_cols;
  arma::vec result = log(1-rho*rho) - d*arma::log(1 + rho*rho -2*rho*data*mu_vec)/2;
  return arma_vec_to_pyarray(result); 
} 

arma::mat logLik_PKBD_intern(const arma::mat &data, const arma::mat &mu_mat, const arma::vec &rho){ 
  
  double d = data.n_cols;
  
  arma::rowvec row_rho = rho.t(); 
  arma::vec term_1 = log(1-row_rho*row_rho)
  arma::vec term_2 = 1 + row_rho*row_rho
  
  arma::mat cross_mat = data*mu_mat
  cross_mat.each_row() %= -2*row_rho;
  cross_mat.each_row() += term_2;
  cross_mat = (-d/2)*arma::log(cross_mat);
  cross_mat.each_row() += term_1;
  return cross_mat; 
} 



void hard(arma::mat &beta_matrix, int K, int n){
  arma::uvec j(1), maxindex = index_max( beta_matrix, 1);
  beta_matrix.zeros();
  for(int i = 0; i < K; i++) {
    j(0) = i;
    beta_matrix.submat(arma::find(maxindex==i),j).ones();
  }
  return;
} 
void soft(arma::mat &beta_matrix, int K, int n){
  return;
} 
void stoch(arma::mat &beta_matrix, int K, int n){
  arma::uvec j(1), maxindex = arma::sum(arma::repelem(arma::randu(n),1,K)>arma::cumsum(beta_matrix, 1), 1);
  beta_matrix.zeros();
  for(int i = 0; i < K; i++) {
    j(0) = i;
    beta_matrix.submat(arma::find(maxindex==i),j).ones();
  } 
  return;
} 

bool E_step(const arma::mat &data, arma::mat &beta_matrix, arma::vec &rho_vec, arma::mat &mu_matrix,
            const arma::rowvec &pi_vector, void (*E_method)(arma::mat&, int, int), int &K, double minalpha,
            int n, double p, double &lik, double reltol, double &max_log_lik){
  
  arma::mat A = logLik_PKBD_intern(data, mu_matrix, rho_vec);
  A += arma::repelem(log(pi_vector),n,1);
  
  arma::vec maxx = max( A, 1);
  maxx += log(sum(exp(A.each_col() - maxx),1));
  double lik_new = sum(maxx); 
  if(std::abs(lik - lik_new) < reltol * (std::abs(lik) + reltol)){
    lik = lik_new;
    return true;
  } else{ 
    lik = lik_new;
    A.each_col() -= maxx;
    beta_matrix = exp(A);
    E_method(beta_matrix, K, n);
    pi_vector = sum(beta_matrix)/n;
    if(minalpha>0){
      subset = find( pi_vector > minalpha);
      beta_matrix = normalise(beta_matrix.cols(subset),1,1);
      K = beta_matrix.n_cols;
      mu_matrix = mu_matrix.cols(subset);
      rho_vec = rho_vec.cols(subset);
    } 
    return false;
  }
} 


py::tuple EM(const py::array_t<double> &data_arr, int K, String E_type, String M_type,  double minalpha, int maxiter, double reltol){
  arma::mat data = pyarray_to_arma_mat(data_arr);
  data = normalise(data, 2, 1); // normalize in case
  double p = data.n_cols;
  int n = data.n_rows;
  
  double (*M_method)(double, double, double, int, double, int);
  void (*E_method)(arma::mat&, int, int);
  
  if(E_type == "softmax"){
    E_method = soft;
  } else if(E_type == "hardmax"){
    E_method = hard;
  } else{ // stochmax 
    E_method = stoch;
  } 
  
  if(M_type == "pkbd"){
    M_method = M_step_PKBD;
  } else{ // "spcauchy" 
    M_method = M_step_spcauchy;
  } 
  
  arma::mat beta_matrix(n, K);
  arma::mat mu_matrix(p, K);
  arma::vec rho_vec(K);
  arma::rowvec pi_vector(K);
  double log_lik;
  
  
  int i = 0;
  bool stop;
  while(i<maxiter){
    stop = E_step(/*c-r*/data,/*r*/beta_matrix,/*c-r*/rho_vec,/*c-r*/mu_matrix,/*c-r*/ pi_vector, E_method, /*r*/K,
                  minalpha, n, p, log_lik, reltol);
    if(stop) break;
    M_step(/*c-r*/data, M_method,/*c-r*/beta_matrix,/*r*/rho_vec,/*r*/mu_matrix,/*r*/pi_vector, K, reltol, p, n);
    i += 1;
  }
  
  return  py::make_tuple(beta_matrix, rho_vec.t(), mu_matrix, pi_vector, log_lik);
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