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

arma::mat logLik_PKBD_intern(const arma::mat &data, const arma::mat &mu_mat, const arma::vec &rho){ 
  
  double d = data.n_cols;
  
  arma::rowvec row_rho = rho.t(); 
  arma::rowvec term_1 = log(1-row_rho*row_rho)
  arma::rowvec term_2 = 1 + row_rho*row_rho
  
  arma::mat cross_mat = data*mu_mat
  cross_mat.each_row() %= -2*row_rho;
  cross_mat.each_row() += term_2;
  cross_mat = (-d/2)*arma::log(cross_mat);
  cross_mat.each_row() += term_1;
  return cross_mat; 
} 


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

double hybridnewton2(double d, double target, double tol = 1e-6, int maxiter = 100) {
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

void M_step_PKBD(const arma::mat &data, const arma::mat &beta_matrix, arma::mat &mu_matrix, arma::vec &rho_vector,
                 double tol = 1e-6, int maxiter = 100){
  int k = mu_matrix.n_cols;            
  int n = data.n_rows;   
  int d = data.n_cols;  
  arma::rowvec alpha = sum(beta_matrix)/n;


  arma::mat crossmat = data*mu_matrix;
  arma::mat rho_mat(n, k);
  rho_mat.each_row() = rho_vector.t();
  arma::mat wscale_mat =  1 + pow(rho_mat, 2) - 2*rho_mat%crossmat;
  arma::mat scaled_weight_matrix = beta_matrix/wscale_mat;
  arma::mat mu = scaled_weight_matrix.t() * data;
  arma::vec mu_norms = arma::vecnorm(mu, 2, 1);
  mu_matrix = mu.each_col()/mu_norms;
  arma::rowvec sums_scaled_weight_matrix = sum(scaled_weight_matrix, 0);
  //standardize each
  double c1, c2, c3;
  for(int i = 0; i < k; i++){
    c1 = d*sums_scaled_weight_matrix(i);
    c2 = 2*n*alpha(i);
    c3 = d*mu_norms(i);
    rho_vector(i) = hybridnewton(c1,c2,c3,tol,maxiter);
  }
}

void M_step_spcauchy(const arma::mat &data, const arma::mat &beta_matrix, arma::mat &mu_matrix, arma::vec &rho_vector,
                     double tol = 1e-6, int maxiter = 100){
  int k = mu_matrix.n_cols;            
  int n = data.n_rows;   
  int d = data.n_cols;                      
  arma::rowvec sums = sum(beta_matrix);
  arma::rowvec alpha = sums/n;
  arma::mat weights =  beta_matrix.each_row()/sums;
  arma::mat weighted_means = data.t() * weights;

  // we iterate over all clusters k, calculate method of moments estimate and use it for MLE estimate
  int niter;
  double norm, rho0;
  arma::vec mu0, psi, psiold, w, results_rho(k);
  arma::mat weighted_trans_data(n, d), results_mu(d, k);
  for(int i = 0; i < k; i++){
    niter = 0;
    mu0 = weighted_means.col(i);
    w = weights.col(i);
    psiold = 2*mu0;
    norm = arma::norm(mu0, 2);
    mu0 = mu0/norm;
    rho0 = hybridnewton(d, norm, tol = tol, maxiter = maxiter);
    psi = rho0*mu0;
    Rcout << "psi0 : " << psi << "\n";

    while(arma::norm(psi-psiold, 2) > tol && niter < maxiter){
      psiold = psi;
      weighted_trans_data = Moebius_S(data, - mu0, rho0).t() * w;
      psi = psiold + ((d+1)*(1-rho0*rho0)/(2*d))*weighted_trans_data;
      Rcout << "psi : " << psi << "\n";
      rho0 = arma::norm(psi, 2);
      mu0 = psi/rho0;
      niter += 1;
    }
    results_mu.col(i) = mu0;
    results_rho(i) = rho0;
  }
  mu_matrix = results_mu;
  rho_vector = results_rho;
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
  
  double (*M_method)(arma::mat&, arma::mat&, arma::mat&, arma::vec&, double, int);
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
  
  m.def("EM", &EM, 
        py::arg("data"),
        py::arg("K"),
        py::arg("E_type"),
        py::arg("M_type"),
        py::arg("minalpha") = 0,
        py::arg("maxiter") = 100,
        py::arg("reltol") = 1e-6,
        R"pbdoc(
            Perform EM algorithm for PKBD clustering.
            
            Args:
              data: Matrix of data points
              K: Number of clusters
              E_type: Type of E-step method
              M_type: Type of M-step method
              minalpha: Minimum alpha value
              maxiter: Maximum number of iterations
              reltol: Relative tolerance
        )pbdoc");
  
}