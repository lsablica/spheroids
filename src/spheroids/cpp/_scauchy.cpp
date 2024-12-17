#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <random>
#include <cmath>

namespace py = pybind11;
using namespace Eigen;

// Hypergeometric function implementation
double hyper2F1(const double b, const double c, const double x) {
  double sum = 1.0;
  double element = 1.0;
  double i = 0.0;
  
  while(std::fabs(element/sum) > 1e-7) {
    element *= (b+i) * x / (c+i);
    sum += element;
    i += 1.0;
  }
  
  return sum;
}

double n1d(const double d, const double x) {
  double z = 4*x/((1+x)*(1+x));
  return 1 + 2*(1-z)*(1-hyper2F1(d/2, d, z))/z;
}

double n1d_deriv(const double d, const double x) {
  double z = 4*x/((1+x)*(1+x));
  double F = hyper2F1(d/2, d, z);
  return (1/2-1/(2*x*x))*(1-F) - (1-z)*((d/2+1)*hyper2F1(d/2+2, d+1, z) + d*(1-F)/z)/z;
}

double hybridnewton(double d, double target, double tol = 1e-6, int maxiter = 100) {
  double x = 0.5;
  double a = 0.0;
  double b = 1.0;
  double oldx = 1.0;
  int niter = 0;
  
  while(std::fabs(x-oldx) > tol && niter < maxiter) {
    oldx = x;
    double g = n1d(d,x) - target;
    double dg = n1d_deriv(d,x);
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

MatrixXd Moebius_S(const Ref<const MatrixXd>& X, const Ref<const VectorXd>& mu, double rho) {
  double rho_sq = rho * rho;
  VectorXd denom = (X * mu).array() * (2 * rho) + 1 + rho_sq;
  
  MatrixXd Y = (1 - rho_sq) * (X.rowwise() + rho * mu.transpose());
  Y = (Y.array().colwise() / denom.array()).matrix();
  Y = Y.rowwise() + rho * mu.transpose();
  
  return Y;
}

MatrixXd rspcauchy(int n, double rho, const Ref<const VectorXd>& mu) {
  double norm = mu.squaredNorm();
  int p = mu.size();
  
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> normal(0, 1);
  
  MatrixXd A(n, p);
  for (int i = 0; i < n; i++) {
    VectorXd v(p);
    for (int j = 0; j < p; j++) {
      v(j) = normal(gen);
    }
    A.row(i) = v.normalized();
  }
  
  if (rho == 0 || norm == 0) {
    return A;
  }
  
  return Moebius_S(A, mu/std::sqrt(norm), rho);
}

VectorXd logLik_sCauchy(const Ref<const MatrixXd>& data, 
                        const Ref<const VectorXd>& mu_vec, 
                        double rho) {
  int d = data.cols();
  VectorXd cross_prod = data * mu_vec;
  double rho_sq = rho * rho;
  
  // Calculate constant term
  VectorXd result = VectorXd::Constant(data.rows(), (d-1) * std::log(1 - rho_sq));
  
  // Calculate variable term using array operations
  VectorXd temp = VectorXd::Constant(data.rows(), 1 + rho_sq);
  temp.array() -= (2 * rho * cross_prod.array());
  result.array() -= (d-1) * temp.array().log();
  
  return result;
}

py::tuple M_step_sCauchy(const Ref<const MatrixXd>& data,
                         const Ref<const VectorXd>& weights,
                         int n, int d,
                         double tol = 1e-6,
                         int maxiter = 100) {
  d = d - 1;
  VectorXd normalized_weights = weights / weights.sum();
  VectorXd weighted_means = data.transpose() * normalized_weights;
  
  int niter = 0;
  double norm, rho0;
  VectorXd mu0, psi, psiold, weighted_trans_data;
  
  psiold = 2 * weighted_means;
  norm = weighted_means.norm();
  mu0 = weighted_means / norm;
  rho0 = hybridnewton(d, norm, tol, maxiter);
  psi = rho0 * mu0;
  while(!psiold.isApprox(psi, tol) && niter < maxiter) {
    psiold = psi;
    weighted_trans_data = (Moebius_S(data, -mu0, rho0).transpose() * normalized_weights);
    psi = psiold.array() + ((d+1)*(1-rho0*rho0)/(2*d)) * weighted_trans_data.array();
    rho0 = psi.norm();
    mu0 = psi / rho0;
    niter += 1;
  }
  return py::make_tuple(mu0, rho0);
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
