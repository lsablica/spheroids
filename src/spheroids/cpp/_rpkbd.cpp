#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <unsupported/Eigen/Polynomials>  
#include <random>
#include <cmath>

namespace py = pybind11;
using namespace Eigen;

MatrixXd rPKBD_ACG(int n, double rho, const Ref<const VectorXd>& mu) {
  double lambda = 2 * rho / (1 + rho * rho);
  double norm = mu.squaredNorm();
  int p = mu.size();
  
  // Initialize random number generators
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> normal(0, 1);
  std::uniform_real_distribution<> uniform(0, 1);
  
  // Create output matrix
  MatrixXd A(n, p);
  
  // Handle special cases
  if (lambda == 0 || norm == 0) {
    for (int i = 0; i < n; i++) {
      VectorXd v(p);
      for (int j = 0; j < p; j++) {
        v(j) = normal(gen);
      }
      A.row(i) = v.normalized();
    }
    return A;
  }
  
  // Normalize mu
  VectorXd mu_normalized = mu / std::sqrt(norm);
  
  // Calculate parameters for envelope
  double pp = static_cast<double>(p);
  VectorXd coe(4);
  coe << -pp*pp*lambda*lambda, 
         2*pp*(pp-2)*lambda*lambda, 
         4*pp-lambda*lambda*(pp-2)*(pp-2), 
         -4*(pp-1);
  
  // Find roots using Eigen's PolynomialSolver
  PolynomialSolver<double, 3> solver(coe);
  VectorXcd roots = solver.roots();
  VectorXd real_roots = roots.real();
  std::sort(real_roots.data(), real_roots.data() + real_roots.size());
  double b = real_roots(1);  // Take second root when sorted
  double minuslogM = std::log((1 + std::sqrt(1 - lambda * lambda / b)) / 2);
  double b2 = -1 + std::sqrt(1 / (1 - b));
  double b1 = b / (1 - b);
  // Generate samples using rejection sampling
  int count = 0;
  int Nt = 0;
  while (count < n) {
    VectorXd candidate(p);
    for (int j = 0; j < p; j++) {
      candidate(j) = normal(gen);
    }
    
    double mutz = candidate.dot(mu_normalized);
    double norm_candidate = std::sqrt(candidate.squaredNorm() + b1 * mutz * mutz);
    double mutx = mutz * (1 + b2) / norm_candidate;
    
    double PKBD = -std::log(1 - lambda * mutx);
    double mACG = std::log(1 - b * mutx * mutx);
    double unif = uniform(gen);
    double ratio = 0.5 * p * (PKBD + mACG + minuslogM);
    
    if (std::log(unif) < ratio) {
      candidate = (candidate + b2 * mutz * mu_normalized) / norm_candidate;
      A.row(count) = candidate;
      count++;
    }
    
    Nt++;
    if (Nt % 1000000 == 0) {
      throw std::runtime_error("Maximum iterations reached in sampling");
    }
  }
  
  return A;
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