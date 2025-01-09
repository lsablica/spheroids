library(RcppArmadillo)
library(Rcpp)
sourceCpp(file = "~/Documents/GitHub/spheroids/src/spheroids/cpp/test.cpp")



# Test the function
# N = 25, K = 5, D = 4
set.seed(123)
X = matrix(rnorm(100), nrow = 25)
X = X/sqrt(rowSums(X^2))

mu_matrix = matrix(rnorm(20), nrow = 4)
mu_matrix = mu_matrix/sqrt(rowSums(mu_matrix^2))

rho_vector = runif(5)

beta_matrix = matrix(rnorm(125), nrow = 25)
beta_matrix = exp(beta_matrix)/rowSums(exp(beta_matrix))

mu_matrix_copy = mu_matrix + 0 
rho_vector_copy = rho_vector + 0

M_step_PKBD(X, beta_matrix, mu_matrix_copy, rho_vector_copy)
M_step_PKBD1(X, beta_matrix[,1], mu_matrix[,1], rho_vector[1], 25, 4)
M_step_PKBD1(X, beta_matrix[,2], mu_matrix[,2], rho_vector[2], 25, 4)
M_step_PKBD1(X, beta_matrix[,3], mu_matrix[,3], rho_vector[3], 25, 4)
M_step_PKBD1(X, beta_matrix[,4], mu_matrix[,4], rho_vector[4], 25, 4)
M_step_PKBD1(X, beta_matrix[,5], mu_matrix[,5], rho_vector[5], 25, 4)


M_step_spcauchy(X, beta_matrix, mu_matrix_copy, rho_vector_copy)
M_step_spcauchy1(X, beta_matrix[,1], 25, 4)
M_step_spcauchy1(X, beta_matrix[,2], 25, 4)
M_step_spcauchy1(X, beta_matrix[,3], 25, 4)
M_step_spcauchy1(X, beta_matrix[,4], 25, 4)
M_step_spcauchy1(X, beta_matrix[,5], 25, 4)


logLik_PKBD_intern(X, mu_matrix, rho_vector)
rho_mat = matrix(rep(rho_vector,25), 25, byrow = TRUE) 
ll = log(1-rho_mat^2) - (4/2)*log(1+ rho_mat ^2-2*rho_mat* X %*% mu_matrix)
logLik_PKBD_intern(X, mu_matrix, rho_vector) - ll


logLik_spcauchy_intern(X, mu_matrix, rho_vector)
ll2 = (4-1)*log(1-rho_mat^2) - (4-1)*log(1+ rho_mat ^2-2*rho_mat* X %*% mu_matrix)
logLik_spcauchy_intern(X, mu_matrix, rho_vector) - ll2

pi_vec = rep(0.2,5)

E_step_test(X, beta_matrix, rho_vector, mu_matrix, pi_vec, 5, minalpha = 0.01, n = 25, p = 4, lik = -1e10, reltol = 1e-6) 

pi_mat = matrix(rep(pi_vec,25), 25, byrow = TRUE) 
B = logLik_PKBD_intern(X, mu_matrix, rho_vector) + log(pi_mat)
lse = log(rowSums(exp(B)))
sum(lse)
exp(B - lse)
colMeans(exp(B - lse))


t(apply(exp(B - lse), 1, function(x) x==max(x)+0))*1
colMeans(t(apply(exp(B - lse), 1, function(x) x==max(x)+0))*1)


D = rbind(circlus::rpkbd(100, 0.97, c(1,0,0,0)), circlus::rpkbd(50, 0.95, c(0,1,0,0)))
EM(D, 2, "softmax", "pkbd", 0.01, maxiter = 100, reltol = 1e-12)

D = rbind(circlus::rspcauchy(100, 0.97, c(1,0,0,0)), circlus::rspcauchy(50, 0.95, c(0,1,0,0)))
EM(D, 2, "softmax", "spcauchy", 0.01, maxiter = 100, reltol = 1e-15)
