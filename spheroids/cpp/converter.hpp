#ifndef ARRAY_CONVERTERS_HPP
#define ARRAY_CONVERTERS_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <armadillo>
#include <cstring> // for std::memcpy

namespace py = pybind11;

// Convert a Python array to an arma::vec
inline arma::vec pyarray_to_arma_vec(const py::array_t<double> &arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim == 1) {
        arma::vec v((double*)buf.ptr, (size_t)buf.shape[0], false, false);
        return v;
    } else if (buf.ndim == 2) {
        size_t length = (size_t)(buf.shape[0] * buf.shape[1]);
        arma::vec v((double*)buf.ptr, length, false, false);
        return v;
    } else {
        throw std::runtime_error("Expected a 1D or 2D vector.");
    }
}

// Convert a Python array to an arma::mat
inline arma::mat pyarray_to_arma_mat(const py::array_t<double> &arr) {
    py::buffer_info buf = arr.request();
    // Interpret the data as (d x n) column-major, then transpose
    arma::mat M((double*)buf.ptr, buf.shape[1], buf.shape[0], false, false);
    return M.t(); // Now M is n x d, matching the Python layout
}

// Convert a Python array to an arma::vec with copy
inline arma::vec pyarray_to_arma_vec_copy(const py::array_t<double> &arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim == 1) {
        return arma::vec(static_cast<double*>(buf.ptr), buf.shape[0], true); // true = copy data
    } else if (buf.ndim == 2) {
        size_t length = (size_t)(buf.shape[0] * buf.shape[1]);
        return arma::vec(static_cast<double*>(buf.ptr), length, true); // true = copy data
    } else {
        throw std::runtime_error("Expected a 1D or 2D vector.");
    }
}

// Convert a Python array to an arma::mat with copy
inline arma::mat pyarray_to_arma_mat_copy(const py::array_t<double> &arr) {
    py::buffer_info buf = arr.request();
    // Create a temporary matrix with the Python data
    arma::mat temp(static_cast<double*>(buf.ptr), buf.shape[1], buf.shape[0], false, false);
    // Return a copy of the transposed matrix
    return arma::mat(temp.t()); // true copy of data in row-major format
}

// Convert an arma::vec to a Python array
inline py::array_t<double> arma_vec_to_pyarray(const arma::vec &v) {
    py::array_t<double> result(v.n_elem);
    py::buffer_info buf = result.request();
    std::memcpy(buf.ptr, v.memptr(), v.n_elem * sizeof(double));
    return result;
}

// Convert an arma::vec to a Python array with copy
inline py::array_t<double> arma_vec_to_pyarray_copy(const arma::vec &v) {
    py::array_t<double> result(v.n_elem);
    py::buffer_info buf = result.request();
    std::memcpy(buf.ptr, v.memptr(), v.n_elem * sizeof(double));
    return result;
}

// Convert an arma::mat to a Python array
inline py::array_t<double> arma_mat_to_pyarray(const arma::mat &M) {
    std::vector<py::ssize_t> shape = {
        (py::ssize_t)M.n_rows,
        (py::ssize_t)M.n_cols
    };
    std::vector<py::ssize_t> strides = {
        (py::ssize_t)(sizeof(double)),
        (py::ssize_t)(sizeof(double) * M.n_rows)
    };
    return py::array_t<double>(
        py::buffer_info(
            const_cast<double*>(M.memptr()),
            sizeof(double),
            py::format_descriptor<double>::format(),
            2,
            shape,
            strides
        )
    );
}

inline py::array_t<double> arma_mat_to_pyarray_copy(const arma::mat &M){
    py::array_t<double> result({(py::ssize_t)M.n_rows, (py::ssize_t)M.n_cols});
    auto buf = result.request();
    double* dst = static_cast<double*>(buf.ptr);

    // Copy in row-major order
    for (size_t r = 0; r < M.n_rows; ++r) {
        for (size_t c = 0; c < M.n_cols; ++c) {
            dst[r * M.n_cols + c] = M(r, c);
        }
    }
    return result;
}

inline arma::mat Moebius_S(const arma::mat &X, arma::vec mu, double rho){
  
  arma::mat Y = (1-rho*rho)*(X.each_row() + rho*mu.t());
  Y = Y.each_col()/(1+2*rho*X*mu+rho*rho);
  Y = Y.each_row() + rho*mu.t();
  
  return Y;
}

#endif // ARRAY_CONVERTERS_HPP
