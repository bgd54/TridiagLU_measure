#ifndef TRID_COMPARE_UTILS_HPP_INCLUDED
#define TRID_COMPARE_UTILS_HPP_INCLUDED
#include <functional>
#include <limits>
#include <vector>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <cassert>
#include <numeric>
#include <random>
#include <omp.h>
#include <thread>
#include <mpi.h>
#include "tridiagLU.h"

#include "mpi_params.hpp"

template <typename Float> class MeshLoader {
protected:
  size_t _solve_dim;
  std::vector<int> _dims;
  std::vector<Float> _a, _b, _c, _d, _u;

public:
  MeshLoader(const std::filesystem::path &file_name);

  size_t solve_dim() const { return _solve_dim; }
  const std::vector<int> &dims() const { return _dims; }
  const std::vector<Float> &a() const { return _a; }
  const std::vector<Float> &b() const { return _b; }
  const std::vector<Float> &c() const { return _c; }
  const std::vector<Float> &d() const { return _d; }
  const std::vector<Float> &u() const { return _u; }
  int num_systems() const {
    int n_sys = 1;
    // Calculate size needed for aa, cc and dd arrays
    for (size_t i = 0; i < _dims.size(); i++) {
      if (i != _solve_dim) {
        n_sys *= _dims[i];
      }
    }
    return n_sys;
  }

private:
  void load_array(std::ifstream &f, size_t num_elements,
                  std::vector<Float> &array);
};

template <typename Float> class RandomMesh {
  size_t _solve_dim;
  std::vector<int> _dims;
  std::vector<Float> _a, _b, _c, _d;

public:
  RandomMesh(const std::vector<int> &dims, size_t solvedim,
             MpiSolverParams &params);

  size_t solve_dim() const { return _solve_dim; }
  const std::vector<int> &dims() const { return _dims; }
  const std::vector<Float> &a() const { return _a; }
  const std::vector<Float> &b() const { return _b; }
  const std::vector<Float> &c() const { return _c; }
  const std::vector<Float> &d() const { return _d; }
  int num_systems() const {
    int n_sys = 1;
    // Calculate size needed for aa, cc and dd arrays
    for (size_t i = 0; i < _dims.size(); i++) {
      if (i != _solve_dim) {
        n_sys *= _dims[i];
      }
    }
    return n_sys;
  }
};


template <typename Float>
MeshLoader<Float>::MeshLoader(const std::filesystem::path &file_name)
    : _a{}, _b{}, _c{}, _d{}, _u{} {
  std::ifstream f(file_name);
  assert(f.good() && "Couldn't open file");
  size_t num_dims;
  f >> num_dims >> _solve_dim;
  // Load sizes along the different dimensions
  size_t num_elements = 1;
  for (size_t i = 0; i < num_dims; ++i) {
    int size;
    f >> size;
    _dims.push_back(size);
    num_elements *= static_cast<size_t>(size);
  }
  // Load arrays
  load_array(f, num_elements, _a);
  load_array(f, num_elements, _b);
  load_array(f, num_elements, _c);
  load_array(f, num_elements, _d);
  if (std::is_same<Float, double>::value) {
    load_array(f, num_elements, _u);
  } else {
    std::string tmp;
    // Skip the line with the double values
    std::getline(f >> std::ws, tmp);
    load_array(f, num_elements, _u);
  }
}

template <typename Float>
void MeshLoader<Float>::load_array(std::ifstream &f, size_t num_elements,
                                   std::vector<Float> &array) {
  array.reserve(num_elements);
  for (size_t i = 0; i < num_elements; ++i) {
    // Load with the larger precision, then convert to the specified type
    double value;
    f >> value;
    array.push_back(value);
  }
}

template <typename Float>
RandomMesh<Float>::RandomMesh(const std::vector<int> &dims, size_t solvedim,
                              MpiSolverParams &params)
    : _solve_dim(solvedim), _dims(dims), _a{}, _b{}, _c{}, _d{} {
  assert(_solve_dim < _dims.size() && "solve dim greater than number of dims");

  size_t num_elements = static_cast<size_t>(
      std::accumulate(_dims.begin(), _dims.end(), 1, std::multiplies<int>()));
  size_t inner_size = static_cast<size_t>(std::accumulate(
      _dims.begin(), _dims.begin() + static_cast<int>(_solve_dim), 1,
      std::multiplies<int>()));
  size_t outer_size = num_elements / inner_size / dims[solvedim];
  _a.resize(num_elements);
  _b.resize(num_elements);
  _c.resize(num_elements);
  _d.resize(num_elements);
  // Setting halo elements
  int rank     = 0;
  int num_proc = 0;
  MPI_Comm_size(params.communicator, &num_proc);
  MPI_Comm_rank(params.communicator, &rank);
#pragma omp parallel for
  for (size_t i = 0; i < outer_size; ++i) {
    size_t base = i * inner_size * dims[solvedim];
#pragma omp simd
    for (size_t j = 0; j < inner_size; ++j) {
      if (rank)
        _a[base + j + inner_size * 0] = 0.3;
      else
        _a[base + j + inner_size * 0] = 0.0;
      if (!rank)
        _b[base + j + inner_size * 0] = 1.0;
      else
        _b[base + j + inner_size * 0] = 0.6;
      _c[base + j + inner_size * 0] = 0.1;
    }
    for (int n = 1; n < dims[solvedim] - 1; ++n) {
#pragma omp simd
      for (size_t j = 0; j < inner_size; ++j) {
        _a[base + j + inner_size * n] = 0.3;
        _b[base + j + inner_size * n] = 0.6;
        _c[base + j + inner_size * n] = 0.1;
      }
    }

#pragma omp simd
    for (size_t j = 0; j < inner_size; ++j) {
      _a[base + j + inner_size * (dims[solvedim] - 1)] = 0.3;
      if (rank == num_proc - 1)
        _b[base + j + inner_size * (dims[solvedim] - 1)] = 1.0;
      else
        _b[base + j + inner_size * (dims[solvedim] - 1)] = 0.6;
      if (rank != num_proc - 1)
        _c[base + j + inner_size * (dims[solvedim] - 1)] = 0.1;
      else
        _c[base + j + inner_size * (dims[solvedim] - 1)] = 0.0;
    }
  }
#pragma omp parallel
  {
    std::mt19937_64 gen(static_cast<size_t>(omp_get_thread_num()));
    std::uniform_real_distribution<Float> rhs_dist;
#pragma omp for
    for (int i = 0; i < num_elements; ++i) {
      _d[i] = rhs_dist(gen);
    }
  }
}

inline void print_local_sizes(int rank, int num_proc, const int *mpi_dims,
                              const std::vector<int> &mpi_coords,
                              const std::vector<int> &local_sizes) {
  std::string idx    = std::to_string(mpi_coords[0]),
              dims   = std::to_string(local_sizes[0]),
              m_dims = std::to_string(mpi_dims[0]);
  for (size_t j = 1; j < local_sizes.size(); ++j) {
    idx += "," + std::to_string(mpi_coords[j]);
    dims += "x" + std::to_string(local_sizes[j]);
    m_dims += "x" + std::to_string(mpi_dims[j]);
  }
  std::string header =
      "########## Local decomp sizes {" + m_dims + "} ##########";
#ifndef NDEBUG
  for (int i = 0; i < num_proc; ++i) {
    // Print the outputs
    if (i == rank) {
      if (rank == 0) {
        std::cout << header << "\n";
      }
      std::cout << "# Rank " << i << "(" + idx + "){" + dims + "}\n";
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    MPI_Barrier(MPI_COMM_WORLD);
  }
#else
  if (rank == 0) {
    std::cout << header << "\n";
    std::cout << "# Rank " << rank << "(" + idx + "){" + dims + "}\n";
  }
#endif

  if (rank == 0) {
    std::cout << std::string(header.size(), '#') << "\n";
  }
}

void print_header(int rank, int ndims, int num_proc, int strategy,
                  const int *dims, std::string executable) {
  if (rank == 0) {
    std::string fname = executable;
    fname             = fname.substr(fname.rfind("/") + 1);
    std::cout << fname << " " << ndims << "DS0NP" << num_proc
              << (strategy ? " Gather-Scatter " : " JACOBI ");
    std::cout << " {" << dims[0];
    for (size_t i = 1; i < ndims; ++i)
      std::cout << "x" << dims[i];
    std::cout << "}\n";
  }
}


inline std::ostream &operator<<(std::ostream &o, const TridiagLU &c) {
  return o << c.reducedsolvetype << "\n evaluate_norm: " << c.evaluate_norm
           << "\n maxiter: " << c.maxiter << "\n atol: " << c.atol
           << "\n rtol: " << c.rtol << "\n exititer: " << c.exititer
           << "\n verbose: " << c.verbose << "\n total_time " << c.total_time
           << "\n";
}

inline void printarr(double *arr, size_t N) {
  std::cout << "[";
  for (size_t i = 0; i < N; ++i) {
    std::cout << " " << arr[i];
  }
  std::cout << " ]";
}

template <typename Callable> void report_per_ranks(const Callable &reporter) {
  // For the debug prints
  int rank, num_proc;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  for (int i = 0; i < num_proc; ++i) {
    // Print the outputs
    if (i == rank) {
      std::cout << "##########################\n"
                << "Rank " << i << "\n"
                << "##########################\n";
      reporter();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

double CalculateError(const double *a, /*!< Array of subdiagonal elements */
                      const double *b, /*!< Array of diagonal elements */
                      const double *c, /*!< Array of superdiagonal elements */
                      const double *y, /*!< Right-hand side */
                      const double *x, /*!< Solution */
                      int N,           /*!< Local size of system */
                      int Ns,          /*!< Number of systems */
                      MPI_Comm *communicator) {
  int nproc, rank;
  MPI_Comm_size(*communicator, &nproc);
  MPI_Comm_rank(*communicator, &rank);
  double error = 0, norm = 0;
  int i, d;
  double xp1, xm1; /* solution from neighboring processes */

  for (d = 0; d < Ns; d++) {
    xp1 = 0;
    if (nproc > 1) {
      MPI_Request request = MPI_REQUEST_NULL;
      if (rank != nproc - 1)
        MPI_Irecv(&xp1, 1, MPI_DOUBLE, rank + 1, 1738, *communicator, &request);
      if (rank) MPI_Send(&x[d], 1, MPI_DOUBLE, rank - 1, 1738, *communicator);
      MPI_Wait(&request, MPI_STATUS_IGNORE);
    }

    xm1 = 0;
    if (nproc > 1) {
      MPI_Request request = MPI_REQUEST_NULL;
      if (rank)
        MPI_Irecv(&xm1, 1, MPI_DOUBLE, rank - 1, 1739, *communicator, &request);
      if (rank != nproc - 1)
        MPI_Send(&x[d + (N - 1) * Ns], 1, MPI_DOUBLE, rank + 1, 1739,
                 *communicator);
      MPI_Wait(&request, MPI_STATUS_IGNORE);
    }

    error = 0;
    norm  = 0;
    for (i = 0; i < N; i++) {
      double val = 0;
      if (i == 0)
        val += a[i * Ns + d] * xm1;
      else
        val += a[i * Ns + d] * x[(i - 1) * Ns + d];
      val += b[i * Ns + d] * x[i * Ns + d];
      if (i == N - 1)
        val += c[i * Ns + d] * xp1;
      else
        val += c[i * Ns + d] * x[(i + 1) * Ns + d];
      val = y[i * Ns + d] - val;
      error += val * val;
      norm += y[i * Ns + d] * y[i * Ns + d];
    }
  }

  double global_error = 0, global_norm = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  if (nproc > 1)
    MPI_Allreduce(&error, &global_error, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
  else
    global_error = error;
  if (nproc > 1)
    MPI_Allreduce(&norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  else
    global_norm = norm;
  if (global_norm != 0.0) global_error /= global_norm;

  return (global_error);
}


#endif /* ifndef TRID_COMPARE_UTILS_HPP_INCLUDED */
