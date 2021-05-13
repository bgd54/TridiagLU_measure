// Written by Gabor Daniel Balogh, Pazmany Peter Catholic University,
// balogh.gabor.daniel@itk.ppke.hu, 2021
// Implementation of a struct bounding together MPI related parameters.

#ifndef MPI_SOLVER_PARAMS_HPP
#define MPI_SOLVER_PARAMS_HPP

#include <algorithm>
#include <mpi.h>
#include <vector>


struct MpiSolverParams {
  // This will be a communicator for x dimension. Separate communicator that
  // includes every node calculating the same set of equations as the current
  // node.
  MPI_Comm communicator;
  MPI_Group cart_group;
  MPI_Group neighbours_group;

  // The number of MPI processes in each dimension. It is `num_dims` large. It
  // won't be owned.
  const int *num_mpi_procs;

  // The coordinates of the current MPI process in the cartesian mesh.
  std::vector<int> mpi_coords;

  MpiSolverParams(MPI_Comm cartesian_communicator, int num_dims,
                  int *num_mpi_procs_)
      : num_mpi_procs(num_mpi_procs_), mpi_coords(num_dims) {
    int cart_rank;
    MPI_Comm_rank(cartesian_communicator, &cart_rank);
    MPI_Cart_coords(cartesian_communicator, cart_rank, num_dims,
                    this->mpi_coords.data());

    constexpr int equation_dim  = 2;
    std::vector<int> neighbours = {cart_rank};
    int mpi_coord               = this->mpi_coords[equation_dim];
    // Collect the processes in the same row/column
    for (int i = 1;
         i <= std::max(num_mpi_procs[equation_dim] - mpi_coord - 1, mpi_coord);
         ++i) {
      int prev, next;
      MPI_Cart_shift(cartesian_communicator, equation_dim, i, &prev, &next);
      if (i <= mpi_coord) {
        neighbours.push_back(prev);
      }
      if (i + mpi_coord < num_mpi_procs[equation_dim]) {
        neighbours.push_back(next);
      }
    }

    // This is needed, otherwise the communications hang
    std::sort(neighbours.begin(), neighbours.end());

    // Create new communicator for neighbours
    // MPI_Group cart_group;
    MPI_Comm_group(cartesian_communicator, &this->cart_group);
    // MPI_Group neighbours_group;
    MPI_Group_incl(this->cart_group, neighbours.size(), neighbours.data(),
                   &this->neighbours_group);
    MPI_Comm_create(cartesian_communicator, this->neighbours_group,
                    &this->communicator);
  }

  ~MpiSolverParams() {
    MPI_Group_free(&this->cart_group);
    MPI_Group_free(&this->neighbours_group);
    MPI_Comm_free(&this->communicator);
  }
};

#endif /* ifndef MPI_SOLVER_PARAMS_HPP */
