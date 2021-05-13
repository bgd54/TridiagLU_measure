#include <iostream>
#include <limits>
#include <mpi.h>
#include <ratio>
#include <unistd.h>
#include <vector>
#include <cstring>
#include <array>
#include <chrono>

#include "utils.hpp"
#include "tridiagLU.h"
#include "mpi_params.hpp"

// This block enables to compile the code with and without the likwid header in
// place
#ifdef LIKWID_PERFMON
#  include <likwid.h>
#else
#  define LIKWID_MARKER_INIT
#  define LIKWID_MARKER_THREADINIT
#  define LIKWID_MARKER_SWITCH
#  define LIKWID_MARKER_REGISTER(regionTag)
#  define LIKWID_MARKER_START(regionTag)
#  define LIKWID_MARKER_STOP(regionTag)
#  define LIKWID_MARKER_CLOSE
#  define LIKWID_MARKER_GET(regionTag, nevents, events, time, count)
#endif

void usage(const char *name) {
  std::cerr << "Usage:\n";
  std::cerr << "\t" << name
            << " [-x nx -y ny -z nz -n num_iterations -m method_idx]"
            << std::endl;
  std::cerr << "\tdefault values: -x 256 -y 1 -z 1 -n 1 -m 0" << std::endl;
  std::cerr << "\t\tm values:\n\t\t 0 - JACOBI\n\t\t 1 - LUGS" << std::endl;
}

void run_tridiagLU(RandomMesh<double> &mesh, int num_iters,
                   MpiSolverParams &params, int strat) {
  std::vector<double> a(mesh.a());
  std::vector<double> b(mesh.b());
  std::vector<double> c(mesh.c());
  std::vector<double> d(mesh.d());
  int nproc, rank, nruns = num_iters;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  TridiagLU context;
  int ierr = tridiagLUInit(&context, &params.communicator);
  assert(ierr == 0 && "Problem during initialization of TridiagLU");
  if (params.mpi_coords[0] == 0 && params.mpi_coords[1] == 0)
    context.verbose = 1;
  // context.maxiter    = std::numeric_limits<int>::max();
  if (strat) {
    strcpy(context.reducedsolvetype, _TRIDIAG_GS_);
  }
  context.rtol                   = 1e-12;
  std::array<double, 6> runtimes = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  LIKWID_MARKER_THREADINIT;
  LIKWID_MARKER_INIT;
  // run first one without time measure, but printing iteration info
  tridiagLU(a.data(), b.data(), c.data(), d.data(),
            mesh.dims()[mesh.solve_dim()], mesh.num_systems(), &context,
            &params.communicator);
  context.verbose = 0;

  // Solve the equations
  double error =
      CalculateError(mesh.a().data(), mesh.b().data(), mesh.c().data(),
                     mesh.d().data(), d.data(), mesh.dims()[mesh.solve_dim()],
                     mesh.num_systems(), &params.communicator);
  while (num_iters--) {
    /* Copy the original values */
    auto tp1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < a.size(); ++i) {
      a[i] = mesh.a()[i];
      b[i] = mesh.b()[i];
      c[i] = mesh.c()[i];
      d[i] = mesh.d()[i];
    }
    runtimes[5] += std::chrono::duration_cast<std::chrono::microseconds>(
                       std::chrono::high_resolution_clock::now() - tp1)
                       .count() / 1000000.0;
    /* Solve the system */
    MPI_Barrier(MPI_COMM_WORLD);
    // BEGIN_PROFILING("tridiagLU");
    tridiagLU(a.data(), b.data(), c.data(), d.data(),
              mesh.dims()[mesh.solve_dim()], mesh.num_systems(), &context,
              &params.communicator);
    // END_PROFILING("tridiagLU");
    MPI_Barrier(MPI_COMM_WORLD);
#ifndef NDEBUG
    error +=
        CalculateError(mesh.a().data(), mesh.b().data(), mesh.c().data(),
                       mesh.d().data(), d.data(), mesh.dims()[mesh.solve_dim()],
                       mesh.num_systems(), &params.communicator);
#endif

    /* Add the walltimes to the cumulative total */
    runtimes[0] += context.total_time;
    runtimes[1] += context.stage1_time;
    runtimes[2] += context.stage2_time;
    runtimes[3] += context.stage3_time;
    runtimes[4] += context.stage4_time;
  }
  runtimes[0] += runtimes[5];
  LIKWID_MARKER_CLOSE;

  /* Calculate maximum value of walltime across all processes */
  // MPI_Allreduce(MPI_IN_PLACE, &runtimes[0], 5, MPI_DOUBLE, MPI_MAX,
  //               MPI_COMM_WORLD);

  /* Print results */
  std::vector<double> times(runtimes.size() * nproc, 0);
  MPI_Gather(runtimes.data(), runtimes.size(), MPI_DOUBLE, times.data(),
             runtimes.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if (ierr == -1) printf("Error - system is singular on process %d\t", rank);
#ifndef NDEBUG
  if (mesh.num_systems() < 10 && mesh.dims()[mesh.solve_dim()] < 20) {
    for (int i = 0; i < nproc; ++i) {
      if (rank == i) {
        for (int n = 0; n < mesh.dims()[mesh.solve_dim()]; ++n) {
          int ns = mesh.num_systems();
          for (int idx = 0; idx < ns; ++idx) {
            printf(" %.2lf | %.2lf | %.2lf | %.2lf = %.2lf ||",
                   mesh.a()[n * ns + idx], mesh.b()[n * ns + idx],
                   mesh.c()[n * ns + idx], d[n * ns + idx],
                   mesh.d()[n * ns + idx]);
          }
          printf("\n");
        }
        printf("------------------------%d------------------------\n", rank);
      }
      MPI_Barrier(MPI_COMM_WORLD);
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }
#endif
  if (!rank) {
    const char *labels[] = {"Total",  "Stage1", "Stage2",
                            "Stage3", "Stage4", "Copy"};
#ifndef NDEBUG
    printf("# ERROR: %E\n",
           error / nruns /
               mesh.num_systems()); // Num systems here is not correct
    for (int i = 0; i < nproc; ++i) {
      std::cout << i << ": {";
      for (int j = 0; j < runtimes.size(); ++j) {
        std::cout << times[i * runtimes.size() + j] << " ";
      }
      std::cout << "}\n";
    }
#else
    printf("# ERROR: %E\n",
           error / mesh.num_systems()); // Num systems here is not correct
#endif
    for (int i = 0; i < runtimes.size(); ++i) {
      double mean = 0.0;
      double max  = std::numeric_limits<double>::min();
      double min  = std::numeric_limits<double>::max();
      for (int j = 0; j < nproc; ++j) {
        double t = times[j * runtimes.size() + i];
        mean += t;
        max = std::max(max, t);
        min = std::min(min, t);
      }
      mean          = mean / nproc;
      double stddev = 0.0;
      for (int j = 0; j < nproc; ++j) {
        stddev += (times[j * runtimes.size() + i] - mean) *
                  (times[j * runtimes.size() + i] - mean);
      }
      stddev = std::sqrt(stddev / nproc);

      std::cout << "\t" << labels[i] << " walltime = ";
      std::cout << min << "; " << max << "; " << mean << "; " << stddev
                << ";\n";
    }
  }
}

std::string executable;

void test_LU_with_generated(const int *global_dims, size_t ndims, int num_iters,
                            std::vector<int> solver_strats, int nproc_x) {
  int num_proc, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  constexpr int solvedim = 2;

  // Create rectangular grid
  std::vector<int> mpi_dims(ndims, 0), periods(ndims, 0);
  mpi_dims[solvedim] = nproc_x > 0 ? nproc_x : num_proc;
  MPI_Dims_create(num_proc, ndims, mpi_dims.data());
  // Create communicator for grid
  MPI_Comm cart_comm;
  MPI_Cart_create(MPI_COMM_WORLD, ndims, mpi_dims.data(), periods.data(), 0,
                  &cart_comm);

  MpiSolverParams params(cart_comm, ndims, mpi_dims.data());

  // The size of the local domain.
  std::vector<int> local_sizes(global_dims, global_dims + ndims);
  // The starting indices of the local domain in each dimension.
  for (size_t i = 0; i < local_sizes.size(); ++i) {
    const int global_dim = global_dims[i];
    size_t domain_offset = params.mpi_coords[i] * (global_dim / mpi_dims[i]);
    local_sizes[i]       = params.mpi_coords[i] == mpi_dims[i] - 1
                               ? global_dim - domain_offset
                               : global_dim / mpi_dims[i];
  }

  print_local_sizes(rank, num_proc, mpi_dims.data(), params.mpi_coords,
                    local_sizes);

  RandomMesh<double> mesh(local_sizes, solvedim, params);
  for (int i : solver_strats) {
    print_header(rank, ndims, num_proc, i, global_dims, executable);
    run_tridiagLU(mesh, num_iters, params, i);
  }
}

int main(int argc, char **argv) {
  auto rc = MPI_Init(&argc, &argv);
  if (rc != MPI_SUCCESS) {
    printf("Error starting MPI program. Terminating.\n");
    MPI_Abort(MPI_COMM_WORLD, rc);
  }
  int num_proc, rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

  int dims[]             = {256, 1, 1};
  constexpr size_t ndims = 3;
  // constexpr int solvedim = 0;
  int num_iters = 1;
  // int mpi_parts_in_s = 0;
  int mpi_parts_in_s = 0; // 0 means ALL
  std::vector<int> solver_strats;
  {
    int opt;
    while ((opt = getopt(argc, argv, "x:y:z:n:m:p:")) != -1) {
      switch (opt) {
      case 'x': dims[0] = atoi(optarg); break;
      case 'y': dims[1] = atoi(optarg); break;
      case 'z': dims[2] = atoi(optarg); break;
      case 'n': num_iters = atoi(optarg); break;
      case 'm': solver_strats.push_back(atoi(optarg)); break;
      case 'p': mpi_parts_in_s = atoi(optarg); break;
      default:
        if (rank == 0) usage(argv[0]);
        return 2;
        break;
      }
    }
  }
  if (solver_strats.size() == 0) {
    solver_strats.push_back(0);
  }
  executable = argv[0];

  MPI_Barrier(MPI_COMM_WORLD);
  test_LU_with_generated(dims, ndims, num_iters, solver_strats, mpi_parts_in_s);
  // PROFILE_REPORT();
  MPI_Finalize();
  return 0;
}
