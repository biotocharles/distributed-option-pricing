# Distributed LSMC Option Pricing Engine
A high-performance implementation of the Least-Squares Monte Carlo (LSMC) algorithm for pricing American Options. This project leverages C++17, the Eigen Linear Algebra Library, and MPI (Message Passing Interface) to transform a computationally intensive sequential process into a scalable, distributed system.

Performance Highlights
Sequential Benchmark: 33,410 ms (500k paths, 100 steps).

Parallel Execution (3 Workers): 2,920 ms.

Observed Speedup: 11.44x (Superlinear efficiency).

Numerical Accuracy: Verified against sequential benchmarks (~7.44V 
0
​	
 ).

Project Architecture
The system is built on a Master-Worker Architecture designed to handle the O(MNK 
2
 ) complexity of the Longstaff-Schwartz method.

Phase 1: Numerical Core

Path Simulation: Geometric Brownian Motion (GBM) path generation.

Regression Engine: High-performance polynomial regression using Eigen's Cholesky Decomposition to solve the normal equations (X 
T
 Xβ=X 
T
 Y).

Optimal Stopping: Backward induction logic to determine exercise vs. continuation values.

Phase 2: MPI Integration

Data Decomposition: Static partitioning of Monte Carlo paths across available worker nodes.

Communication: Master-Worker protocol using MPI_Send, MPI_Recv, and MPI_Bcast.

Aggregation: Distributed partial sums collected by the Master for final global expectation calculation.

Project Structure
mpi_paths.cpp: Entry point handling MPI environment and Master-Worker logic.

path.cpp: Implementation of backward induction and regression solvers.

Requirements
Compiler: mpic++ (supporting C++17 or later).

Libraries: Eigen 3.4+.

Runtime: MPI (OpenMPI or MPICH).

Build & Run
1. Compilation

Use the MPI wrapper and enable optimizations:

Bash
mpic++ -std=gnu++17 -O3 path.cpp -o lsmc_dist
mpic++ -std=gnu++17 -O3 mpi_paths.cpp 
2. Execution

Run with N processes (1 Master + N−1 Workers):

Bash
mpirun -np 4 ./lsmc_dist
