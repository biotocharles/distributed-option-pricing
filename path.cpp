#include <vector>
#include <cmath>
#include <random> 
#include <numeric>
#include <iostream>
#include "Eigen/Dense"
#include "Eigen/Cholesky"
#include <iomanip>
#include <chrono>   
#include <mpi.h>
using namespace std;
using namespace Eigen;
const int K = 3; // quadratic basis
struct parameters
{
    double s0; //initial asset price
    double k; // strike price
    double r; //risk free rate
    double sigma; //volatility
    double T; //time to maturity
    int N; //time steps
    int M; //paths
};

//funciton to simulate a single path of stock prices with N + 1 time steps

vector <double> generatePath(const parameters params, mt19937& rng)
{
    double dt = params.T / params.N;
    double drift_term = (params.r - 0.5 * params.sigma * params.sigma) * dt;
    double vol_factor = params.sigma * sqrt(dt);

    vector <double> path(params.N + 1);
    path[0] = params.s0;

    // normal dist generator for zij
    normal_distribution<> norm_dist(0.0, 1.0);

    for (int i = 1; i <= params.N; i++)
    {
        double zij = norm_dist(rng);
        path[i] = path[i-1] * exp(drift_term + vol_factor * zij);
    }
    return path;
} 

struct simData
{
    vector <vector <double> > s_values; // stores all stock prices M x (N+1)
    vector <vector <double> > f_values; // stores all optimal option values (fi) M x (N+1)
    const parameters *params;
};

//function to initialize data structures
void initializeData(simData& data, const parameters& p)
{
    data.params = &p;
    //allocating memory for M rows
    data.s_values.resize(p.M);
    data.f_values.resize(p.M);

    //Allocate memory for N+1 time steps in each path
    for (int i = 0; i < p.M; i++)
    {
        data.s_values[i].resize(p.N + 1);
        data.f_values[i].resize(p.N + 1);
    }
    cout << "Allocated memory for " << p.M << "paths, each with " << p.N + 1 << "time steps" << endl;

}

void generateAllPaths(simData &data, const parameters &p)
{
    //generate s_values data (stock prices)

    // rng is initiated outside the loop, because every time the function calls it, a new seed is initialized.
    random_device rd;
    mt19937 rng(rd());
    
    for (int i = 0; i < p.M; i++)
    {
        data.s_values[i] = generatePath(p, rng);
    }
    cout << "Succesfully generated " << p.M << "unique price paths" << endl; 
    
}

vector <double> regression_solver(const vector < vector<double> > &X_data, const vector<double> &Y_vector)
{
    size_t n_prime = Y_vector.size();
    // 1. Safety Check: Ensure enough data points for regression
    if (n_prime < K) {
        return vector<double>(K, 0.0);
    }

    //converting c++ vectors to eigen matrices
    MatrixXd X_eigen(n_prime, K); //N' x K
    VectorXd Y_eigen(n_prime); // N'x 1

    for(size_t i = 0; i < n_prime; i++)
    {
        Y_eigen(i) = Y_vector[i];
        for (int k = 0; k < K; k++)
        {
            X_eigen(i, k) = X_data[i][k]; //basis vector of degree K (3)
        }
    }

    //calculate normal equations A and B where A = X^T * X and b = X^T * Y
    MatrixXd A = X_eigen.transpose() * X_eigen;
    VectorXd b = X_eigen.transpose() * Y_eigen;

    //solve the system A * beta = b using cholesky decomposition (LLT)
    //.llt performs the decompositon, .solve() performs forward and bacward s
    VectorXd beta_eigen = A.llt().solve(b);

    //convert eigen result back to cpp vector
    vector <double> beta_vector(K);
    for (int k = 0; k < K; k++) beta_vector[k] = beta_eigen(k);

    return beta_vector;
}

void backward_induction(simData &data)
{
    const parameters& p = *(data.params);
    double dt = p.T / (double)p.N;
    double discount_factor = exp(-p.r * dt); // e^(-r * Delta t)

    // BASE CASE: Initialize F_matrix at maturity (i=N)
    for (int j = 0; j < p.M; ++j) 
    {
        double stock_price = data.s_values[j][p.N];

        // Payoff: max(K - S_N, 0)
        data.f_values[j][p.N] = max(p.k - stock_price, 0.0);
    }

    //backward loop (iterating from N - 1 to 1)
    for (int i = p.N - 1; i >= 1; i--)
    {
        vector < vector <double> > X_data;
        vector <double> Y_vector;
        for (int j = 0; j < p.M; ++j) {
            double current_S = data.s_values[j][i];
            
            // Filter 1: Only use paths that are IN-THE-MONEY (S_i <= K)
            if (current_S <= p.k) { 
                
                // Y Vector (Response): Discounted optimal value from the next step (f_i+1)
                double next_f_value = data.f_values[j][i + 1];
                Y_vector.push_back(next_f_value * discount_factor);
                
                // X Matrix (Regressor): Basis functions (1, S, S^2)
                X_data.push_back({1.0, current_S, current_S * current_S});
            }
        }
        vector <double> beta = regression_solver(X_data, Y_vector);
        for (int j = 0; j < p.M; ++j) {
            double current_S = data.s_values[j][i];
            double immediate_payoff = max(p.k - current_S, 0.0);
            
            // Calculate Continuation Value (C_i) using the solved coefficients
            double continuation_value = beta[0] + beta[1] * current_S + beta[2] * current_S * current_S;
            
            // Optimal Decision: f_i = max(I_i, C_i)
            data.f_values[j][i] = max(immediate_payoff, continuation_value);
        }
    }
}
double calculate_final_price(const simData& data) 
{
    const parameters& p = *(data.params);
    double dt = p.T / (double)p.N;
    double final_discount_factor = exp(-p.r * dt); 
    double sum_f0 = 0.0;

    // The final price V0 is the average of the discounted f1 values.
    // The backward induction stops at i=1, so we discount that column one last time.
    for (int j = 0; j < p.M; ++j) 
    {
        // f_1 is the optimal option value at time 1
        sum_f0 += data.f_values[j][1] * final_discount_factor;
    }

    return sum_f0 / p.M;
}
double lsmc_worker_wrapper(const parameters& global_params, int rank, int start_path, int num_paths)
{
    // --- TEMPORARY MOCK RETURN for testing MPI flow ---
    // If you integrate your full LSMC code here, remove this mock line.
    return 2.45 + (double)rank * 0.01; 

}
int main(int argc, char* argv[])

{
    //MPI setup
    MPI_Init(&argc, &argv);
    int world_rank;
    int world_size;

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_rank < 2)
    {
        if (world_rank == 0)
        {
            cerr << "Error, need at least 2 processes, 1 Master and 1 Worker " << endl;
        }
        MPI_Finalize;
        return -1;
    }
    parameters global_params;

    //Master logic implementation
    if (world_rank == 0)
    {
        global_params = {100.0, 105.0, 0.05, 0.30, 1.0, 100, 500000};
        cout << "Master starting simulation with " << world_size - 1 << " workers" << endl;

        MPI_Bcast(&global_params, sizeof(parameters), MPI_BYTE, 0, MPI_COMM_WORLD); // proper solution to be implemented with MPI datatype. here MITBYTE used

        int num_workers = world_size - 1;
        int M_total = global_params.M;
        int paths_per_worker = M_total/num_workers;
        double total_price = 0.0;

        auto start_time = chrono::high_resolution_clock::now(); //start time

        for (int worker_rank = 1; worker_rank < world_size; worker_rank++)
        {
            int num_paths = paths_per_worker;

            //use MPI_Send to dispatfch workload to workers
            MPI_Send(&num_paths, 1, MPI_INT, worker_rank, 101, MPI_COMM_WORLD);
        }
        double worker_price;
        for (int worker_rank = 1; worker_rank < world_size; ++worker_rank)
        {
            
            // MPI_Recv: Receive Partial Result (g(Pi))
            MPI_Recv(&worker_price, 1, MPI_DOUBLE, worker_rank, 200, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total_price += worker_price;
        }
        auto end_time = chrono::high_resolution_clock::now(); //end time
        double final_option_price = total_price/ num_workers;
        auto duration = chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "\n--- PARALLEL LSMC RESULTS (Workers: " << num_workers << ") ---" << std::endl;
        std::cout << "Final Price (V0): $" << std::fixed << std::setprecision(6) << final_option_price << std::endl;
        std::cout << "Target Speedup Time: " << (double)33410.0 / num_workers << " ms" << std::endl;
        std::cout << "Parallel Computation Time: " << duration.count() << " milliseconds" << std::endl;
        std::cout << "---------------------------------------------" << std::endl;
    }

    //Worker logic implementation
    else 
    { 
        
        
        MPI_Bcast(&global_params, sizeof(parameters), MPI_BYTE, 0, MPI_COMM_WORLD);// Receive Broadcasted Parameters
        int num_paths;
        MPI_Recv(&num_paths, 1, MPI_INT, 0, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //Receive Workload Parameters
        double subset_price = run_lsmc_subset_logic(global_params, world_rank, 0, num_paths);// Execute LSMC
        MPI_Send(&subset_price, 1, MPI_DOUBLE, 0, 200, MPI_COMM_WORLD); //Send Result
    }
    MPI_Finalize();
    return 0;



    // Define problem inputs (Accuracy Test Case)
    // Running 500k paths over 100 steps ensures measurable slowness.
    parameters op { 
        100.0,  // S0 (Initial Asset Price)
        105.0,  // K (Strike Price)
        0.05,   // r (Risk-Free Rate)
        0.30,   // sigma (Volatility)
        1.0,    // T (Time to Maturity: 1 year)
        100,    // N (Time Steps)
        500000  // M (Paths)
    };

    simData data;
    
    // --- 1. INITIALIZATION ---
    cout << "Starting Sequential LSMC Verification (M=" << op.M << ", N=" << op.N << ")" << endl;
    initializeData(data, op);
    
    // 2. PATH SIMULATION (Partially time consuming, but separated for clarity)
    generateAllPaths(data, op);
    // Timing the Backward Induction and Final Calculation (The O(MNK^2) part)
    auto start_time = chrono::high_resolution_clock::now(); //start
     
    backward_induction(data);// BACKWARD INDUCTION and REGRESSION
    double option_price = calculate_final_price(data);     // FINAL PRICE CALCULATION
    
    auto end_time = chrono::high_resolution_clock::now();//end
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

    cout << "\n-----------------------------------------------------" << endl;
    cout << "Final Price (V0): $" << fixed << setprecision(6) << option_price << endl;
    cout << "Min Intrinsic Value: $" << fixed << setprecision(4) << max(op.k - op.s0, 0.0) << endl;
    cout << "-----------------------------------------------------" << endl;
    cout << "Total LSMC Computation Time: " << duration.count() << " milliseconds" << endl;
    cout << "-----------------------------------------------------" << endl;

    return 0;
}