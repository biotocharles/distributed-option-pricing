import time
import numpy as np
from pyspark.sql import SparkSession

def worker_lsmc_task(partition_id, num_paths, params):
    S0 = params['s0']
    K_strike = params['k']
    r = params['r']
    sigma = params['sigma']
    T = params['T']
    N = params['N']
    
    dt = T / N
    df = np.exp(-r * dt)
    
    np.random.seed(partition_id + int(time.time()))
    
    brownian = np.random.standard_normal((num_paths, N))
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    
    S = np.zeros((num_paths, N + 1))
    S[:, 0] = S0
    
    increments = np.exp(drift + diffusion * brownian)
    S[:, 1:] = S0 * np.cumprod(increments, axis=1)

    cash_flows = np.maximum(K_strike - S[:, -1], 0)

    for t in range(N - 1, 0, -1):
        S_t = S[:, t]
        itm_mask = S_t < K_strike
        
        if np.sum(itm_mask) == 0:
            cash_flows = cash_flows * df
            continue

        Y = cash_flows[itm_mask] * df 
        X_itm = S_t[itm_mask]
        
        coeffs = np.polyfit(X_itm, Y, 2)
        continuation_value = np.polyval(coeffs, X_itm)
        
        exercise_value = K_strike - X_itm
        
        cash_flows = cash_flows * df
        
        early_exercise_indices = exercise_value > continuation_value
        
        full_itm_indices = np.where(itm_mask)[0]
        exercising_subset_indices = full_itm_indices[early_exercise_indices]
        
        cash_flows[exercising_subset_indices] = exercise_value[early_exercise_indices]

    final_values = cash_flows * df
    
    return np.mean(final_values)

def main():
    spark = SparkSession.builder \
        .appName("DistributedLSMC_Benchmarking") \
        .getOrCreate()
    
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    params = {
        's0': 100.0,
        'k': 200.0,        
        'r': 0.05,
        'sigma': 0.30,
        'T': 10.0/365.0,   
        'N': 50,           
        'M': 1000000       
    }
    
    num_workers = 4
    total_paths = params['M']
    paths_per_worker = total_paths // num_workers
    
    print(f"--- STARTING PYSPARK LSMC ---")
    print(f"Total Paths: {total_paths}")
    print(f"Workers (Partitions): {num_workers}")
    print(f"Paths per Worker: {paths_per_worker}")

    start_time = time.time()

    rdd = sc.parallelize(range(num_workers), num_workers)
    results_rdd = rdd.map(lambda x: worker_lsmc_task(x, paths_per_worker, params))
    final_price = results_rdd.mean()

    end_time = time.time()
    duration_ms = (end_time - start_time) * 1000

    print("\n--- PYSPARK RESULTS ---")
    print(f"Final Price (V0): ${final_price:.6f}")
    print(f"Computation Time: {duration_ms:.2f} ms")
    print("-----------------------")

    spark.stop()

if __name__ == "__main__":
    main()