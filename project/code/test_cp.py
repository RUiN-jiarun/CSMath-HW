import benchmark_cp

if __name__ == '__main__':
    benchmark_cp.test_speed_3ord()

    benchmark_cp.valid_3ord()

"""
Speed benchmark for 3 order tensor CP decomposition.

----------TensorNP----------
Iterations: [30] | Total: [3.08016s] | Avg.: [0.10267s]
----------Tensorly----------
Iterations: [30] | Total: [5.28228s] | Avg.: [0.17608s]

Correctness benchmark for 3 order tensor CP decomposition.

----------TensorNP----------
error (0.18030571017344435)
----------Tensorly----------
error (0.16475278160403942)
"""