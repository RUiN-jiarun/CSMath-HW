import tensorly as tl
import tensorly.decomposition
import timeit
import tqdm
import cp
import numpy as np

def td_time_benchmark(f, initializer, order=3, max_iter=5, max_time=30, 
                 verbose=1, *args, **kw) -> float:
    """Do speed benchmark on given function. Return average time cost.

    Args:
        f (function): A python function to run with, receive tensor as parameter.
        initializer (function | class): Receive numpy array as raw data. Return specific type.
        max_iter (int, optional): Max iteration to do with function. Defaults to 5.
        max_time (int, optional): Max time to run with function. Defaults to 30.
        verbose (int, optional): How much information to log. Defaults to 0.
        *args, **kw: Parameters of f.
    """
    timer = timeit.default_timer
    used_time = 0

    if verbose > 1:
        pbar = tqdm.tqdm(total=max_iter)

    for i in range(max_iter):
        data_shape = list(range(2, order + 2))
        data = np.random.randn(*data_shape)
        tensor = initializer(data)
        
        tick = timer()
        f(tensor, *args, **kw)
        tock = timer()
        used_time += tock - tick

        if verbose > 1:
            pbar.update(1)

        if used_time > max_time:
            if verbose > 0:
                print(f'Iterations: [{i+1}] | Total: [{used_time:.5f}s] | Avg.: [{used_time/(i+1):.5f}s]')
            break
        
        if i == max_iter - 1:
            if verbose > 0:
                print(f'Iterations: [{i+1}] | Total: [{used_time:.5f}s] | Avg.: [{used_time/(i+1):.5f}s]')

def test_speed_3ord():
    """Decomposition on 3 order tensors.
    """
    print('\nSpeed benchmark for 3 order tensor CP decomposition.\n')

    print('----------TensorNP----------')
    td_time_benchmark(cp.cp, np.array, max_iter=30, verbose=1, r=3, 
                 stop_iter=500, tol=1e-5, normalize_factor=True)
    print('----------Tensorly----------')
    td_time_benchmark(tensorly.decomposition.parafac, tl.tensor, max_iter=30, rank=3,
                 init='random', n_iter_max=500, tol=1e-5, normalize_factors=True)  

def valid_3ord():
    print('\nCorrectness benchmark for 3 order tensor CP decomposition.\n')
    
    shape = (2, 3, 4)
    max_iter = 30
    print('----------TensorNP----------')
    norm_errors = 0
    for _ in range(max_iter):
        tensor = np.random.randn(2, 3, 4)
        factors, lamda = cp.cp(tensor, r=3, stop_iter=500, tol=1e-5, normalize_factor=True)
        rec_tensor = cp.reconstruct_cp(factors, lamda, shape)
        norm_error = np.linalg.norm(rec_tensor - tensor) / np.linalg.norm(tensor)
        norm_errors += norm_error
    print(f'error ({norm_errors/max_iter})')

    print('----------Tensorly----------')
    norm_errors = 0
    for _ in range(max_iter):
        tensor = np.random.randn(2, 3, 4)
        tl_tensor = tl.tensor(tensor)
        cp_tensor = tensorly.decomposition.parafac(
            tl_tensor, rank=3, n_iter_max=500, tol=1e-6, normalize_factors=True, init='random')
        rec_tensor = cp.reconstruct_cp(cp_tensor.factors, cp_tensor.weights, shape)
        norm_error = np.linalg.norm(rec_tensor - tensor) / np.linalg.norm(tensor)
        norm_errors += norm_error
    print(f'error ({norm_errors/max_iter})')