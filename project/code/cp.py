import numpy as np
from tensor_core import norm, mttkrp, fold, seq_kr
import logging


def reconstruct_cp(factors, lamda, shape):
    """Reconstruct tensor using CP format.

    Args:
        factors (ndarray): Decomposition factor matrices.
        lamda (ndarray): Normalization weights.
        shape (tuple): Shape of raw tensor

    Returns:
        ndarray: tensor.
    """
    # Equal to factors[0] @ diag(lamda)
    factors[0] = factors[0] * lamda
    rec_tensor = fold(np.dot(factors[0], seq_kr(factors, exclude=0, reverse=False).T), 
                      0, shape)
    return rec_tensor


def rec_error_calc(tensor, lamda, norm_tensor, factors, mttkrp=None):
    """Reconstruction error calculate.

    This function reconstruct tensor using factors and normalization lambda. And calculate 
    the norm-2 distance between raw and reconstruct tensor.

    Args:
        tensor ([type]): [description]
        lamda ([type]): [description]
        norm_tensor ([type]): [description]
        factors ([type]): [description]
        mttkrp ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    if mttkrp is not None:
        # ||tensor - rec||^2 = ||tensor||^2 + ||rec||^2 - 2*<tensor, rec>
        coef = np.outer(lamda, lamda)
        for i in range(len(factors)):
            coef = coef * np.dot(factors[i].T, factors[i])
        factors_norm = np.sqrt(coef.sum())
        
        # mttkrp and factor for the last mode. This is equivalent to the
        # inner product <tensor, factorization>
        iprod = np.sum(np.sum(mttkrp * factors[-1], axis=0))
        unnorml_rec_error = np.sqrt(np.abs(norm_tensor**2 + factors_norm**2 - 2 * iprod))
    else:
        rec_tensor =  reconstruct_cp(factors, lamda, tensor.shape)
        # The unnormalized reconstruction error.
        unnorml_rec_error =  norm(tensor - rec_tensor, 2)
    return unnorml_rec_error


def cp(tensor: np.ndarray, r: int, stop_iter=0, init='random', normalize_factor=True, 
       cvg_criterion='abs_rec_error', tol=1e-8, random_seed=None, verbose=0, 
       return_errors=False):
    """CP-ALS decomposition

    args:
        tensor (np.ndarray): tensor to decompose
        r (int): Number of components
        iteration (int): Number of iteration performs
    """
    if init != 'random' and init != 'svd':
        raise('Not a supported initialization method!')
    if random_seed: 
        np.seed(seed=random_seed)
        
    norm_tensor = norm(tensor, 2)
    rec_errors = []
    order = tensor.ndim
    factors = [None] * order
    lamda = np.ones(order).reshape(1, -1)

    # Initialize factor matrix 
    for i in range(order):
        factors[i] = np.random.rand(tensor.shape[i], r)   
    
    # ALS optimization
    for iteration in range(stop_iter):
        for i in range(order):
            v = np.ones((r, r))
            for i1 in range(order):
                if i1 == i:
                    continue
                v = v * (factors[i1].T @ factors[i1])
            
            # mttkrp: see below for details.
            # https://github.com/andrewssobral/tensor_toolbox/blob/master/%40ttensor/mttkrp.m
            # http://tensor-compiler.org/docs/data_analytics/index.html
            # mttkrp = unfold(tensor, i) @ seq_kr(factors, exclude=i, reverse=True)
            mtt = mttkrp(tensor, factors, i)
            factors[i] = mtt @ np.linalg.pinv(v)
            if normalize_factor:
                scales = norm(factors[i], 2, axis=0)
                lamda = np.where(scales==0, np.ones(np.shape(scales)), scales)
                factors[i] = factors[i] / lamda
        unnorml_rec_error = rec_error_calc(tensor, lamda, norm_tensor, factors, 
                                           mtt)
        rec_error = unnorml_rec_error / norm_tensor
        rec_errors.append(rec_error)
        if tol:
            if iteration >= 1:
                rec_error_decrease = rec_errors[-2] - rec_errors[-1]
                if verbose > 1:
                    print("iteration [{}] error: {:.5f} | decrease = {:7.2e}".
                          format(iteration, rec_error, rec_error_decrease))
                if cvg_criterion == 'abs_rec_error':
                    stop_flag = abs(rec_error_decrease) < tol
                elif cvg_criterion == 'rec_error':
                    stop_flag = rec_error_decrease < tol
                if stop_flag:
                    if verbose:
                        print("CP-ALS converged after {} iterations".format(iteration))
                    break 
            else:
                if verbose > 1:
                    print("iteration [{}] reconstruction error: {}".
                          format(iteration, rec_errors[-1]))

    if return_errors:
        return factors, lamda, rec_errors
    else:
        return factors, lamda

