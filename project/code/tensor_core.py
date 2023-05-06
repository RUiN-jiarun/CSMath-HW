from typing import Iterable, Union
import numpy as np


def kron(a, b):
    """Kronecker product of two tensors.
    """
    s1, s2 = np.shape(a)
    s3, s4 = np.shape(b)
    a = np.reshape(a, (s1, 1, s2, 1))
    b = np.reshape(b, (1, s3, 1, s4))
    return np.reshape(a * b, (s1 * s3, s2 * s4))


def kr(a, b):
    """Column-wise Khtri-Rao Product

    Version 2, one line code with vectorization and np.kron.
    Which is the Kronecker product of every column of A and B.
    """
    # Ensure same column numbers 
    if a.shape[1] != b.shape[1]:
        raise ValueError('All the tensors must have the same number of columns')
    c = np.vstack([np.kron(a[:, k], b[:, k]) for k in range(a.shape[1])]).T
    return c


def seq_kr(matrices: list, exclude: Union[int, list, None] = None, reverse=False):
    """Do Khatri-Rao Product in a sequence of matrices.

    Args:
        matrices (list): Matrices to compute. [A1, A2, A3, ...]
        exclude (Union[int, list, None], optional): Index of matrix which to ignore in compute. 
        Starts from 0. Defaults to None.
        reverse (bool, optional): If True, the order of the matrices is reversed. Defaults to False.

    Returns:
        Tensor: Matrix as a result of computation.
    """  
    # Generate index except exclude matrix
    idx = list(range(len(matrices)))
    if isinstance(exclude, int):
        idx.pop(exclude)
    if isinstance(exclude, Iterable):
        for i in exclude:
            idx.remove(i)

    if reverse:
        idx = idx[::-1]

    res = matrices[idx[0]]
    for i in range(1, len(idx)):
        res = kr(res, matrices[idx[i]])
    return res


def seq_kron(matrices, exclude: Union[int, list, None] = None, reverse=False):
    """Do Kronecker Product in a sequence of matrices.

    Args:
        matrices (list): Matrices to compute. [A1, A2, A3, ...]
        exclude (Union[int, list, None], optional): Index of matrix which to ignore in compute. Starts from 0. Defaults to None.
        reverse (bool, optional): If True, the order of the matrices is reversed. Defaults to False.

    Returns:
        Tensor: Matrix as a result of computation.
    """
    idx = list(range(len(matrices)))
    if isinstance(exclude, int):
        idx.pop(exclude)
    if isinstance(exclude, Iterable):
        for i in exclude:
            idx.remove(i)

    if reverse:
        idx = idx[::-1]
        
    res = matrices[idx[0]]
    for i in range(1, len(idx)):
        res = kron(res, matrices[idx[i]])
    return res

def unfold(tensor, mode):
    """Returns the mode-`n` unfolding of `tensor` with modes starting at `0`.
    
    args:
        tensor : ndarray
        mode : int, default is 0,indexing starts at 0, therefore mode is in ``range(0, tensor.ndim)``
    
    return:
        ndarray: unfolded_tensor of shape ``(tensor.shape[mode], -1)``
    """
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))


def fold(tensor, mode, shape):
    """Given a mode-`mode` unfolding tensor, fold to specific shape

    Args:
        tensor ([type]): [description]
        mode (int): [description]
        shape ([type]): [description]

    Returns:
        [type]: [description]
    """
    shape = list(shape)
    shape.insert(0, shape.pop(mode))
    return np.moveaxis(np.reshape(tensor, shape), 0, mode)

def norm(tensor, ord=2, axis=None):
    """Norm of `tensor`. 

    This function is able to return one of an given `ord` norm of tensor, depending on the value of the `ord`.

    Args:
        tensor ([type]): Input tensor.
        ord (int, optional): Order of the norm. Support `'inf'` or `int`. Defaults to 2.
        axis (int | None, optional): It specifies the axis of `tensor` along to compute the vector norms. 
        If `axis` is `None`, then compute norm of flatten `tensor`. Defaults to None.

    Returns:
        float | ndarray: Norm of the tensor or vectors.
    """
    if ord == 'inf':
        return np.abs(tensor).max(axis=axis)
    if ord == 1:
        return np.abs(tensor).sum(axis=axis)
    else:
        return np.power(np.abs(tensor), ord).sum(axis=axis) ** (1 / ord)
    
def mttkrp(tensor, factors, mode):
    return unfold(tensor, mode) @ seq_kr(factors, exclude=mode, reverse=False)