import torch 
from typing import Union, Callable, List, Tuple

def jacobian_approx(p, f):

    try:
        jac = torch.autograd.functional.jacobian(f, p, vectorize=True) # create_graph=True
    except RuntimeError:
        jac = torch.autograd.functional.jacobian(f, p, strict=True, vectorize=False)

    return jac

def LM(
        p: torch.Tensor,
        function: Callable, 
        args: Union[Tuple, List] = (), 
        ftol: float = 1e-8,
        ptol: float = 1e-8,
        gtol: float = 1e-8,
        tau: float = 1e-3,
        rho1: float = .25, 
        rho2: float = .75, 
        bet: float = 2,
        gam: float = 3,
        max_iter: int = 50,
    ) -> List[torch.Tensor]:
    """
    Levenberg-Marquardt implementation for least-squares fitting of non-linear functions
    
    :param p: initial value(s)
    :param function: user-provided function which takes p (and additional arguments) as input
    :param jac_fun: user-provided Jacobian function which takes p (and additional arguments) as input
    :param args: optional arguments passed to function
    :param ftol: relative change in cost function as stop condition
    :param ptol: relative change in independant variables as stop condition
    :param gtol: maximum gradient tolerance as stop condition
    :param tau: factor to initialize damping parameter
    :param rho1: first gain factor threshold for damping parameter adjustment for Marquardt
    :param rho2: second gain factor threshold for damping parameter adjustment for Marquardt
    :param bet: multiplier for damping parameter adjustment for Marquardt
    :param gam: divisor for damping parameter adjustment for Marquardt
    :param max_iter: maximum number of iterations
    :return: list of results
    """

    if len(args) > 0:
        # pass optional arguments to function
        fun = lambda p: function(p, *args)
    else:
        fun = function

    # use numerical Jacobian if analytical is not provided
    jac_fun = lambda p: jacobian_approx(p, f=fun)

    f = fun(p)
    j = jac_fun(p)
    g = torch.matmul(j.T, f)
    H = torch.matmul(j.T, j)
    u = tau * torch.max(torch.diag(H))
    v = 2
    p_list = [p]
    cnt = 0
    while cnt < max_iter:
        D = torch.eye(j.shape[1], device=j.device)
        D *= torch.max(torch.maximum(H.diagonal(), D.diagonal()))
        h = -torch.linalg.lstsq(H + u * D, g, rcond=None, driver=None)[0]   # least square
        f_h = fun(p + h)
        rho_denom = torch.matmul(h, u * h - g)
        rho_nom = torch.matmul(f, f) - torch.matmul(f_h, f_h)

        if rho_denom > 0:
            rho = rho_nom / rho_denom  
        elif rho_nom > 0:
            rho = torch.inf
        else:
            rho = -torch.inf

        if rho > 0:
            p = p + h
            j = jac_fun(p)
            g = torch.matmul(j.T, fun(p))
            H = torch.matmul(j.T, j)
        p_list.append(p.clone())
        f_prev = f.clone()
        f = fun(p)

        if rho < rho1:
            u *= bet 
        elif rho > rho2: 
            u /= gam 
        else: 
            u = u

        # stop conditions
        gcon = max(abs(g)) < gtol
        pcon = (h**2).sum()**.5 < ptol*(ptol + (p**2).sum()**.5)
        fcon = ((f_prev-f)**2).sum() < ((ftol*f)**2).sum() if rho > 0 else False
        if gcon or pcon or fcon:
            break
        cnt += 1

    return p_list
