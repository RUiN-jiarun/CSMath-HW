import torch 
from typing import Union, Callable, List, Tuple

def jacobian_approx(p, f):

    try:
        J = torch.autograd.functional.jacobian(f, p, vectorize=True) # create_graph=True
    except RuntimeError:
        J = torch.autograd.functional.jacobian(f, p, strict=True, vectorize=False)

    return J

def LM( p: torch.Tensor,
        function: Callable, 
        data: Union[Tuple, List] = (), 
        ftol: float = 1e-8,
        ptol: float = 1e-8,
        gtol: float = 1e-8,
        tau: float = 1e-3,
        rho1: float = 0.25, 
        rho2: float = 0.75, 
        bet: float = 2,
        gam: float = 3,
        max_iter: int = 50,
    ) -> List[torch.Tensor]:
    """
    Levenberg-Marquardt implementation for least-squares fitting of non-linear functions
    
    :param p: initial value(s)
    :param function: user-provided function which takes p (and additional arguments) as input
    :param data: optional arguments passed to function
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


    fun = lambda p: function(p, *data)

    # use numerical Jacobian if analytical is not provided
    jac_fun = lambda p: jacobian_approx(p, f=fun)

    f = fun(p)
    J = jac_fun(p)
    g = torch.matmul(J.T, f)
    H = torch.matmul(J.T, J)
    mu = tau * torch.max(torch.diag(H))
    v = 2

    p_list = [p]
    cnt = 0
    while cnt < max_iter:
        I = torch.eye(J.shape[1], device=J.device)
        I *= torch.max(torch.maximum(H.diagonal(), I.diagonal()))
        h = -torch.linalg.lstsq(H + mu * I, g, rcond=None, driver=None)[0]   # least square
        f_h = fun(p + h)
        rho_denom = torch.matmul(h, mu * h - g)
        rho_nom = torch.matmul(f, f) - torch.matmul(f_h, f_h)

        if rho_denom > 0:
            rho = rho_nom / rho_denom  
        elif rho_nom > 0:
            rho = torch.inf
        else:
            rho = -torch.inf

        if rho > 0:
            p = p + h
            J = jac_fun(p)
            g = torch.matmul(J.T, fun(p))
            H = torch.matmul(J.T, J)
        p_list.append(p.clone())
        f_prev = f.clone()
        f = fun(p)

        if rho < rho1:
            mu *= bet 
        elif rho > rho2: 
            mu /= gam 

        # stop iteration
        gcon = max(abs(g)) < gtol
        pcon = (h**2).sum()**0.5 < ptol*(ptol + (p**2).sum()**0.5)
        fcon = ((f_prev-f)**2).sum() < ((ftol*f)**2).sum() if rho > 0 else False
        if gcon or pcon or fcon:
            break
        cnt += 1

    return p_list
