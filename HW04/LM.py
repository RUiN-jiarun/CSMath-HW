import numpy as np
from matplotlib import pyplot as plt
import torch


class LM():
    def __init__(self,device='cpu'):
        self.device = device

    def solve(self,f=None, x0=None, y=torch.FloatTensor([0]), max_iter=100, tol=1e-6, lambda0=1e-3, delta=1, scale_lambda=False):
        """
        Minimize a function `f` using the modified Levenberg-Marquardt algorithm.
        Parameters
        ----------
        f : callable
            The function to minimize. Should take a tensor `x` as input and return a scalar tensor.
        J : callable
            The Jacobian of the function `f`. Should take a tensor `x` as input and return a tensor with shape (m, n),
            where `m` is the number of outputs of `f` and `n` is the number of inputs.
        x0 : torch.Tensor
            The initial guess for the minimum.
        max_iter : int, optional
            The maximum number of iterations. Default is 100.
        tol : float, optional
            The tolerance for the change in the loss. Default is 1e-6.
        lambda0 : float, optional
            The initial value of the damping parameter. Default is 1.0.
        delta : float, optional
            The step size used to compute the approximate Hessian. Default is 1e-6.
        scale_lambda: boolean, optional
            if true, makes lambda scaled according to JtJ matrix
        Returns
        -------
        x_min : torch.Tensor
            The estimated minimum.
        """

        x = x0.clone().requires_grad_(False).to(self.device)
        y = y.clone().requires_grad_(False).to(self.device)

        J = torch.autograd.functional.jacobian

        # initialization
        f_val = f(x, y).reshape(-1, 1).repeat((x.shape[0],1,1))
        J_val = J(f, (x,y))[0].unsqueeze(1)
        JtJ = torch.matmul(J_val.permute(0,2,1), J_val)
        Jtf = torch.matmul(J_val.permute(0,2,1), f_val)
        multiplier = 2
        x_list = []
        max_lambda = torch.FloatTensor([1e5]).to(self.device)
        min_lambda = torch.FloatTensor([1e-5]).to(self.device)
        eye_ = torch.eye(JtJ.shape[1]).to(self.device)
        if scale_lambda:
            lambda_ = torch.FloatTensor([lambda0]) * torch.max((JtJ*eye_)).to(self.device)
        else:
            lambda_ = torch.FloatTensor([lambda0]).to(self.device)

        for i in range(max_iter):
            if scale_lambda:
                eye_J = eye_ * torch.max((JtJ*eye_))
            else:
                eye_J = eye_
            h = -1*torch.linalg.lstsq(JtJ + lambda_ * eye_J,Jtf)[0]
            # h = (JtJ + lambda_ * eye_J).inverse().matmul(Jtf)
            dparam = (delta * h)
            # x_new = torch.min(torch.max(x + dparam.squeeze(), min_bound),max_bound)
            x_new = x + dparam.squeeze()
            f_new_val = f(x_new,y).reshape(-1, 1)
            rho = (torch.norm(f_val) - torch.norm(f_new_val.repeat((x.shape[0],1,1))) ) / torch.matmul(h.permute(0,2,1), lambda_ * h - Jtf).mean() \
                if torch.matmul(dparam.permute(0,2,1), lambda_ * dparam - Jtf).mean() > 0 else torch.inf \
                if (torch.norm(f_val) - torch.norm(f_new_val.repeat((x.shape[0],1,1))) )  > 0 else -torch.inf

            if rho > 1e-4:
                x = x_new
                J_val = J(f, (x,y))[0].unsqueeze(1)
                JtJ = torch.matmul(J_val.permute(0, 2, 1), J_val)
                Jtf = torch.matmul(J_val.permute(0, 2, 1), f_val)
                lambda_= lambda_ * torch.max(torch.tensor([1 / 3, 1 - (2 * rho - 1) ** 3]))
                multiplier = 2
                if torch.norm(f_val - f_new_val) < tol:
                    print("tol is satisfied")
                    break
            else:
                lambda_= (torch.min(lambda_ * multiplier, max_lambda))
                multiplier *= 2

            # x_list.append(x.clone())
            f_val = f(x, y).reshape(-1, 1).repeat((x.shape[0],1,1))
            if i == max_iter - 1:
                print("Warning: Maximum number of iterations reached.")

        return x.detach().cpu(), f_val

def fun(coeff):
    t = torch.linspace(0, 100, 100)
    # A*e^(-Bx)
    return coeff[0].unsqueeze(1) * torch.exp(-t * coeff[1].unsqueeze(1))


def model(x, y=None):
    return torch.mean((x - y) ** 2)

coeff = torch.rand(2, 2) * 0.1
y = fun(coeff)
x = torch.randn((2, 100))
x_new, f_val = LM(device='cuda').solve(f=model, x0=x, y=y, max_iter=100, delta=1)

print(y)

print('x* = ', x_new)
print('f(x*) = ', f_val.cpu())

plt.plot(y[0], 'b-')
plt.scatter(np.arange(100), x_new[0].numpy().T)
# plt.show()

plt.plot(y[1])
plt.scatter(np.arange(100), x_new[1].numpy().T)
plt.show()

