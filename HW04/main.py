import matplotlib.pyplot as plt
import torch
from leven_marq import LM

gauss_term = lambda t, mu, sigma: torch.exp(-.5 * (t-mu)**2 / sigma**2)
asymm_term = lambda t, mu, sigma, eta: 1 + torch.erf(eta * (t-mu) / (sigma * 2**.5))

erf_term = lambda t, mu, sigma, eta: torch.exp(-.5*((t-mu)**2/ sigma**2) * eta**2)
pd_asymm_wrt_mu = lambda t, mu, sigma, eta: erf_term(t, mu, sigma, eta) * -1 * (2/torch.pi)**.5 * eta/sigma
pd_asymm_wrt_sigma = lambda t, mu, sigma, eta: erf_term(t, mu, sigma, eta) * -1 * (2/torch.pi)**.5 * eta*(t-mu)/sigma**2
pd_asymm_wrt_eta = lambda t, mu, sigma, eta: erf_term(t, mu, sigma, eta) * (2/torch.pi)**.5 * (t-mu)/sigma

def emg_model(p, t: torch.Tensor = None):

    alpha, mu, sigma, eta = p

    alpha = 1 if alpha is None else alpha
    gauss = gauss_term(t, mu, sigma)
    asymm = asymm_term(t, mu, sigma, eta)

    return alpha * gauss * asymm

torch.seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

norm, mean, sigm, skew = 8, -1, 60, 7.5

gt_params = torch.tensor([norm, mean, sigm, skew], dtype=torch.float64, device=device)
init_params = torch.tensor([5.5, -0.75, 10, 3], dtype=torch.float64, device=device, requires_grad=True)
cost_fun = lambda p, t, y: (y - emg_model(p, t))**2

print('gt params: ', gt_params.detach().cpu())

t = torch.linspace(-1e2, 1e2, int(2e2)).to(device)
data = emg_model(gt_params, t)
data_raw = data + 0.001 * torch.randn(len(data), dtype=torch.float64, device=device)

coeffs = LM(p=init_params, function=cost_fun, args=(t, data_raw))

ret_params = torch.allclose(coeffs[-1], gt_params, atol=1e-1)
print('total iters: ', len(coeffs))
print('estimate params: ', coeffs[-1].detach().cpu())
eps_list = []
for i in range(len(coeffs)):
    eps_list.append(torch.sum(cost_fun(coeffs[i], t=t, y=data_raw)).item())
print('eps = ', eps_list[-1])

plt.plot(eps_list)
plt.title('Loss curve')
plt.ylabel('epsilon')
plt.xlabel('iters') 
plt.show()

dot = plt.scatter(t.cpu(), data_raw.cpu(), s=0.2)
l1, = plt.plot(data.detach().cpu(), color='blue')
l2, = plt.plot(emg_model(coeffs[-1], t).detach().cpu(), 'r:')
plt.legend(handles=[dot, l1, l2], labels=['data_raw','gt_function', 'estimate_function'], loc='upper left')
plt.show()



