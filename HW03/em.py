import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

class EM(object):
    def __init__(self, data, ndim, p_sample, init_mu, init_sigma, max_iter=10, eps=.1):
        self.data = data
        self.ndim = ndim
        self.p_sample = p_sample
        self.mu = init_mu
        self.sigma = init_sigma
        self.max_iter = max_iter
        self.eps = eps
    
    def E(self):
        n_samples = self.data.shape[0]
        p_estimate = np.zeros(shape=(self.ndim, n_samples))
        p_estimate_sum = np.zeros(shape=(1, n_samples))
        for j in range(n_samples):
            for i in range(self.ndim):
                p_gaussian = 1 / np.sqrt(((2 * np.pi) ** 2) * np.linalg.det(self.sigma[i])) * np.exp(-0.5 * np.dot(np.dot((self.data[j] - self.mu[i]), np.linalg.inv(self.sigma[i])), (self.data[j] - self.mu[i]).T))
                p_estimate[i][j] = self.p_sample[i] * p_gaussian
                p_estimate_sum[:,j] += p_estimate[i][j]
        p_estimate = p_estimate / p_estimate_sum
        
        return p_estimate
    
    def M(self, p_estimate):
        n_samples = self.data.shape[0]
        ndim = p_estimate.shape[0]

        mu_estimate_sum = [0 for i in range(ndim)]
        mu_estimate = [0 for i in range(ndim)]
        sigma_estimate_sum = [0 for i in range(ndim)]
        sigma_estimate = [0 for i in range(ndim)]
        n_estimate = np.sum(p_estimate, axis=1)
        p_sample_estimate = n_estimate / n_samples # Mixture density we obtain from EM
        for i in range(ndim):
            for j in range(n_samples):
                mu_estimate_sum[i] += p_estimate[i][j] * self.data[j]
            mu_estimate[i] = mu_estimate_sum[i] / n_estimate[i] # Mu we obtain from EM
        for i in range(ndim):
            sigma_estimate_sum[i] = np.zeros(shape=(2,2))
            for j in range(n_samples):
                sigma_estimate_sum[i] += p_estimate[i][j] * np.dot((self.data[j] - mu_estimate[i]).reshape(-1,1), (self.data[j] - mu_estimate[i]).reshape(1,-1))
            sigma_estimate[i] = sigma_estimate_sum[i] / n_estimate[i]

        return p_sample_estimate, mu_estimate, sigma_estimate
    
    def EM(self):
        
        color = ['red', 'green', 'blue', 'cyan', 'yellow', 'gray', ]
        n_samples = self.data.shape[0]
        p_estimate = np.zeros(shape=(self.ndim, n_samples))

        for i in range(self.max_iter):
            ells = [patches.Ellipse(
                xy = self.mu[j], 
                width = 6 * np.sqrt(self.sigma[j][0][0]), 
                height = 6 * np.sqrt(self.sigma[j][1][1])
            ) for j in range(self.ndim)]
            fig = plt.figure(1, figsize=(18,10))
            ax = fig.add_subplot(1,1,1)
            ax.set_title('Iteration: {}'.format(i+1))
            for j, e in enumerate(ells):
                ax.add_artist(e)
                e.set_clip_box(ax.bbox)
                e.set_alpha(1)
                e.set_facecolor('none')
                e.set_edgecolor(color[j])
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            plt.scatter(self.data[:,0], self.data[:,1], color='black', s=1)
            
            # plt.savefig('EM_iterations_ndim4.png')
            
            p_estimate = self.E()
            old_mu = self.mu
            p_sample, self.mu, self.sigma = self.M(p_estimate=p_estimate)

            loss = np.sum(np.sum(np.abs(np.array(old_mu) - np.array(self.mu))))
            print("Iterations: ", i+1, ", Loss: ", loss)

            if loss < self.eps:
                plt.savefig('img/MOG_Using_EM_ndim4.png')
                plt.show()
                break
            else:
                plt.show()
        
        return
