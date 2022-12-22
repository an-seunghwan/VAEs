#%%
"""
Reference:
[1] https://zhiyzuo.github.io/VI/#python-implementation
[2] https://brunomaga.github.io/Variational-Inference-GMM
"""
#%%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
#%%
np.random.seed(1)
num_components = 3
mu_arr = np.random.choice(np.arange(-10, 10, 2), num_components) + np.random.random(num_components) * 0.1
mu_arr
#%%
n = 1000
x = []
for mu in mu_arr:
    x.append(np.random.normal(loc=mu, scale=1, size=n))
x = np.concatenate(x, axis=0)
x = x[:, None]
#%%
fig, ax = plt.subplots(figsize=(15, 4))
sns.distplot(x[:n, :], ax=ax, rug=True)
sns.distplot(x[n:n*2, :], ax=ax, rug=True)
sns.distplot(x[n*2:, :], ax=ax, rug=True)
plt.tight_layout()
plt.savefig('./assets/raw_data_hist.png', bbox_inches='tight')
plt.show()
plt.close()
#%%
"""Initialize"""
sigma2 = 1
K = 3
m = np.random.normal(size=(K, 1), scale=3)
s2 = np.ones((K, 1))
phi = np.random.uniform(size=(n, K))
phi = phi / phi.sum(axis=1, keepdims=True)
#%%
max_iter = 100
ELBO_old = 0
ELBO_list = []
for i in range(max_iter):
    
    """Variational update for cluster assignment"""
    phi = np.exp((-0.5 * (m ** 2 + s2)).T + x * m.T)
    phi = phi / phi.sum(axis=1, keepdims=True)
    
    """Variational update for cluster mean and variance"""
    m = (phi * x).sum(axis=0)
    m = (m / (phi.sum(axis=0) + 1 / sigma2))[:, None]
    s2 = (1 / (phi.sum(axis=0) + 1 / sigma2))[:, None]
    
    """ELBO"""
    # decoder
    term1 = -0.5 * np.log(2 * np.pi) - 0.5 * (m ** 2 + s2)
    term1 = (phi * (term1.T + (x * m.T) - 0.5 * (x ** 2))).sum()

    # categorical prior
    term2 = -n * np.log(K)

    # component prior
    term3 = (-0.5 * np.log(2 * np.pi) - 0.5 * np.log(sigma2) - 0.5 * (m ** 2 + s2) / sigma2).sum()

    # categorical cross entropy
    term4 = (phi * np.log(phi)).sum()

    # component cross entropy
    term5 = (-0.5 * np.log(2 * np.pi) - 0.5 * np.log(s2) - 0.5).sum()

    ELBO = term1 + term2 + term3 - term4 - term5
    if np.abs(ELBO - ELBO_old) < 1e-6:
        print("Iteration: {}".format(i), m[:, 0])
        print("ELBO converged at {:.2f} with iteration {}".format(ELBO, i))
        break
    ELBO_old = ELBO
    print("Iteration: {}".format(i), m[:, 0])
    ELBO_list.append(ELBO)
#%%
fig, ax = plt.subplots(figsize=(15, 4))
sns.distplot(x[:n], ax=ax, hist=True, norm_hist=True)
sns.distplot(np.random.normal(m[0, :], 1, n), color='k', hist=False, kde=True)
sns.distplot(x[n:n*2], ax=ax, hist=True, norm_hist=True)
sns.distplot(np.random.normal(m[1, :], 1, n), color='k', hist=False, kde=True)
sns.distplot(x[n*2:], ax=ax, hist=True, norm_hist=True)
sns.distplot(np.random.normal(m[2, :], 1, n), color='k', hist=False, kde=True)
plt.tight_layout()
plt.savefig('./assets/vgmm_est_and_hist.png', bbox_inches='tight')
plt.show()
plt.close()
#%%
fig, ax = plt.subplots(figsize=(15, 4))
sns.distplot(np.random.normal(m[0, :], 1, n), color='k', hist=False, kde=True)
sns.distplot(np.random.normal(m[1, :], 1, n), color='k', hist=False, kde=True)
sns.distplot(np.random.normal(m[2, :], 1, n), color='k', hist=False, kde=True)
plt.tight_layout()
plt.savefig('./assets/vgmm_est.png', bbox_inches='tight')
plt.show()
plt.close()
#%%
plt.figure(figsize=(6, 4))
plt.plot(np.arange(len(ELBO_list)), ELBO_list)
plt.xlabel("Iteration", fontsize=13)
plt.ylabel("ELBO", fontsize=13)
plt.tight_layout()
plt.savefig('./assets/ELBO.png', bbox_inches='tight')
plt.show()
plt.close()
#%%