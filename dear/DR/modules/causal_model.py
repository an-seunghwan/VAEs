#%%
"""
Reference:
[1]: https://github.com/xwshen51/DEAR/blob/main/causal_model.py
"""
#%%
import torch
from torch import nn
import numpy as np
#%%
class InvertiblePriorLinear(nn.Module):
    """Invertible Prior for Linear case

    Parameter:
        p: mean and std parameter for scaling
    """
    def __init__(self):
        super(InvertiblePriorLinear, self).__init__()
        self.p = nn.Parameter(torch.rand([2]))

    def forward(self, eps):
        o = self.p[0] * eps + self.p[1]
        return o
    
    def inverse(self, o):
        eps = (o - self.p[1]) / self.p[0]
        return eps
#%%
# torch.manual_seed(0)
# eps_in = torch.randn(10, 1)
# ipl = InvertiblePriorLinear()
# eps_out = ipl(eps_in)
# eps_in_ = ipl.inverse(eps_out)
# assert (eps_in - eps_in_).abs().sum() < 1e-6
#%%
class InvertiblePWL(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self, vmin=-5, vmax=5, n=100, use_bias=True):
        super(InvertiblePWL, self).__init__()
        
        self.int_length = (vmax - vmin) / (n - 1)
        self.vmin = vmin
        self.vmax = vmax
        self.n = n
        if use_bias:
            self.b = nn.Parameter(torch.randn([1]) + vmin)
        else:
            self.b = vmin
        self.points = nn.Parameter(torch.from_numpy(
            np.linspace(vmin, vmax, n).astype('float32')).view(1, n),
            requires_grad = False)
        self.p = nn.Parameter(torch.randn([n + 1]) / 5)
        
    def to_positive(self, x):
        return torch.exp(x) + 1e-3

    def forward(self, eps):
        # bias term
        delta_h = self.int_length * self.to_positive(self.p[1:self.n]).detach()
        delta_bias = torch.zeros([self.n]).to(eps.device)
        delta_bias[0] = self.b
        for i in range(self.n-1):
            delta_bias[i+1] = delta_bias[i] + delta_h[i]
            
        index = torch.sum(((eps - self.points) >= 0).long(), 1).detach() 
        start_points = index - 1 # where indicator becomes 1 (index)
        start_points[start_points < 0] = 0 # smaller than boundary, set to zero 
        delta_bias = delta_bias[start_points]
        
        start_points = torch.squeeze(self.points)[torch.squeeze(start_points)].detach() # points where indicator becomes 1
        
        # weight term
        w = self.to_positive(self.p[index])
        
        out = (eps - start_points.view(-1,1)) * w.view(-1,1) + delta_bias.view(-1,1)
        return out

    def inverse(self, out):
        # bias term
        delta_h = self.int_length * self.to_positive(self.p[1:self.n]).detach()
        delta_bias = torch.zeros([self.n]).to(out.device)
        delta_bias[0] = self.b
        for i in range(self.n-1):
            delta_bias[i+1] = delta_bias[i] + delta_h[i]
            
        index = torch.sum(((out - delta_bias) >= 0).long(), 1).detach() 
        start_points = index - 1
        start_points[start_points < 0] = 0
        delta_bias = delta_bias[start_points]
        
        start_points = torch.squeeze(self.points)[torch.squeeze(start_points)].detach()
        
        # weight term
        w = self.to_positive(self.p[index])
        
        eps = (out - delta_bias.view(-1,1)) / w.view(-1,1) + start_points.view(-1,1)
        return eps
#%%
# torch.manual_seed(0)
# eps_in = torch.randn(10, 1)
# ipl = InvertiblePWL()
# eps_out = ipl(eps_in)
# eps_in_ = ipl.inverse(eps_out)
# assert (eps_in - eps_in_).abs().sum() < 1e-6
#%%
# visualization
# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(7, 4))
# torch.manual_seed(0)
# np.random.seed(0)
# ipl = InvertiblePWL(vmin=-1, vmax=1, n=5)
# ipl.p.data = torch.from_numpy(np.random.uniform(-4, 4, size=ipl.n + 1))
# eps_in = torch.from_numpy(np.linspace(ipl.vmin - 1, ipl.vmax + 1, 10000)[:, None])
# eps_out = ipl(eps_in)
# plt.plot(eps_in.detach().numpy()[:, 0],
#          eps_out.detach().numpy()[:, 0])
# for x in ipl.points.numpy()[0, :]:
#     plt.axvline(x, color='black', linestyle='--') 
#%%
class InvertiblePriorInv(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self,prior):
        super(InvertiblePriorInv, self).__init__()
        self.prior = prior
        
    def forward(self, o):
        return self.prior.inverse(o)
    
    def inverse(self, eps):
        return self.prior(eps)
#%%
class SCM(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self, d, A=None, scm_type='nlrscm'):
        super().__init__()
        self.d = d
        self.A_given = A
        self.A_fix_idx = A == 0
        self.A = nn.Parameter(torch.zeros(d, d))

        # Elementwise nonlinear mappings
        if scm_type=='linscm':
            prior_net_model = lambda : InvertiblePriorLinear()
            prior_net_enc_model = lambda x: InvertiblePriorInv(x)
        elif scm_type=='nlrscm':
            prior_net_model = lambda : InvertiblePWL()
            prior_net_enc_model = lambda x: InvertiblePriorInv(x)
        else:
            raise NotImplementedError("Not supported prior network.")

        for i in range(d):
            setattr(self, "prior_net%d" % i, prior_net_model()) # forward
            setattr(self, "enc_net%d" % i, prior_net_enc_model(getattr(self, "prior_net%d" % i))) # inverse

    def set_zero_grad(self):
        if self.A_given is None:
            pass
        else:
            for i in range(self.d):
                for j in range(self.d):
                    if self.A_fix_idx[i, j]:
                        self.A.grad.data[i, j].zero_()

    def prior_nlr(self, z):
        '''Nonlinear transformation f_2(z)'''
        zs = torch.split(z, 1, dim=1)
        z_new = []
        for i in range(self.d):
            z_new.append(getattr(self, "prior_net%d" % i)(zs[i]))
        return torch.cat(z_new, dim=1)

    def enc_nlr(self, z):
        '''f_2^{-1}(z)'''
        zs = torch.split(z, 1, dim=1)
        z_new = []
        for i in range(self.d):
            z_new.append(getattr(self, "enc_net%d" % i)(zs[i]))
        return torch.cat(z_new, dim=1)

    def mask(self, z): # Az
        z = torch.matmul(z, self.A)
        return z

    def inv_cal(self, eps): # (I-A)^{-1} @ epsilon
        adj_normalized = torch.inverse(torch.eye(self.A.shape[0], device=self.A.device) - self.A)
        z_pre = torch.matmul(eps, adj_normalized)
        return z_pre

    def get_eps(self, z):
        '''Returns epsilon from f_2^{-1}(z)'''
        return torch.matmul(z, torch.eye(self.A.shape[0], device=self.A.device) - self.A)

    def intervene(self, z, z_ori):
        # f_2^{-1}(z)
        z_ori = self.enc_nlr(z_ori)
        z = self.enc_nlr(z)
        # masked nonlinear z
        z_new = self.mask(z)
        z_new = z_new + self.get_eps(z_ori)
        return self.prior_nlr(z_new)

    def forward(self, eps=None, z=None):
        if eps is not None and z is None:
            # (I-A)^{-1} @ epsilon, [n x d]
            z = self.inv_cal(eps) 
            # nonlinear transform, [n x d]
            return self.prior_nlr(z)
        else:
            # f_2^{-1}(z), [n x d]
            z = self.enc_nlr(z)
            # mask z
            z_new = self.mask(z) # new f_2^{-1}(z) (without noise)
            return z_new, z
#%%
# d = 4
# A = torch.zeros(d, d)
# A[:2, 2:] = 1
# scm = SCM(d, A)

# eps = torch.randn(10, d)
# scm(eps)
#%%