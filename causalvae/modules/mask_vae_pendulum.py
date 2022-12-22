#%%
#Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#This program is free software; 
#you can redistribute it and/or modify
#it under the terms of the MIT License.
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.
#%%
"""
Reference:
[1]: https://github.com/huawei-noah/trustworthyAI/blob/master/research/CausalVAE/codebase/models/mask_vae_pendulum.py
"""
#%%
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import modules.util as ut
import modules.mask as nns

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
# device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
#%%
class CausalVAE(nn.Module):
    def __init__(self, device="cuda:0", name='vae', z_dim=16, z1_dim=4, z2_dim=4, image_size=64, inference = False, alpha=0.3, beta=1):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.image_size = image_size
        self.channel = 3
        """FIXME"""
        self.scale = np.array([[0, 1],[0, 1],[0, 1],[0, 1]])
        # self.scale = np.array([[0, 44],[100, 40],[6.5, 3.5],[10, 5]])
        
        # Small note: unfortunate name clash with torch.nn
        # nn here refers to the specific architecture file found in
        # codebase/models/nns/*.py
        self.enc = nns.Encoder(self.z_dim, self.channel, image_size=self.image_size).to(device)
        self.dec = nns.Decoder_DAG(self.z_dim, self.z1_dim, self.z2_dim, image_size=self.image_size).to(device)
        self.dag = nns.DagLayer(self.z1_dim, self.z1_dim, i = inference).to(device)
        #self.cause = nn.CausalLayer(self.z_dim, self.z1_dim, self.z2_dim)
        self.attn = nns.Attention(self.z2_dim).to(device)
        self.mask_z = nns.MaskLayer(self.z_dim).to(device)
        self.mask_u = nns.MaskLayer(self.z1_dim, z2_dim=1).to(device)

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = nn.Parameter(torch.zeros(1).to(device), requires_grad=False)
        self.z_prior_v = nn.Parameter(torch.ones(1).to(device), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)
        
        self.device = device

    def encode(self, x, label, mask = None, sample = False, adj = None, alpha=0.3, beta=1, lambdav=0.001):
        q_m, q_v = self.enc.encode(x.to(self.device))
        q_m, q_v = q_m.reshape([q_m.size()[0], self.z1_dim,self.z2_dim]),torch.ones(q_m.size()[0], self.z1_dim,self.z2_dim).to(self.device)

        decode_m, decode_v = self.dag.calculate_dag(q_m.to(self.device), torch.ones(q_m.size()[0], self.z1_dim,self.z2_dim).to(self.device))
        decode_m, decode_v = decode_m.reshape([q_m.size()[0], self.z1_dim,self.z2_dim]),decode_v
        if sample == False:
          if mask != None and mask < 2:
              z_mask = torch.ones(q_m.size()[0], self.z1_dim,self.z2_dim).to(self.device)*adj
              decode_m[:, mask, :] = z_mask[:, mask, :]
            #   decode_v[:, mask, :] = z_mask[:, mask, :]
          m_zm, m_zv = self.dag.mask_z(decode_m.to(self.device)).reshape([q_m.size()[0], self.z1_dim,self.z2_dim]),decode_v.reshape([q_m.size()[0], self.z1_dim,self.z2_dim])
          m_u = self.dag.mask_u(label.to(self.device))
          
          f_z = self.mask_z.mix(m_zm).reshape([q_m.size()[0], self.z1_dim,self.z2_dim]).to(self.device)
          e_tilde = self.attn.attention(decode_m.reshape([q_m.size()[0], self.z1_dim,self.z2_dim]).to(self.device),q_m.reshape([q_m.size()[0], self.z1_dim,self.z2_dim]).to(self.device))[0]
          if mask != None and mask < 2:
              z_mask = torch.ones(q_m.size()[0],self.z1_dim,self.z2_dim).to(self.device)*adj
              e_tilde[:, mask, :] = z_mask[:, mask, :]
              
          f_z1 = f_z+e_tilde
          if mask!= None and mask == 2 :
              z_mask = torch.ones(q_m.size()[0],self.z1_dim,self.z2_dim).to(self.device)*adj
              f_z1[:, mask, :] = z_mask[:, mask, :]
            #   m_zv[:, mask, :] = z_mask[:, mask, :]
          if mask!= None and mask == 3 :
              z_mask = torch.ones(q_m.size()[0],self.z1_dim,self.z2_dim).to(self.device)*adj
              f_z1[:, mask, :] = z_mask[:, mask, :]
            #   m_zv[:, mask, :] = z_mask[:, mask, :]
          g_u = self.mask_u.mix(m_u).to(self.device)
          z_given_dag = ut.conditional_sample_gaussian(f_z1, m_zv*lambdav)
          
          ###
          # q_m, q_v: epsilon posterior mean, variance
          # decode_m, decode_v: latent posterior mean, variance
          # f_z1, m_zv*lambdav: masked latent posteior mean, variance
          # z_given_dag: sampled latent variable
          # g_u: masked label information
          
          return q_m, q_v, decode_m, decode_v, f_z, e_tilde, f_z1, m_zv, z_given_dag, g_u

    def decode(self, z_given_dag, label):
        decoded_bernoulli_logits,x1,x2,x3,x4 = self.dec.decode_sep(z_given_dag.reshape([z_given_dag.size()[0], self.z_dim]), label.to(self.device))
        return decoded_bernoulli_logits

    def negative_elbo_bound(self, x, label, mask = None, sample = False, adj = None, alpha=0.3, beta=1, lambdav=0.001):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        assert label.size()[1] == self.z1_dim

        q_m, q_v, decode_m, decode_v, _, _, f_z1, _, z_given_dag, g_u = self.encode(x, label, mask, sample, adj, alpha, beta, lambdav)
        
        decoded_bernoulli_logits = self.decode(z_given_dag, label)
        
        rec = ut.log_bernoulli_with_logits(x, decoded_bernoulli_logits.reshape(x.size()))
        rec = -torch.mean(rec)

        p_m, p_v = torch.zeros(q_m.size()), torch.ones(q_m.size())
        cp_m, cp_v = ut.condition_prior(self.scale, label, self.z2_dim)
        cp_v = torch.ones([q_m.size()[0],self.z1_dim,self.z2_dim]).to(self.device)
        cp_z = ut.conditional_sample_gaussian(cp_m.to(self.device), cp_v.to(self.device))
        kl = torch.zeros(1).to(self.device)
        kl = alpha*ut.kl_normal(q_m.view(-1,self.z_dim).to(self.device), q_v.view(-1,self.z_dim).to(self.device), p_m.view(-1,self.z_dim).to(self.device), p_v.view(-1,self.z_dim).to(self.device))
        
        for i in range(self.z1_dim):
            kl = kl + beta*ut.kl_normal(decode_m[:,i,:].to(self.device), cp_v[:,i,:].to(self.device), cp_m[:,i,:].to(self.device), cp_v[:,i,:].to(self.device))
        kl = torch.mean(kl)
        mask_kl = torch.zeros(1).to(self.device)

        for i in range(4):
            mask_kl = mask_kl + 1*ut.kl_normal(f_z1[:,i,:].to(self.device), cp_v[:,i,:].to(self.device), cp_m[:,i,:].to(self.device), cp_v[:,i,:].to(self.device))
        
        u_loss = torch.nn.MSELoss()
        mask_l = torch.mean(mask_kl) + u_loss(g_u, label.float().to(self.device))
        nelbo = rec + kl + mask_l

        return nelbo, kl, rec, mask_l, decoded_bernoulli_logits.reshape(x.size()), z_given_dag

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec.decode(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        return ut.sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),
            self.z_prior[1].expand(batch, self.z_dim))

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
#%%
# device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
# vae = CausalVAE(device=device)
# print(vae)
# x = torch.randn(10, 3, 64, 64)
# u = torch.randn(10, 4)
# nelbo, kl, rec, decoded_bernoulli_logits, z_given_dag = vae.negative_elbo_bound(x, u)
# decoded_bernoulli_logits.shape
# z_given_dag.shape
#%%