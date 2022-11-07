#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 20:07:26 2022

@author: vlourenco
"""

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import L1Loss

beta   = 1e-3

class concat(nn.Module):

    def __init__(self, in_size, output_dim,hidden = 50, dropout=0.5):

        super(concat, self).__init__()
        self.linear_1 = nn.Linear(in_size*3, output_dim)


    def forward(self, l1, a1, v1):
     
        fusion = torch.cat([l1, a1, v1], dim=-1)
        y_1 = torch.relu(self.linear_1(fusion))

        return y_1
    

class fusion(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.d_l = dim
        
        # build encoder
        self.encoder_l = nn.Sequential(nn.Linear(self.d_l, 1024),
                                   #  nn.ReLU(inplace=True),
                                    # nn.Linear(1024, 1024),
                                     nn.ReLU(inplace=True) )  


        self.encoder_a = nn.Sequential(nn.Linear(self.d_l, 1024),
                                   #  nn.ReLU(inplace=True),
                                    # nn.Linear(1024, 1024),
                                     nn.ReLU(inplace=True) )  

        self.encoder_v = nn.Sequential(nn.Linear(self.d_l, 1024),
                                   #  nn.ReLU(inplace=True),
                                    # nn.Linear(1024, 1024),
                                     nn.ReLU(inplace=True) )  
        self.encoder = nn.Sequential(nn.Linear(self.d_l, 1024),
                                   #  nn.ReLU(inplace=True),
                                    # nn.Linear(1024, 1024),
                                     nn.ReLU(inplace=True) )  

        self.fc_mu_l  = nn.Linear(1024, self.d_l) 
        self.fc_std_l = nn.Linear(1024, self.d_l)

        self.fc_mu_a  = nn.Linear(1024, self.d_l) 
        self.fc_std_a = nn.Linear(1024, self.d_l)

        self.fc_mu_v  = nn.Linear(1024, self.d_l) 
        self.fc_std_v = nn.Linear(1024, self.d_l)

        self.fc_mu  = nn.Linear(1024, self.d_l) 
        self.fc_std = nn.Linear(1024, self.d_l)
        
        # build decoder
        self.decoder_l = nn.Linear(self.d_l, 1)
        self.decoder_a = nn.Linear(self.d_l, 1)
        self.decoder_v = nn.Linear(self.d_l, 1)
        self.decoder = nn.Linear(self.d_l, 1)

      #  self.fusion1 = graph_fusion(self.d_l, self.d_l)
        self.fusion1 = concat(self.d_l, self.d_l)
      #  self.fusion1 = tensor(self.d_l, self.d_l)
      #  self.fusion1 = addition(self.d_l, self.d_l)
      #  self.fusion1 = multiplication(self.d_l, self.d_l)
     #   self.fusion1 = low_rank(self.d_l, self.d_l)

    def encode(self, x):
        """
        x : [batch_size,784]
        """
        x = self.encoder(x)
        return self.fc_mu(x), nn.functional.softplus(self.fc_std(x)-5, beta=1)


    def encode_l(self, x):
        """
        x : [batch_size,784]
        """
        x = self.encoder_l(x)
        return self.fc_mu_l(x), nn.functional.softplus(self.fc_std_l(x)-5, beta=1)

    def encode_a(self, x):
        """
        x : [batch_size,784]
        """
        x = self.encoder_a(x)
        return self.fc_mu_a(x), nn.functional.softplus(self.fc_std_a(x)-5, beta=1)

    def encode_v(self, x):
        """
        x : [batch_size,784]
        """
        x = self.encoder_v(x)
        return self.fc_mu_v(x), nn.functional.softplus(self.fc_std_v(x)-5, beta=1)
    
    def decode_l(self, z):
        return self.decoder_l(z)

    def decode_a(self, z):

        return self.decoder_a(z)

    def decode(self, z):

        return self.decoder(z)

    def decode_v(self, z):

        return self.decoder_v(z)
    
    def reparameterise(self, mu, std):
        """
        mu : [batch_size,z_dim]
        std : [batch_size,z_dim]        
        """        
        # get epsilon from standard normal
        eps = torch.randn_like(std)
        return mu + std*eps

    def loss_function(self, y_pred, y, mu, std):
   
        loss_fct = L1Loss()

        CE = loss_fct(y_pred.view(-1,), y.view(-1,))
        KL = 0.5 * torch.mean(mu.pow(2) + std.pow(2) - 2*std.log() - 1)
        return (beta*KL + CE) 




    def forward(
        self,
        x_l,
        x_a,
        x_v,
        label_ids
    ):


        mu_l, std_l = self.encode_l(x_l)
        z_l = self.reparameterise(mu_l, std_l)
        output_l =  self.decode_l(z_l)
 
        loss_l = self.loss_function(output_l, label_ids, mu_l, std_l)

        mu_a, std_a = self.encode_a(x_a)
        z_a = self.reparameterise(mu_a, std_a)
        output_a =  self.decode_a(z_a)
 
        loss_a = self.loss_function(output_a, label_ids, mu_a, std_a)

        mu_v, std_v = self.encode_v(x_v)
        z_v = self.reparameterise(mu_v, std_v)
        output_v =  self.decode_v(z_v)
 
        loss_v = self.loss_function(output_v, label_ids, mu_v, std_v)

       # outputf = torch.cat([z_l, z_a, z_v], dim=-1)

        outputf = self.fusion1(z_l, z_a, z_v)

        mu, std = self.encode(outputf)
        z = self.reparameterise(mu, std)
        output =  self.decode(z)


        loss = self.loss_function(output, label_ids, mu, std)

        return output, loss_l + loss_a + loss_v + loss


    def test(
        self,
        x_l,
        x_a,
        x_v
    ):


        mu_l, std_l = self.encode_l(x_l)
        z_l = self.reparameterise(mu_l, std_l)
        output_l =  self.decode_l(z_l)
 


        mu_a, std_a = self.encode_a(x_a)
        z_a = self.reparameterise(mu_a, std_a)
        output_a =  self.decode_a(z_a)


        mu_v, std_v = self.encode_v(x_v)
        z_v = self.reparameterise(mu_v, std_v)
        output_v =  self.decode_v(z_v)
 
        outputf = self.fusion1(z_l, z_a, z_v)

        mu, std = self.encode(outputf)
        z = self.reparameterise(mu, std)
        output =  self.decode(z)
    


        return output