#!/usr/bin/env python


import six

import chainer
import chainer.functions as F
from chainer.functions.loss.vae import gaussian_kl_divergence
import chainer.links as L


class VAE(chainer.Chain):
    """ Variational AutoEncoder"""


    def __init__(self, n_ch, n_latent, n_first):
        super(VAE, self).__init__()
        with self.init_scope():
            # encoder 入力から隠れベクトルの作成
            self.le1 = L.Convolution2D(n_ch, n_first, 4, 2, 1)
            self.le2 = L.Convolution2D(n_first, n_first , 4, 2, 1)
            self.le3 = L.Convolution2D(n_first , n_first , 4, 2, 1)
            self.le4_mu = L.Linear(None, n_latent)
            self.le4_ln_var = L.Linear(None, n_latent)
            # decoder

            self.ld1 = L.Linear(n_latent,16384)
            self.ld2 = L.Deconvolution2D(
                n_first, n_first, 4, 2, 1)
            self.ld3 = L.Deconvolution2D(
                n_first, n_first, 4, 2, 1)
            self.ld4 = L.Deconvolution2D(n_first, 1, 4, 2, 1)
        self.n_first = n_first
       # 400*128/(k1k2k3/2*2*2)
       #k*stride*128*64 = 65536
       #h1 = F.reshape(h1, (batch_size, self.n_first * 4, 16, 16))
        

    def forward(self, x, sigmoid=False):
        """AutoEncoder"""
        return self.decode(self.encode(x), sigmoid)

    def encode(self, x):
        h1 = F.leaky_relu(self.le1(x), slope = 0.2)
        h2 = F.leaky_relu(self.le2(h1), slope = 0.2)
        h3 = F.leaky_relu(self.le3(h2), slope = 0.2)
        mu = self.le4_mu(h3)
        ln_var = self.le4_ln_var(h3)  # log(sigma**2)
        return mu, ln_var

    def decode(self, z, sigmoid=True):
        batch_size = len(z)
        h1 = F.relu(self.ld1(z))
        h1 = F.reshape(h1, (batch_size, self.n_first , 16, 16))
        h2 = F.relu(self.ld2(h1))
        h3 = F.relu(self.ld3(h2))
        h4 = self.ld4(h3)
        if sigmoid:
            return F.sigmoid(h4)
        else:
            return h4

    def get_loss_func(self, beta=1.0, k=1):
        """
        VAEの損失の計算
        ATrue
            C (int): 正則化項をどれだけ効かせるかの変数、通常1.0が使用される
            k (int): サンプルを何回行うか
        """

        def lf(x):
            mu, ln_var = self.encode(x)
            batchsize = len(mu)
            # 復元誤差の計算
            rec_loss = 0
            for l in six.moves.range(k):
                z = F.gaussian(mu, ln_var)
                # rec_loss += F.bernoulli_nll(x, self.decode(z, sigmoid=False)) / (k * batchsize)
                rec_loss += F.bernoulli_nll(
                    x, self.decode(z, sigmoid=False)) / (k * batchsize)
            self.rec_loss = rec_loss
            self.loss = self.rec_loss + \
                beta * gaussian_kl_divergence(mu, ln_var) / batchsize
            chainer.report(
                {'rec_loss': rec_loss, 'loss': self.loss}, observer=self)
            return self.loss
        return lf



        
