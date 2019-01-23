# -*- coding: utf-8 -*-
import contextlib
import os, sys, time, pdb
import numpy                                as np
import torch
import torch.nn


from   torch.distributions              import (OneHotCategorical)
from   torch.nn                         import (Module,Parameter,)




class World(Module):
    @contextlib.contextmanager
    def saveparams(self, restore=False):
        """
        Copy learned parameters and optionally restore them.
        """
        
        if restore:
            with torch.no_grad():
                clones = [t.clone() for t in self.parameters()]
        yield
        if restore:
            with torch.no_grad():
                for tnew, told in zip(self.parameters(), clones):
                    tnew.copy_(told)
    
    @contextlib.contextmanager
    def saveparamsgt(self, restore=False):
        """
        Copy ground-truth parameters and optionally restore them.
        """
        
        if restore:
            with torch.no_grad():
                clones = [t.clone() for t in self.parameters_gt()]
        yield
        if restore:
            with torch.no_grad():
                for tnew, told in zip(self.parameters_gt(), clones):
                    tnew.copy_(told)
    
    def parameters(self):
        s = set(self.structural_parameters())
        l = [p for p in super().parameters() if p not in s]
        return iter(l)
    
    def structural_parameters(self):
        return iter([self.gamma])



class CategoricalWorld(World):
    def __init__(self, a, *args):
        super().__init__()
        
        self.M = a.num_vars
        self.N = a.num_cats
        
        self.register_parameter("W0",      Parameter(torch.FloatTensor(self.M, self.H, self.M, self.N)))
        self.register_parameter("B0",      Parameter(torch.FloatTensor(self.M, self.H                )))
        self.register_parameter("W1",      Parameter(torch.FloatTensor(self.M, self.N, self.H        )))
        self.register_parameter("B1",      Parameter(torch.FloatTensor(self.M, self.N                )))
        self.register_parameter("gamma",   Parameter(torch.FloatTensor(self.M, self.M                )))
        
        self.register_buffer   ("W0gt",    torch.FloatTensor(self.M, self.H, self.M, self.N))
        self.register_buffer   ("B0gt",    torch.FloatTensor(self.M, self.H                ))
        self.register_buffer   ("W1gt",    torch.FloatTensor(self.M, self.N, self.H        ))
        self.register_buffer   ("B1gt",    torch.FloatTensor(self.M, self.N                ))
        self.register_buffer   ("gammagt", torch.zeros_like(self.gamma))
        
        for i in range(self.M):
            torch.nn.init.kaiming_normal_(self.W0[i])
            torch.nn.init.kaiming_normal_(self.W1[i])
        torch.nn.init.uniform_(self.B0,    -.1, +.1)
        torch.nn.init.uniform_(self.B1,    -.1, +.1)
        torch.nn.init.uniform_(self.gamma, -.1, +.1)
        with torch.no_grad(): self.gamma.diagonal().fill_(float("-inf"))
        
        for i in range(self.M):
            torch.nn.init.kaiming_normal_(self.W0gt[i])
            torch.nn.init.kaiming_normal_(self.W1gt[i])
        torch.nn.init.uniform_(self.B0gt,  -1, +1)
        torch.nn.init.uniform_(self.B1gt,  -1, +1)
        self.initgraph() # self.gammagt
    
    @property
    def H(self):
        if self.M > self.N: return 4*self.M
        else:               return 4*self.N
    
    def initgraph(self):
        expParents = 5
        idx        = np.arange(self.M).astype(np.float32)[:,np.newaxis]
        idx_maxed  = np.minimum(idx*0.5, expParents)
        p          = np.broadcast_to(idx_maxed/(idx+1), (self.M, self.M))
        B          = np.random.binomial(1, p)
        B          = np.tril(B, -1)
        self.gammagt.copy_(torch.as_tensor(B))
        return self
    
    def alteredges(self):
        if self.M <= 1: return
        v1y = np.random.randint(1, self.M)
        v2y = np.random.randint(1, self.M)
        v1x = np.random.randint(0, v1y)
        v2x = np.random.randint(0, v2y)
        v1e = not self.gammagt[v1y,:v1y].byte().any().item()
        v1f =     self.gammagt[v1y,:v1y].byte().all().item()
        v2e = not self.gammagt[v2y,:v2y].byte().any().item()
        v2f =     self.gammagt[v2y,:v2y].byte().all().item()
        with torch.no_grad():
            if   v1e and v2e:
                self.gammagt[v1y,v1x] = 1
            elif v1f and v2f:
                self.gammagt[v2y,v2x] = 0
            elif v1e:
                while not self.gammagt[v2y,v2x].item():
                    v2x = (v2x+1) % v2y
                self.swap(v1y,v1x,v2y,v2x)
            elif v1f:
                while     self.gammagt[v2y,v2x].item():
                    v2x = (v2x+1) % v2y
                self.swap(v1y,v1x,v2y,v2x)
            else:
                while     self.gammagt[v1y,v1x].eq(self.gammagt[v2y,v2x]).item():
                    v1x = (v1x+1) % v1y
                self.swap(v1y,v1x,v2y,v2x)
        return self
    
    def alterdists(self):
        for i in range(self.M):
            torch.nn.init.kaiming_normal_(self.W0gt[i])
            torch.nn.init.kaiming_normal_(self.W1gt[i])
            torch.nn.init.uniform_(self.B0gt[i], -.2, +.2)
            torch.nn.init.uniform_(self.B1gt[i], -.2, +.2)
        return self
    
    def alterdist(self, i=None):
        i = np.random.randint(0, self.M)
        self.B1gt[i].zero_()
        self.W1gt[i].zero_()
        self.W1gt[i, np.random.randint(0, self.N)] = 100
        return self
    
    def swap(self, v1y, v1x, v2y, v2x):
        t = self.gammagt[v1y,v1x]
        self.gammagt[v1y,v1x] = self.gammagt[v2y,v2x]
        self.gammagt[v2y,v2x] = t
        return self
    
    def configpretrainiter(self):
        """
        Sample a configuration for pretraining.
        
        For pretraining, this matrix is all-to-all connected.
        """
        
        while True:
            gammaexp = torch.ones_like(self.gamma)
            gammaexp.diagonal().zero_()
            yield gammaexp
    
    def configiter(self):
        """Sample a configuration from this world."""
        while True:
            with torch.no_grad():
                gammaexp = self.gamma.sigmoid()
                gammaexp = torch.empty_like(gammaexp).uniform_().lt_(gammaexp)
                gammaexp.diagonal().zero_()
            yield gammaexp
    
    def sampleiter(self, bs=1):
        """
        Ancestral sampling with MLP.
        
        1 sample is a tensor (1, M, N).
        A minibatch of samples is a tensor (bs, M, N).
        1 variable is a tensor (bs, 1, N)
        """
        while True:
            with torch.no_grad():
                h = []   # Hard (onehot) samples  (bs,1,N)
                for i in range(self.M):
                    O = torch.zeros(bs, self.M-i, self.N)   # (bs,M-i,N)
                    v = torch.cat(h+[O], dim=1)             # (bs,M-i,N) + (bs,1,N)*i
                    v = torch.einsum("hik,i,bik->bh", self.W0gt[i], self.gammagt[i], v)
                    v = v + self.B0gt[i].unsqueeze(0)
                    v = v.relu()
                    v = torch.einsum("oh,bh->bo",     self.W1gt[i], v)
                    v = v + self.B1gt[i].unsqueeze(0)
                    v = v.softmax(dim=1).unsqueeze(1)
                    h.append(OneHotCategorical(v).sample())
                s = torch.cat(h, dim=1)
            yield s
    
    def logprob(self, sample, config):
        """
        Log-probability of sample variables given sampled configuration.
        input  sample = (bs, M, N)  # Actual value of the sample
        input  config = (M, M)      # Configuration
        return logprob = (bs, M)
        """
        
        v = torch.einsum("ihjk,ij,bjk->bih", self.W0, config, sample)
        v = v + self.B0.unsqueeze(0)
        v = v.relu()
        v = torch.einsum("ioh,bih->bio",     self.W1, v)
        v = v + self.B1.unsqueeze(0)
        v = v.log_softmax(dim=2)
        v = torch.einsum("bio,bio->bi", v, sample)
        return v
    
    def dLdgamma(self, sample, config):
        """
        sample = (bs, M, N)     # Actual value of the sample
        config = (M, M)         # Configuration
        
        gamma  = (M,M)
        logprob = (bs, M)
        return g_ij = (M,M)
        """
        
        siggamma = self.gamma.sigmoid().unsqueeze(0)         # (1,  M, M)
        logp     = self.logprob(sample, config).unsqueeze(2) # (bs, M, 1)
        g_Bij    = (siggamma-config)*logp
        return g_Bij.mean(0)
    
    def forward(self, sample, config):
        """Returns the NLL of the samples under the given configuration"""
        return self.logprob(sample, config)
    
    def reconstrain(self):
        with torch.no_grad():
            self.gamma.clamp_(-5,+5)
            self.gamma.diagonal().fill_(float("-inf"))
    
    def parameters_gt(self):
        return iter([self.W0gt, self.B0gt, self.W1gt, self.B1gt])


class CategoricalWorld2Var(World):
    def __init__(self, a, *args):
        super().__init__()
        
        self.N = a.num_cats
        
        self.register_parameter("pA",      Parameter(torch.FloatTensor(self.N)))
        self.register_parameter("pB",      Parameter(torch.FloatTensor(self.N)))
        self.register_parameter("pAtoB",   Parameter(torch.FloatTensor(self.N, self.N)))
        self.register_parameter("pBtoA",   Parameter(torch.FloatTensor(self.N, self.N)))
        self.register_parameter("gamma",   Parameter(torch.as_tensor(0.0, dtype=torch.float32)))
        
        self.register_buffer   ("pAgt",    torch.FloatTensor(self.N))
        self.register_buffer   ("pAtoBgt", torch.FloatTensor(self.N, self.N))
        self.register_buffer   ("gammagt", torch.as_tensor(1.0, dtype=torch.float32))
        
        torch.nn.init.uniform_(self.pA,      -2, +2)
        torch.nn.init.uniform_(self.pB,      -2, +2)
        torch.nn.init.uniform_(self.pAtoB,   -2, +2)
        torch.nn.init.uniform_(self.pBtoA,   -2, +2)
        
        torch.nn.init.uniform_(self.pAgt,    -2, +2)
        torch.nn.init.uniform_(self.pAtoBgt, -2, +2)
    
    def alterdists(self):
        return self.alterdist()
        
    def alterdist(self):
        torch.nn.init.uniform_(self.pAgt,    -2, +2)
        return self
    
    def configpretrainiter(self):
        """
        Sample a configuration for pretraining.
        
        For pretraining, this matrix is all-to-all connected.
        """
        
        while True:
            yield torch.ones_like(self.gamma)
    
    def configiter(self):
        """Sample a configuration from this world."""
        while True:
            with torch.no_grad():
                gammaexp = self.gamma.sigmoid()
                gammaexp = torch.empty_like(gammaexp).uniform_().lt_(gammaexp)
            yield gammaexp
    
    def sampleiter(self, bs=1):
        """
        Ancestral sampling with probability tables.
        
        1 sample is a tensor (1, 2*N).
        A minibatch of samples is a tensor (bs, 2*N).
        1 variable is a tensor (bs, N)
        """
        while True:
            with torch.no_grad():
                pA = self.pAgt.softmax(dim=0).expand(bs,-1)
                a  = OneHotCategorical(pA).sample()
                pB = torch.einsum("ij,bi->bj", self.pAtoBgt.softmax(dim=1), a)
                b  = OneHotCategorical(pB).sample()
                s  = torch.cat([a,b], dim=1)
            yield s
    
    def logprob(self, sample, config):
        """
        Log-probability of sample variables given sampled configuration.
        input  sample = (bs, 2*N)  # Actual value of the sample
        input  config = ()         # Configuration
        return logprob = (bs, 2)
        """
        
        A      = sample[:,:self.N]
        B      = sample[:,self.N:]
        lpA    = self.pA.log_softmax(dim=0)
        lpB    = self.pB.log_softmax(dim=0)
        lpAtoB = self.pAtoB.log_softmax(dim=1)
        lpBtoA = self.pBtoA.log_softmax(dim=1)
        llA    = torch.einsum("i,bi->b",     lpA,    A)
        llB    = torch.einsum("j,bj->b",     lpB,    B)
        llAtoB = torch.einsum("ij,bi,bj->b", lpAtoB, A, B)
        llBtoA = torch.einsum("ji,bj,bi->b", lpBtoA, B, A)
        vAAtoB = torch.stack([llA, llAtoB], dim=1)
        vBBtoA = torch.stack([llB, llBtoA], dim=1)
        return config*vAAtoB + (1-config)*vBBtoA
    
    def dLdgamma(self, sample, config):
        """
        sample = (bs, 2*N)  # Actual value of the sample
        config = ()         # Configuration
        
        gamma  = ()
        logprob = (bs, 2)
        return g = ()
        """
        
        siggamma = self.gamma.sigmoid().unsqueeze(0)  # (1)
        logp     = self.logprob(sample, config)       # (bs, 2)
        g_Bi     = (siggamma-config)*logp
        return g_Bi.sum(1).mean()
    
    def forward(self, sample, config):
        """Returns the NLL of the samples under the given configuration"""
        return self.logprob(sample, config)
    
    def reconstrain(self):
        with torch.no_grad():
            self.gamma.clamp_(-5,+5)
    
    def parameters_gt(self):
        return iter([self.pAgt, self.pAtoBgt])


if __name__ == "__main__":
    class PlainObject(object): pass
    o = PlainObject()
    o.num_vars = 100
    o.num_cats = 10
    w = CategoricalWorld(o)
    i = w.sampleiter()
    c = w.configiter()
    print(w.dLdgamma(next(i), next(c)))
