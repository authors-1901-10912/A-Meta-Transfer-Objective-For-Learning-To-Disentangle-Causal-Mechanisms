# -*- coding: utf-8 -*-
import nauka
import os, sys, time, pdb
import torch
import uuid

from   .models                          import *


class ExperimentBase(nauka.exp.Experiment):
    """
    The base class for all experiments.
    """
    def __init__(self, a):
        self.a = type(a)(**a.__dict__)
        self.a.__dict__.pop("__argp__", None)
        self.a.__dict__.pop("__argv__", None)
        self.a.__dict__.pop("__cls__",  None)
        if self.a.workDir:
            super().__init__(self.a.workDir)
        else:
            projName = "CausalOptimization"
            expNames = [] if self.a.name is None else self.a.name
            workDir  = nauka.fhs.createWorkDir(self.a.baseDir, projName, self.uuid, expNames)
            super().__init__(workDir)
        self.mkdirp(self.logDir)
    
    def reseed(self, password=None):
        """
        Reseed PRNGs for reproducibility at beginning of interval.
        """
        password = password or "Seed: {} Interval: {:d}".format(self.a.seed,
                                                                self.S.intervalNum,)
        nauka.utils.random.setstate           (password)
        nauka.utils.numpy.random.set_state    (password)
        nauka.utils.torch.random.manual_seed  (password)
        nauka.utils.torch.cuda.manual_seed_all(password)
        return self
    
    @property
    def uuid(self):
        u = nauka.utils.pbkdf2int(128, self.name)
        u = uuid.UUID(int=u)
        return str(u)
    @property
    def dataDir(self):
        return self.a.dataDir
    @property
    def logDir(self):
        return os.path.join(self.workDir, "logs")
    @property
    def isDone(self):
        return (self.S.epochNum >= self.a.num_epochs or
               (self.a.fastdebug and self.S.epochNum >= self.a.fastdebug))
    @property
    def exitcode(self):
        return 0 if self.isDone else 1



class Experiment(ExperimentBase):
    """
    Causal experiment.
    """
    
    @property
    def name(self):
        """A unique name containing every attribute that distinguishes this
        experiment from another and no attribute that does not."""
        attrs = [
            self.a.seed,
            self.a.model,
            self.a.num_epochs,
            self.a.batch_size,
            self.a.cuda,
            self.a.fastdebug,
        ]
        return "-".join([str(s) for s in attrs]).replace("/", "_")
    
    def load(self, path):
        self.S = torch.load(os.path.join(path, "snapshot.pkl"))
        return self
    
    def dump(self, path):
        torch.save(self.S,  os.path.join(path, "snapshot.pkl"))
        return self
    
    def fromScratch(self):
        pass
        """Reseed PRNGs for initialization step"""
        self.reseed(password="Seed: {} Init".format(self.a.seed))
        
        """Create snapshottable-state object"""
        self.S = nauka.utils.PlainObject()
        
        """Model Instantiation"""
        self.S.model = None
        if   self.a.model == "cat":
            self.S.model  = CategoricalWorld(self.a)
        elif self.a.model == "cat2":
            self.S.model  = CategoricalWorld2Var(self.a)
        elif self.a.model == "gauss":
            raise NotImplementedError("This isn't actually implemented!!!")
            self.S.model  = GaussianWorld(self.a)
        if   self.S.model is None:
            raise ValueError("Unsupported model \""+self.a.model+"\"!")
        
        if self.a.cuda:
            self.S.model  = self.S.model.cuda(self.a.cuda[0])
        else:
            self.S.model  = self.S.model.cpu()
        
        """Optimizer Selection"""
        self.S.moptimizer = nauka.utils.torch.optim.fromSpec(self.S.model.parameters(),            self.a.model_optimizer)
        self.S.goptimizer = nauka.utils.torch.optim.fromSpec(self.S.model.structural_parameters(), self.a.gamma_optimizer)
        
        """Counters"""
        self.S.epochNum    = 0
        self.S.intervalNum = 0
        
        return self
    
    def run(self):
        """Run by intervals until experiment completion."""
        while not self.isDone:
            self.interval().snapshot().purge()
        return self
    
    def interval(self):
        """
        An interval is defined as the computation- and time-span between two
        snapshots.
        
        Hard requirements:
        - By definition, one may not invoke snapshot() within an interval.
        - Corollary: The work done by an interval is either fully recorded or
          not recorded at all.
        - There must be a step of the event logger between any TensorBoard
          summary log and the end of the interval.
        
        For reproducibility purposes, all PRNGs are reseeded at the beginning
        of every interval.
        """
        
        self.reseed()
        
        
        """Training Loop"""
        self.S.model.train()
        for q in range(self.a.dpe):
            """Distributions Loop"""
            if self.a.fastdebug and q>=self.a.fastdebug: break
            
            self.S.model.alterdists()
            
            
            if self.a.pretrain:
                config = next(self.S.model.configpretrainiter())
                for b, batch in enumerate(self.S.model.sampleiter(self.a.batch_size)):
                    """Pretrain Loop"""
                    if self.a.fastdebug and b>=self.a.fastdebug: break
                    if                      b>=self.a.pretrain:  break
                    
                    self.S.moptimizer.zero_grad()
                    nll = -self.S.model.logprob(batch, config).mean()
                    nll.backward()
                    self.S.moptimizer.step()
                    if self.a.verbose and b % self.a.verbose == 0:
                        print("Pretrain NLL: "+str(nll.item()))
            
            
            for j in range(self.a.ipd):
                """Interventions Loop"""
                if self.a.fastdebug and j>=self.a.fastdebug: break
                
                with self.S.model.saveparamsgt(restore=True):
                    """Intervention"""
                    self.S.model.alterdist()
                    self.S.goptimizer.zero_grad()
                    self.S.model.gamma.grad = torch.zeros_like(self.S.model.gamma)
                    
                    gammagrads = [] # List of R tensors of shape (M,M,) indexed by (i,j)
                    logregrets = [] # List of R tensors of shape (M,)   indexed by (i,)
                    batch      = next(self.S.model.sampleiter(self.a.batch_size))
                    for r, config in enumerate(self.S.model.configiter()):
                        """Configurations Loop"""
                        if self.a.fastdebug and r>=self.a.fastdebug: break
                        if                      r>=self.a.cpi:       break
                        
                        gammagrad = 0
                        logregret = 0
                        with self.S.model.saveparams(restore=True):
                            for t in range(self.a.xfer_epi_size):
                                """Transfer Adaptation Loop"""
                                if self.a.fastdebug and t>=self.a.fastdebug:     break
                                if                      t>=self.a.xfer_epi_size: break
                                
                                self.S.moptimizer.zero_grad()
                                logp = self.S.model.logprob(batch, config)
                                nll  = -logp.mean()
                                nll.backward()
                                self.S.moptimizer.step()
                                
                                with torch.no_grad():
                                    gammagrad += self.S.model.gamma.sigmoid() - config
                                    logregret += logp.mean(0)
                                
                                if self.a.verbose and (j*self.a.xfer_epi_size*self.a.ipd +
                                                       r*self.a.xfer_epi_size +
                                                       t) % self.a.verbose == 0:
                                    pass #print("Train NLL: "+str(nll.item()))
                        
                        gammagrads.append(gammagrad)
                        logregrets.append(logregret)
                    
                    with torch.no_grad():
                        gammagrads = torch.stack(gammagrads)
                        logregrets = torch.stack(logregrets)
                        normregret = logregrets.softmax(0)
                        if self.a.model == "cat2":
                            dRdgamma   = torch.einsum("k,ki->",     gammagrads, normregret)
                        else:
                            dRdgamma   = torch.einsum("kij,ki->ij", gammagrads, normregret)
                        self.S.model.gamma.grad.copy_(dRdgamma)
                    
                    # Gamma Regularizers
                    siggamma = self.S.model.gamma.sigmoid()
                    Lmaxent  = ((siggamma)*(1-siggamma)).sum().mul(-self.a.lmaxent)
                    Lsparse  = siggamma.sum().mul(self.a.lsparse)
                    (Lmaxent + Lsparse).backward()
                    
                    # Gamma Update
                    self.S.goptimizer.step()
                    self.S.model.reconstrain()
                    
                    if self.a.verbose and j % self.a.verbose == 0:
                        with torch.no_grad():
                            # Compute Binary Cross-Entropy over gammas, ignoring diagonal
                            siggamma  = self.S.model.gamma.sigmoid()
                            pospred   = siggamma.clone()
                            negpred   = 1-siggamma.clone()
                            posgt     = self.S.model.gammagt
                            neggt     = 1-self.S.model.gammagt
                            if self.a.model != "cat2":
                                pospred.diagonal().fill_(1)
                                negpred.diagonal().fill_(1)
                            bce      = -pospred.log2_().mul_(posgt) -negpred.log2_().mul_(neggt)
                            bce      = bce.sum()
                            if self.a.model != "cat2":
                                bce.div_(siggamma.numel() - siggamma.diagonal().numel())
                            
                            print("Gamma GT:   "+os.linesep+str(self.S.model.gammagt.detach()))
                            print("Gamma:      "+os.linesep+str(siggamma))
                            print("Gamma Grad: "+os.linesep+str(self.S.model.gamma.grad.detach()))
                            print("Gamma CE:   "+str(bce.item()))
                            print("")
        
        
        """Validation Loop"""
        self.S.model.eval()
        with torch.no_grad():
            pass # Need to implement!!!
        
        
        """Exit"""
        sys.stdout.write("Epoch {:d} done.\n".format(self.S.epochNum))
        self.S.epochNum    += 1
        self.S.intervalNum += 1
        return self
