import sys
sys.path.insert(0,"../")

import numpy as np
import pandas as pd
import optax
from optax.schedules import warmup_cosine_decay_schedule

from utils import Sampler,TASExperiment

from src.kernel import TSGaussGaussKernel,VFTSGaussKernel
from src.lddmm import Shooting
from src.utils import time_shape_embedding
from src.loss import SumVarifoldLoss

functional_signal = lambda x:  np.sin(2*np.pi*x)
#time series to time series'graph 
q0 = time_shape_embedding(functional_signal(np.linspace(0,1,300)).reshape(-1,1),)
q0_mask = np.full_like(q0[:,:1],True).astype(bool) #mask is important when time series have different length

Kv = VFTSGaussKernel(1,0.1,100,1,1) #return the kernel functions related to the prescribed hyper parameter
shoot = Shooting(Kv)# shooting functions taking p_0,q_0,q_0_mask

spl = Sampler(q0,q0_mask,shoot,[10],[10],[50],0) #t_amp (norme pertubation temporel), s_amp (norme pertubation spatiale),smoothness, random state
ps,qs,df= spl.rvs(50) #sampling de perturvation
qs_mask = np.full_like(qs[:,:,:1],True).astype(bool)
Kls = [TSGaussGaussKernel(2,1,2,0.6)] #kernel varifold
dataloss = SumVarifoldLoss(Kls)

tasexp = TASExperiment(1,0.1,[1,5,10,50,100,200,300],1,[0.1,1,10,100],q0,q0_mask,dataloss,20,gamma_loss=0,niter=400,optimizer=optax.adabelief(warmup_cosine_decay_schedule(0,0.05,40,400,0)))
tasexp.fit(ps,qs,qs_mask)
tasexp.df_.to_csv("./results/misspecified.csv")